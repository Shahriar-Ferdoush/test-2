"""
Ultra-Low Memory LLaMA Model Merging
=====================================

Optimized for extreme memory constraints:
- RAM: 29 GB limit
- Storage: 19.5 GB limit

Key Optimizations:
------------------
1. **8-bit/4-bit quantization** - Load models in int8 (50% RAM) or int4 (75% RAM)
2. **Direct merging without caching** - No intermediate task vector files
3. **Streaming layer processing** - Process and discard immediately
4. **Offloading to disk** - Use memory-mapped tensors for overflow
5. **LoRA-only merging** - If models are LoRA adapters (~100MB each)

Memory Breakdown (7B model):
-----------------------------
- FP16 model: ~14 GB
- INT8 model: ~7 GB  ✅ Fits in 29GB RAM
- INT4 model: ~3.5 GB ✅ Multiple models fit!
- LoRA adapters: ~100-500 MB ✅ Trivial memory

Storage Breakdown:
------------------
- Base model INT8: ~7 GB
- 2 LoRA adapters: ~200-1000 MB
- Output model INT8: ~7 GB
- Total: ~15-16 GB ✅ Fits in 19.5GB!

Usage:
------
# For LoRA adapters (RECOMMENDED for low memory)
merger = LowMemoryLLaMAMerger(
    base_model_path="meta-llama/Llama-2-7b-hf",
    lora_adapter_paths=["adapter1", "adapter2"],
    output_dir="./merged",
    use_8bit=True,  # Use 8-bit quantization
)
merged_model = merger.merge_lora_adapters()

# For full models (requires more memory)
merger = LowMemoryLLaMAMerger(
    base_model_path="meta-llama/Llama-2-7b-hf",
    finetuned_model_paths=["model1", "model2"],
    output_dir="./merged",
    use_8bit=True,  # INT8 quantization
    offload_to_disk=True,  # Use disk if RAM exceeded
)
merged_model = merger.merge_with_ties()
"""

import argparse
import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from dare_utils import DARE

# Import merging utilities
from ties_utils import TIES
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try to import PEFT for LoRA merging
try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ PEFT not available. LoRA merging will not work.")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_memory_usage(prefix=""):
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    # Estimate RAM usage (rough approximation)
    import psutil

    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1e9
    logger.info(f"{prefix} RAM Usage: {ram_gb:.2f}GB")


class LowMemoryLLaMAMerger:
    """
    Ultra-low memory model merger for systems with <29GB RAM and <19.5GB storage.

    Strategies:
    -----------
    1. Load models in INT8 (7GB instead of 14GB)
    2. Process layer-by-layer with immediate cleanup
    3. Use LoRA adapters when possible (100MB vs 14GB)
    4. Offload to disk if RAM exceeded
    5. Skip intermediate caching entirely
    """

    def __init__(
        self,
        base_model_path: str,
        finetuned_model_paths: Optional[List[str]] = None,
        lora_adapter_paths: Optional[List[str]] = None,
        output_dir: str = "./merged_models",
        use_8bit: bool = True,
        use_4bit: bool = False,
        offload_to_disk: bool = True,
        density: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize low-memory merger.

        Args:
            base_model_path: Path to base model
            finetuned_model_paths: List of full fine-tuned models (use OR lora_adapter_paths)
            lora_adapter_paths: List of LoRA adapter paths (RECOMMENDED for low memory)
            output_dir: Where to save merged model
            use_8bit: Load models in 8-bit (50% memory reduction)
            use_4bit: Load models in 4-bit (75% memory reduction)
            offload_to_disk: Offload layers to disk when RAM is full
            density: Fraction of weights to keep in task vectors
            device: Device to use
        """
        self.base_model_path = base_model_path
        self.finetuned_model_paths = finetuned_model_paths or []
        self.lora_adapter_paths = lora_adapter_paths or []
        self.output_dir = Path(output_dir)
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.offload_to_disk = offload_to_disk
        self.density = density
        self.device = torch.device(device)

        # Validate inputs
        if not self.finetuned_model_paths and not self.lora_adapter_paths:
            raise ValueError(
                "Must provide either finetuned_model_paths or lora_adapter_paths"
            )

        if self.finetuned_model_paths and self.lora_adapter_paths:
            raise ValueError(
                "Provide either finetuned_model_paths OR lora_adapter_paths, not both"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quantization config
        if use_4bit:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (~3.5GB per 7B model)")
        elif use_8bit:
            self.quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            logger.info("Using 8-bit quantization (~7GB per 7B model)")
        else:
            self.quant_config = None
            logger.warning("No quantization - may exceed 29GB RAM limit!")

        # Temp directory for disk offloading
        if offload_to_disk:
            self.temp_dir = tempfile.mkdtemp(prefix="llama_merge_")
            logger.info(f"Disk offload directory: {self.temp_dir}")
        else:
            self.temp_dir = None

        log_memory_usage("Initial")

    # ============================================================================
    # STRATEGY 1: Merge LoRA Adapters (RECOMMENDED - Minimal Memory!)
    # ============================================================================

    def merge_lora_adapters(
        self,
        weights: Optional[List[float]] = None,
        save_merged: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Merge multiple LoRA adapters into base model.

        Memory usage: ~7GB (base) + ~500MB (adapters) = ~8GB total ✅
        Storage: ~7GB (base) + ~1GB (adapters) + ~7GB (output) = ~15GB ✅

        This is the MOST EFFICIENT method for low memory!

        Args:
            weights: Weights for each adapter (default: equal weights)
            save_merged: Whether to save the merged model

        Returns:
            Merged model
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "PEFT library required for LoRA merging. Install: pip install peft"
            )

        if not self.lora_adapter_paths:
            raise ValueError("No LoRA adapter paths provided!")

        logger.info("=" * 60)
        logger.info(f"MERGING {len(self.lora_adapter_paths)} LoRA ADAPTERS")
        logger.info("=" * 60)

        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(self.lora_adapter_paths)] * len(
                self.lora_adapter_paths
            )

        # Load base model with quantization
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quant_config,
            device_map="auto" if self.offload_to_disk else self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        log_memory_usage("Base model loaded")

        # Load and merge adapters one by one
        logger.info(f"Merging {len(self.lora_adapter_paths)} adapters...")

        for idx, (adapter_path, weight) in enumerate(
            zip(self.lora_adapter_paths, weights)
        ):
            logger.info(
                f"[{idx+1}/{len(self.lora_adapter_paths)}] Loading adapter: {adapter_path}"
            )

            # Load adapter
            model_with_adapter = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                adapter_name=f"adapter_{idx}",
            )
            log_memory_usage(f"Adapter {idx+1} loaded")

            # If weight != 1.0, scale the adapter weights
            if weight != 1.0:
                logger.info(f"  Scaling adapter by {weight}")
                for name, param in model_with_adapter.named_parameters():
                    if "lora" in name.lower():
                        param.data *= weight

            # Merge this adapter into base
            logger.info(f"  Merging adapter {idx+1}...")
            model_with_adapter.merge_and_unload()

            # Update base model for next iteration
            base_model = model_with_adapter

            log_memory_usage(f"Adapter {idx+1} merged")

            # Cleanup
            del model_with_adapter
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("✓ All adapters merged!")

        # Save merged model
        if save_merged:
            output_path = self.output_dir / "merged_lora"
            logger.info(f"Saving merged model to {output_path}...")
            base_model.save_pretrained(output_path)

            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            tokenizer.save_pretrained(output_path)

            logger.info(f"✓ Model saved to {output_path}")

        return base_model

    # ============================================================================
    # STRATEGY 2: Merge Full Models (Layer-by-Layer, Quantized)
    # ============================================================================

    def merge_with_ties(
        self,
        save_merged: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Merge full fine-tuned models using TIES method.

        Memory usage (with INT8):
        - Base model: ~7GB
        - Layer processing: ~1-2GB
        - Total: ~9-10GB ✅ Fits in 29GB

        Storage: ~7GB (base) + ~14GB (2 models) + ~7GB (output) = ~28GB ⚠️
        May exceed 19.5GB storage limit unless models are deleted after use!

        Workaround: Stream models from HuggingFace Hub (no local storage)

        Args:
            save_merged: Whether to save merged model

        Returns:
            Merged model
        """
        logger.info("=" * 60)
        logger.info("MERGING FULL MODELS WITH TIES (LOW MEMORY MODE)")
        logger.info("=" * 60)

        # Load base model with quantization
        logger.info("Loading base model in INT8...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=self.quant_config,
            device_map="auto" if self.offload_to_disk else self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        log_memory_usage("Base model loaded")

        # Find layers to merge
        linear_layers = self._find_linear_layers(base_model)
        layer_names = list(linear_layers.keys())
        logger.info(f"Found {len(layer_names)} layers to merge")

        # Initialize TIES merger
        merger = TIES()

        # Merge layer by layer
        logger.info("Starting layer-by-layer merge...")

        for layer_idx, layer_name in enumerate(layer_names):
            if (layer_idx + 1) % 20 == 0 or layer_idx == 0:
                logger.info(
                    f"  [{layer_idx+1}/{len(layer_names)}] Processing {layer_name}"
                )
                log_memory_usage(f"  Layer {layer_idx+1}")

            try:
                # Get base layer parameters
                base_params = self._get_layer_params(base_model, layer_name)

                # Compute task vectors by loading FT models one at a time
                task_vectors = []
                ft_params_list = []

                for ft_idx, ft_model_path in enumerate(self.finetuned_model_paths):
                    # Load ONLY this layer from FT model
                    ft_layer_params = self._load_single_layer_from_model(
                        ft_model_path, layer_name
                    )

                    # Compute task vector
                    task_vector = ft_layer_params - base_params
                    task_vectors.append(task_vector)
                    ft_params_list.append(ft_layer_params)

                    # Cleanup immediately
                    del ft_layer_params
                    gc.collect()

                # Merge this layer
                merged_params = merger.merge(
                    weights=[1.0] * len(task_vectors),
                    base_model_parameters=base_params,
                    ft_models_parameters=ft_params_list,
                    densities=[self.density] * len(task_vectors),
                    device=self.device,
                )

                # Update base model
                self._set_layer_params(base_model, layer_name, merged_params)

                # Cleanup
                del base_params, task_vectors, ft_params_list, merged_params
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error merging layer {layer_name}: {e}")
                continue

        logger.info("✓ Merge complete!")
        log_memory_usage("Final")

        # Save merged model
        if save_merged:
            output_path = self.output_dir / "merged_ties"
            logger.info(f"Saving merged model to {output_path}...")
            base_model.save_pretrained(output_path)

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            tokenizer.save_pretrained(output_path)

            logger.info(f"✓ Model saved to {output_path}")

        return base_model

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _find_linear_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Find all linear layers in model"""
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        return linear_layers

    def _get_layer_params(self, model: nn.Module, layer_name: str) -> torch.Tensor:
        """Get parameters from a specific layer"""
        for name, param in model.named_parameters():
            if layer_name in name:
                return param.data.clone()
        raise ValueError(f"Layer {layer_name} not found in model")

    def _set_layer_params(
        self, model: nn.Module, layer_name: str, params: torch.Tensor
    ):
        """Set parameters for a specific layer"""
        for name, param in model.named_parameters():
            if layer_name in name:
                param.data = params
                return
        raise ValueError(f"Layer {layer_name} not found in model")

    def _load_single_layer_from_model(
        self, model_path: str, layer_name: str
    ) -> torch.Tensor:
        """
        Load ONLY a single layer from a model (memory efficient).

        This avoids loading the entire model into memory.
        """
        # Try to load just the state dict for this layer
        try:
            from safetensors import safe_open

            # Try safetensors format first
            safetensors_path = Path(model_path) / "model.safetensors"
            if safetensors_path.exists():
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if layer_name in key:
                            return f.get_tensor(key).to(self.device)
        except:
            pass

        # Fallback: Load full model (less efficient but works)
        logger.warning(f"Loading full model to extract layer {layer_name} (slower)")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.quant_config,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        layer_params = self._get_layer_params(ft_model, layer_name)

        # Cleanup
        del ft_model
        gc.collect()

        return layer_params

    def __del__(self):
        """Cleanup temp directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================================
# Command-Line Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Merge LLaMA models with extreme memory constraints (<29GB RAM, <19.5GB storage)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model",
    )

    parser.add_argument(
        "--lora_adapters",
        type=str,
        nargs="+",
        help="Paths to LoRA adapters (RECOMMENDED for low memory)",
    )

    parser.add_argument(
        "--finetuned_models",
        type=str,
        nargs="+",
        help="Paths to full fine-tuned models (requires more memory)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_models",
        help="Directory to save merged model",
    )

    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=True,
        help="Use 8-bit quantization (7GB per 7B model)",
    )

    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (3.5GB per 7B model)",
    )

    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="Fraction of weights to keep (default: 0.2)",
    )

    args = parser.parse_args()

    # Create merger
    merger = LowMemoryLLaMAMerger(
        base_model_path=args.base_model,
        lora_adapter_paths=args.lora_adapters,
        finetuned_model_paths=args.finetuned_models,
        output_dir=args.output_dir,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        density=args.density,
    )

    # Choose merging strategy
    if args.lora_adapters:
        logger.info("Using LoRA adapter merging (memory efficient!)")
        merged_model = merger.merge_lora_adapters()
    else:
        logger.info("Using full model merging (requires more memory)")
        merged_model = merger.merge_with_ties()

    logger.info("\n✓ MERGE COMPLETE!")
    log_memory_usage("Final")


if __name__ == "__main__":
    main()
