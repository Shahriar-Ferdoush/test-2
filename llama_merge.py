"""
LLaMA Model Merging with Memory-Efficient Processing

This script implements three merging methods for fine-tuned LLaMA models:
1. Simple TIES merging (magnitude-based trimming)
2. Simple DARE merging (random dropout)
3. SparseGPT-based merging (Hessian importance)

Memory Management Strategy:
--------------------------
To avoid OOM errors when merging multiple large models:
1. Compute task vectors ONE MODEL AT A TIME and save to disk
2. Compute Hessians ONE MODEL AT A TIME and save to disk
3. Merge LAYER BY LAYER, loading only necessary tensors

This allows merging multiple 7B+ models on limited GPU memory.

Usage:
-----
python llama_merge.py --config config.yaml

Or programmatically:
    merger = LLaMAMerger(
        base_model_path="meta-llama/Llama-3.2-1B",
        finetuned_model_paths=["model1", "model2"],
        dataset_names=["dataset1", "dataset2"],
        output_dir="./merged_models"
    )
    merger.merge_all_methods()
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import our merging utilities
from dare_utils import DARE
from datasets import load_dataset
from peft import PeftModel
from sparsegpt_importance import (
    HessianCalculator,
    compute_importance_scores,
    generate_importance_mask,
)
from ties_utils import TIES
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Helper function for Kaggle-visible output
def log_print(msg, level="INFO"):
    """Print to stdout (visible in Kaggle) and log"""
    print(msg, flush=True)  # flush=True ensures immediate output in Kaggle
    if level == "INFO":
        logger.info(msg)
    elif level == "WARNING":
        logger.warning(msg)
    elif level == "ERROR":
        logger.error(msg)


class LLaMAMerger:
    """
    Memory-efficient LLaMA model merger with three methods:
    - TIES (magnitude-based trimming)
    - DARE (random dropout)
    - SparseGPT (Hessian-based importance)
    """

    def __init__(
        self,
        base_model_path: str,
        finetuned_model_paths: List[str],
        dataset_names: List[str],
        output_dir: str = "./merged_models",
        cache_dir: str = "./merge_cache",
        num_calibration_samples: int = 128,
        calibration_seq_length: int = 512,
        density: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize LLaMA merger.

        Args:
            base_model_path: Path to base (pretrained) model
            finetuned_model_paths: List of paths to fine-tuned models
                                   Can be HuggingFace model IDs or local paths
            dataset_names: List of dataset names for calibration
                          One per fine-tuned model
            output_dir: Directory to save merged models
            cache_dir: Directory for intermediate files (task vectors, Hessians)
            num_calibration_samples: Number of samples for Hessian computation
            calibration_seq_length: Sequence length for calibration
            density: Fraction of weights to keep (0.2 = 20%)
            device: Device to use ('cuda' or 'cpu')
        """
        self.base_model_path = base_model_path
        self.finetuned_model_paths = finetuned_model_paths
        self.dataset_names = dataset_names
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.num_calibration_samples = num_calibration_samples
        self.calibration_seq_length = calibration_seq_length
        self.density = density
        self.device = torch.device(device)

        # Validate inputs
        if len(finetuned_model_paths) != len(dataset_names):
            raise ValueError(
                f"Number of models ({len(finetuned_model_paths)}) must match "
                f"number of datasets ({len(dataset_names)})"
            )

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache subdirectories
        # NOTE: Task vectors are temporary (deleted after mask generation)
        # Only importance masks are kept long-term (~20MB per model)
        self.task_vector_dir = self.cache_dir / "task_vectors"
        self.task_vector_dir.mkdir(exist_ok=True)

        self.mask_dir = self.cache_dir / "importance_masks"
        self.mask_dir.mkdir(exist_ok=True)

        log_print(f"Initialized LLaMAMerger with {len(finetuned_model_paths)} models")
        log_print(f"Device: {self.device}")
        log_print(f"Density: {self.density} (keep {self.density*100}% of weights)")
        log_print(
            f"Memory-efficient: Task vectors cached temporarily, deleted after use"
        )
        log_print(
            f"Final cache size: ~{len(finetuned_model_paths) * 20}MB (masks only)"
        )

    # ============================================================================
    # STORAGE-OPTIMIZED APPROACH
    # ============================================================================
    # Instead of:
    #   1. Save full task vectors (6GB each)
    #   2. Save full Hessians (6GB each)
    #   3. Load both for merging
    #
    # We now:
    #   1. Compute task vectors on-demand (no storage)
    #   2. Compute importance masks and cache (100MB each)
    #   3. Apply masks during merging
    #
    # Storage savings: 24GB → ~400MB (98% reduction!)
    # ============================================================================

    # ============================================================================
    # STEP 1: Load Calibration Data
    # ============================================================================

    def load_calibration_data(
        self, dataset_name: str, tokenizer: AutoTokenizer
    ) -> List[torch.Tensor]:
        """
        Load and prepare calibration data from dataset.

        Args:
            dataset_name: Name of dataset to load from HuggingFace
            tokenizer: Tokenizer for encoding text

        Returns:
            List of tokenized input tensors, each of shape [1, seq_len]
        """
        logger.info(f"Loading calibration data from {dataset_name}...")

        try:
            # Load dataset - only first N samples to avoid downloading entire dataset
            log_print(
                f"  Loading dataset split (first {self.num_calibration_samples} samples)..."
            )
            dataset = load_dataset(
                dataset_name, split=f"train[:{self.num_calibration_samples}]"
            )
            log_print(f"  ✓ Dataset loaded: {len(dataset)} samples")

            # Get text field (common names: 'text', 'content', 'conversation', etc.)
            text_field = self._detect_text_field(dataset)
            logger.info(f"  Using text field: '{text_field}'")

            calibration_data = []

            # Process samples
            log_print(f"  Tokenizing samples...")
            for i in range(min(self.num_calibration_samples, len(dataset))):
                if (i + 1) % 20 == 0:
                    log_print(
                        f"    Progress: {i+1}/{min(self.num_calibration_samples, len(dataset))} samples tokenized"
                    )
                try:
                    text = dataset[i][text_field]

                    # Handle conversation format
                    if isinstance(text, list):
                        text = " ".join(text)

                    # Tokenize
                    tokens = tokenizer(
                        text,
                        max_length=self.calibration_seq_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    calibration_data.append(tokens["input_ids"])

                except Exception as e:
                    logger.warning(f"    Skipping sample {i}: {e}")
                    continue

            log_print(f"  ✓ Loaded {len(calibration_data)} calibration samples")
            return calibration_data

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.info("Creating dummy calibration data as fallback...")
            return self._create_dummy_calibration_data(tokenizer)

    def _detect_text_field(self, dataset) -> str:
        """Detect which field contains text data."""
        possible_fields = ["text", "content", "conversation", "messages", "prompt"]
        for field in possible_fields:
            if field in dataset.column_names:
                return field
        # Default to first string field
        for field in dataset.column_names:
            if isinstance(dataset[0][field], (str, list)):
                return field
        raise ValueError(
            f"Could not detect text field in dataset. Columns: {dataset.column_names}"
        )

    def _create_dummy_calibration_data(
        self, tokenizer: AutoTokenizer
    ) -> List[torch.Tensor]:
        """Create dummy calibration data as fallback."""
        logger.warning("Using dummy calibration data - Hessians may be less accurate")
        calibration_data = []
        for _ in range(self.num_calibration_samples):
            # Random tokens
            tokens = torch.randint(
                0, tokenizer.vocab_size, (1, self.calibration_seq_length)
            )
            calibration_data.append(tokens)
        return calibration_data

    # ============================================================================
    # OPTIMIZED STEP: Compute and Cache Importance Masks Only
    # ============================================================================

    def compute_and_cache_importance_masks(self):
        """
        STORAGE-OPTIMIZED: Compute importance masks and cache only masks.

        This saves ~98% storage compared to caching full Hessians:
        - Old: Save 6GB Hessian per model = 12GB total
        - New: Save 100MB mask per model = 200MB total

        Process:
        1. Load model
        2. Compute Hessians (in memory only)
        3. Generate importance mask from Hessians
        4. Save ONLY mask (sparse boolean array)
        5. Delete everything, move to next model
        """
        log_print("=" * 60)
        log_print("STEP 2: Computing Importance Masks (SparseGPT)")
        log_print("=" * 60)
        log_print("Note: Using cached task vectors, computing importance on-the-fly")

        # CRITICAL WARNING: Check calibration sample count
        if self.num_calibration_samples < 64:
            log_print("\n" + "=" * 60)
            log_print("⚠⚠⚠ WARNING: LOW CALIBRATION SAMPLE COUNT ⚠⚠⚠")
            log_print("=" * 60)
            log_print(f"Using only {self.num_calibration_samples} calibration samples")
            log_print("Hessian estimates will be VERY NOISY with < 64 samples!")
            log_print("This can lead to poor importance estimates and model collapse.")
            log_print("")
            log_print("Recommendations:")
            log_print("  - For testing: Use magnitude-based TIES/DARE (no SparseGPT)")
            log_print("  - For production: Use ≥64 samples (128+ recommended)")
            log_print("=" * 60 + "\n")

        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process each fine-tuned model
        for idx, (ft_model_path, dataset_name) in enumerate(
            zip(self.finetuned_model_paths, self.dataset_names)
        ):
            log_print(
                f"\n{'='*60}\n[{idx+1}/{len(self.finetuned_model_paths)}] Processing: {ft_model_path}\n{'='*60}"
            )

            # Check if already cached
            mask_file = self.mask_dir / f"importance_mask_{idx}.pt"
            if mask_file.exists():
                log_print(f"  ✓ Importance mask already cached: {mask_file}")
                log_print(f"  Skipping computation...")
                continue

            # Load calibration data
            log_print(f"  [1/4] Loading calibration data from {dataset_name}...")
            calibration_data = self.load_calibration_data(dataset_name, tokenizer)

            # Load model
            log_print(f"  [2/4] Loading model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self._load_finetuned_model(ft_model_path, self.base_model_path)

            # Try GPU first, fallback to CPU if OOM
            try:
                model = model.to(device)
                model.eval()
                log_print(f"    ✓ Model loaded on {device}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    log_print(f"    ⚠ GPU memory exceeded, falling back to CPU")
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model = model.to(device)
                    model.eval()
                    log_print(f"    ✓ Model loaded on {device}")
                else:
                    raise

            # Store device for forward passes
            self._current_device = device

            # Find all linear layers
            linear_layers = self._find_linear_layers(model)
            log_print(f"  Found {len(linear_layers)} linear layers")

            # Compute Hessians (in memory only, not saved!)
            log_print(f"  [3/4] Computing Hessians (temporary, in-memory)...")
            hessian_inv_diags = self._compute_hessians_for_model(
                model, linear_layers, calibration_data
            )

            # Generate importance masks from Hessians
            log_print(
                f"  [4/4] Generating importance masks (top {int(self.density*100)}% weights)..."
            )

            # Load pre-computed task vectors from cache
            log_print("    Loading task vectors from cache...")
            task_vector_file = self.task_vector_dir / f"task_vector_{idx}.pt"
            if not task_vector_file.exists():
                raise FileNotFoundError(
                    f"Task vector cache not found: {task_vector_file}\n"
                    f"Please run compute_and_save_task_vectors() first!"
                )
            task_vector_dict = torch.load(task_vector_file, map_location="cpu")
            log_print(f"    ✓ Loaded {len(task_vector_dict)} task vectors")

            importance_masks = {}
            for layer_name, h_inv_diag in hessian_inv_diags.items():
                # Find matching key in task vector cache
                layer_key = None
                for key in task_vector_dict.keys():
                    if layer_name in key or key in layer_name:
                        layer_key = key
                        break

                if layer_key is None:
                    # Try adding .weight suffix
                    if f"{layer_name}.weight" in task_vector_dict:
                        layer_key = f"{layer_name}.weight"
                    else:
                        log_print(
                            f"    ⚠ Layer {layer_name} not found in task vectors, skipping"
                        )
                        continue

                # Get task vector from cache
                task_vector = task_vector_dict[layer_key]

                # Calculate importance on TASK VECTOR (the actual change)
                # importance = task_vector^2 / (H^{-1})^2
                # This measures: "How important is THIS CHANGE given the Hessian?"
                eps = 1e-10  # Numerical stability
                h_inv_diag_broadcasted = h_inv_diag.unsqueeze(0)  # [1, in_features]
                importance = task_vector.pow(2) / (
                    (h_inv_diag_broadcasted + eps).pow(2)
                )

                # Get top-k indices (MUCH more efficient than storing full mask!)
                k = int(importance.numel() * self.density)
                flat_importance = importance.flatten()
                topk_indices = torch.topk(flat_importance, k).indices

                # Store ONLY the indices of important weights (not the full mask)
                # This reduces storage from ~14GB to ~100MB per model!
                importance_masks[layer_name] = {
                    "indices": topk_indices.cpu(),
                    "shape": importance.shape,
                    "density": self.density,
                }

            log_print(f"  ✓ Generated masks for {len(importance_masks)} layers")

            # Save masks ONLY (not Hessians!)
            log_print(f"  Saving importance masks: {mask_file}")
            torch.save(importance_masks, mask_file)

            # Calculate saved space and compare to alternatives
            mask_size_mb = (
                mask_file.stat().st_size / (1024**2) if mask_file.exists() else 0
            )

            # Estimate full dense mask size: 112 layers * avg weight shape * 4 bytes/float32
            # For LLaMA 1B: ~45M parameters in linear layers → ~180MB per model as dense mask
            estimated_dense_mb = 180  # Conservative estimate
            savings_pct = (
                (1 - mask_size_mb / estimated_dense_mb) * 100
                if estimated_dense_mb > 0
                else 0
            )

            log_print(
                f"  ✓ Saved {mask_size_mb:.1f}MB (vs ~{estimated_dense_mb}MB dense, {savings_pct:.1f}% savings!)"
            )

            # Free memory
            log_print(f"  Cleaning up memory...")
            del (
                model,
                calibration_data,
                hessian_inv_diags,
                importance_masks,
                task_vector_dict,
            )
            gc.collect()
            torch.cuda.empty_cache()
            log_print(f"  ✓ Model {idx+1}/{len(self.finetuned_model_paths)} complete!")

        log_print("\n✓ Importance mask computation complete!")
        log_print(
            f"Storage efficiency: Indices-only format uses ~{len(self.finetuned_model_paths) * 20}MB\n"
            f"  vs ~{len(self.finetuned_model_paths) * 180}MB for dense masks\n"
            f"  vs ~{len(self.finetuned_model_paths) * 6000}MB for full Hessians!"
        )

    # ============================================================================
    # HELPER: Compute Task Vectors On-The-Fly (Fast, Memory-Efficient)
    # ============================================================================

    def _compute_task_vectors_for_layer(
        self, layer_name: str
    ) -> tuple[List[torch.Tensor], torch.device]:
        """
        Compute task vectors for a specific layer on-the-fly using cached models.

        IMPORTANT: This loads models into memory ONCE and reuses them for all layers.
        This is MUCH faster than loading task vectors from disk 112 times!

        Performance comparison:
        - Load from disk per layer: 112 layers × 2GB file × 2 models = 224 slow I/O ops
        - Load models once, compute on-fly: 2 model loads + 112 fast in-memory computations

        The task vector CACHING is still useful for the importance mask computation,
        but during merging it's faster to keep models in RAM and compute deltas.

        Args:
            layer_name: Name of the layer to compute task vectors for

        Returns:
            Tuple of (task_vectors, device)
        """
        # Load models into cache on first call (ONCE for all layers)
        if not hasattr(self, "_base_model_cache"):
            log_print("\n[OPTIMIZATION] Loading models into memory (one-time cost)...")
            log_print("  This is faster than loading task vectors from disk 112 times!")
            log_print("  Loading base model...")
            self._base_model_cache = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16, device_map="cpu"
            )

            log_print(
                f"  Loading {len(self.finetuned_model_paths)} fine-tuned models..."
            )
            self._ft_models_cache = []
            for i, ft_path in enumerate(self.finetuned_model_paths):
                ft_model = self._load_finetuned_model(ft_path, self.base_model_path)
                self._ft_models_cache.append(ft_model)
                log_print(
                    f"    [{i+1}/{len(self.finetuned_model_paths)}] Loaded {ft_path.split('/')[-1]}"
                )

            log_print("  ✓ All models cached in memory\n")

            # Determine device
            self._model_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Get base layer parameters
        base_params = self._get_layer_params(self._base_model_cache, layer_name)

        # Compute task vector for each fine-tuned model
        task_vectors = []
        for ft_model in self._ft_models_cache:
            ft_params = self._get_layer_params(ft_model, layer_name)

            # Compute task vector: τ = θ_ft - θ_base
            # Ensure both are on same device
            task_vector = ft_params.to(base_params.device) - base_params
            task_vectors.append(task_vector)

        return task_vectors, self._model_device

    def _load_lora_merged_model(
        self, lora_path: str, layer_name: str = None
    ) -> AutoModelForCausalLM:
        """
        Load LoRA adapter and merge with base model.
        Optimized to only load needed layers if layer_name specified.

        Args:
            lora_path: Path to LoRA adapter
            layer_name: Optional layer name for selective loading

        Returns:
            Model with LoRA merged
        """
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.float16, device_map="cpu"
        )

        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()

        return merged_model

    # ============================================================================
    # STEP 1: Compute and Cache Task Vectors (Shared by all methods)
    # ============================================================================

    def compute_and_save_task_vectors(self):
        """
        Compute task vectors (ft_weights - base_weights) for each model.

        This is computed ONCE and used by ALL three merging methods:
        - TIES-Magnitude
        - DARE-Random
        - TIES-SparseGPT

        Process ONE MODEL AT A TIME to save memory:
        1. Load base model
        2. For each fine-tuned model:
           a. Load fine-tuned model
           b. Compute task vector: τ = θ_ft - θ_base
           c. Save task vector to disk
           d. Unload fine-tuned model
        3. Unload base model
        """
        log_print("=" * 60)
        log_print("STEP 1: Computing Task Vectors (Shared by All Methods)")
        log_print("=" * 60)

        # Load base model ONCE
        log_print(f"Loading base model: {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Keep on CPU to save GPU memory
        )

        # Get state dict
        base_state_dict = base_model.state_dict()

        # Process each fine-tuned model
        for idx, ft_model_path in enumerate(self.finetuned_model_paths):
            log_print(
                f"\n{'='*60}\n[{idx+1}/{len(self.finetuned_model_paths)}] Processing: {ft_model_path}\n{'='*60}"
            )

            # Check if already cached
            task_vector_file = self.task_vector_dir / f"task_vector_{idx}.pt"
            if task_vector_file.exists():
                log_print(f"  ✓ Task vector already cached: {task_vector_file}")
                log_print(f"  Skipping computation...")
                continue

            # Load fine-tuned model
            log_print(f"  [1/4] Loading fine-tuned model...")
            start_time = time.time()
            ft_model = self._load_finetuned_model(ft_model_path, self.base_model_path)
            log_print(f"  ✓ Model loaded in {time.time()-start_time:.1f}s")

            # Get state dict
            log_print(f"  [2/4] Extracting state dict...")
            ft_state_dict = ft_model.state_dict()
            log_print(f"  ✓ Found {len(ft_state_dict)} parameters")

            # Compute task vectors: τ = θ_ft - θ_base
            log_print(f"  [3/4] Computing task vectors...")
            task_vectors = {}
            num_keys = len(base_state_dict.keys())
            for i, key in enumerate(base_state_dict.keys()):
                if (i + 1) % 50 == 0:
                    log_print(f"    Progress: {i+1}/{num_keys} layers processed")
                if key in ft_state_dict:
                    # Task vector for this layer
                    task_vectors[key] = (
                        ft_state_dict[key].cpu().float()
                        - base_state_dict[key].cpu().float()
                    )
            log_print(f"  ✓ Computed task vectors for {len(task_vectors)} layers")

            # Save to disk
            log_print(f"  [4/4] Saving task vectors: {task_vector_file}")
            torch.save(task_vectors, task_vector_file)
            log_print(f"  ✓ Saved to disk")

            # Free memory
            log_print(f"  Cleaning up memory...")
            del ft_model, ft_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            log_print(f"  ✓ Model {idx+1}/{len(self.finetuned_model_paths)} complete!")

        # Free base model
        del base_model, base_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        log_print("\n✓ Task vector computation complete!")

    def _load_finetuned_model(
        self, ft_model_path: str, base_model_path: str
    ) -> nn.Module:
        """
        Load fine-tuned model (handles both full models and LoRA adapters).

        Args:
            ft_model_path: Path to fine-tuned model or LoRA adapter
            base_model_path: Path to base model (for LoRA)

        Returns:
            Loaded model
        """
        import os

        # Check if path exists and determine model type
        is_lora = False
        actual_model_path = ft_model_path

        if os.path.exists(ft_model_path):
            files_in_path = os.listdir(ft_model_path)
            logger.info(f"  Found {len(files_in_path)} items in {ft_model_path}")
            logger.info(f"  Items: {files_in_path[:10]}")

            # Check if it's a LoRA adapter
            if "adapter_config.json" in files_in_path:
                is_lora = True
                logger.info("  ✓ Detected LoRA adapter format")
            elif "config.json" in files_in_path:
                logger.info("  ✓ Detected full model format")
            else:
                # Check if there's a single subdirectory (common in Kaggle datasets)
                subdirs = [
                    d
                    for d in files_in_path
                    if os.path.isdir(os.path.join(ft_model_path, d))
                ]
                if (
                    len(subdirs) == 1
                    and len(
                        [
                            f
                            for f in files_in_path
                            if os.path.isfile(os.path.join(ft_model_path, f))
                        ]
                    )
                    == 0
                ):
                    # Only one subdirectory and no files at this level - likely nested
                    nested_path = os.path.join(ft_model_path, subdirs[0])
                    logger.warning(
                        f"  No config files at top level, checking subdirectory: {subdirs[0]}"
                    )

                    nested_files = os.listdir(nested_path)
                    logger.info(f"  Found in subdirectory: {nested_files[:10]}")

                    if "adapter_config.json" in nested_files:
                        is_lora = True
                        actual_model_path = nested_path
                        logger.info(
                            f"  ✓ Found LoRA adapter in subdirectory: {subdirs[0]}"
                        )
                    elif "config.json" in nested_files:
                        actual_model_path = nested_path
                        logger.info(
                            f"  ✓ Found full model in subdirectory: {subdirs[0]}"
                        )
                    else:
                        logger.warning(
                            f"  No config files found in subdirectory either"
                        )
                else:
                    logger.warning(f"  No config files found in {ft_model_path}")
                    logger.warning(f"  Subdirectories: {subdirs}")
                    logger.warning(
                        f"  Files: {[f for f in files_in_path if os.path.isfile(os.path.join(ft_model_path, f))]}"
                    )

        # Try loading based on detected type
        if is_lora:
            # Load as LoRA adapter directly
            try:
                logger.info("  Loading as LoRA adapter...")
                logger.info(f"  Base model: {base_model_path}")
                logger.info(f"  LoRA adapter: {actual_model_path}")

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path, torch_dtype=torch.float16, device_map="cpu"
                )
                logger.info("  ✓ Base model loaded")

                model = PeftModel.from_pretrained(base_model, actual_model_path)
                logger.info("  ✓ LoRA adapter loaded")

                # Merge LoRA weights into base model
                model = model.merge_and_unload()
                logger.info("  ✓ LoRA weights merged into base model")
                return model
            except Exception as e:
                logger.error(f"\n{'='*80}")
                logger.error("ERROR: Failed to load LoRA adapter")
                logger.error(f"\nBase model path: {base_model_path}")
                logger.error(f"LoRA adapter path: {actual_model_path}")
                logger.error(f"Original path: {ft_model_path}")
                logger.error(f"\nError: {str(e)}")
                logger.error("\nTroubleshooting:")
                logger.error("1. Check base_model_path is correct")
                logger.error(
                    "2. Ensure adapter files exist (adapter_config.json, adapter_model.*)"
                )
                logger.error("3. Verify base model and adapter are compatible")
                logger.error(f"4. Check files at: {actual_model_path}")
                if os.path.exists(actual_model_path):
                    logger.error(f"   Files found: {os.listdir(actual_model_path)}")
                logger.error(f"{'='*80}\n")
                raise
        else:
            # Try loading as full model
            try:
                logger.info("  Attempting to load as full fine-tuned model...")
                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path, torch_dtype=torch.float16, device_map="cpu"
                )
                logger.info("  ✓ Loaded as full model")
                return model
            except Exception as e:
                logger.error(f"\n{'='*80}")
                logger.error("ERROR: Failed to load as full model")
                logger.error(f"\nPath: {actual_model_path}")
                logger.error(f"Original path: {ft_model_path}")
                logger.error(f"\nError: {str(e)[:300]}")
                logger.error("\nPossible causes:")
                logger.error(
                    "1. Missing model files (config.json, pytorch_model.bin/model.safetensors)"
                )
                logger.error("2. Incorrect path")
                logger.error(
                    "3. Model is actually a LoRA adapter (check for adapter_config.json)"
                )
                logger.error("4. Nested directory structure (check subdirectories)")
                if os.path.exists(actual_model_path):
                    logger.error(f"\nContents of {actual_model_path}:")
                    logger.error(f"  {os.listdir(actual_model_path)}")
                logger.error(f"{'='*80}\n")
                raise ValueError(
                    f"Failed to load model from {actual_model_path}. "
                    "Check logs above for details."
                )

    # ============================================================================
    # STEP 3: Compute Hessians for SparseGPT
    # ============================================================================

    def compute_and_save_hessians(self):
        """
        Compute Hessian inverse diagonals for each task using calibration data.

        Process ONE MODEL AT A TIME:
        1. For each fine-tuned model:
           a. Load model
           b. Load calibration data
           c. Compute Hessians for all linear layers
           d. Save Hessians to disk
           e. Unload model
        """
        log_print("=" * 60)
        log_print("STEP 2: Computing Hessians for SparseGPT")
        log_print("=" * 60)

        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process each fine-tuned model
        for idx, (ft_model_path, dataset_name) in enumerate(
            zip(self.finetuned_model_paths, self.dataset_names)
        ):
            logger.info(
                f"\n[{idx+1}/{len(self.finetuned_model_paths)}] Processing {ft_model_path}"
            )

            # Check if already cached
            hessian_file = self.hessian_dir / f"hessian_{idx}.pt"
            if hessian_file.exists():
                logger.info(f"  Hessians already computed: {hessian_file}")
                continue

            # Load calibration data
            log_print(f"  Loading calibration data from {dataset_name}...")
            calibration_data = self.load_calibration_data(dataset_name, tokenizer)

            # Load model
            log_print(f"  Loading model...")
            model = self._load_finetuned_model(ft_model_path, self.base_model_path)
            model.eval()
            model.to(self.device)

            # Find all linear layers
            linear_layers = self._find_linear_layers(model)
            logger.info(f"  Found {len(linear_layers)} linear layers")

            # Compute Hessians
            log_print(f"  Computing Hessians...")
            hessian_inv_diags = self._compute_hessians_for_model(
                model, linear_layers, calibration_data
            )

            # Save to disk
            logger.info(f"  Saving Hessians: {hessian_file}")
            torch.save(hessian_inv_diags, hessian_file)

            # Free memory
            del model, calibration_data
            gc.collect()
            torch.cuda.empty_cache()

        log_print("\n✓ Hessian computation complete!")

    def _find_linear_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Find all Linear layers in the model."""
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip lm_head if present (usually not merged)
                if "lm_head" not in name:
                    linear_layers[name] = module
        return linear_layers

    def _compute_hessians_for_model(
        self,
        model: nn.Module,
        linear_layers: Dict[str, nn.Linear],
        calibration_data: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Hessian inverse diagonals for all linear layers.

        Args:
            model: The model to compute Hessians for
            linear_layers: Dictionary of layer_name -> Linear module
            calibration_data: List of input tensors

        Returns:
            Dictionary mapping layer_name -> H^{-1} diagonal tensor
        """
        hessian_inv_diags = {}

        # MEMORY OPTIMIZATION: Process calibration data in batches
        # Instead of accumulating all activations, process in chunks of BATCH_SIZE
        BATCH_SIZE = 16  # Process 16 samples at a time to avoid OOM
        num_batches = (len(calibration_data) + BATCH_SIZE - 1) // BATCH_SIZE

        log_print(
            f"    Computing Hessians with {len(calibration_data)} samples in {num_batches} batches of {BATCH_SIZE}..."
        )
        log_print(f"    This prevents memory overflow by processing incrementally.")

        # Initialize Hessian calculators for all layers
        hessian_calcs = {}
        for name, layer in linear_layers.items():
            weight_shape = layer.weight.shape
            hessian_calcs[name] = HessianCalculator(
                weight_shape, device=torch.device("cpu")
            )

        forward_device = getattr(self, "_current_device", self.device)
        start_time = time.time()

        # Process calibration data in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(calibration_data))
            batch_samples = calibration_data[batch_start:batch_end]

            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
                samples_done = batch_end
                log_print(
                    f"      Batch {batch_idx+1}/{num_batches} ({samples_done}/{len(calibration_data)} samples) | ETA: {eta:.0f}s"
                )

            # Collect activations for this batch only
            activations = {name: [] for name in linear_layers.keys()}
            hooks = []

            def get_activation_hook(name):
                def hook(module, inp, out):
                    # Save input activations (detach and keep on CPU to save GPU memory)
                    activations[name].append(inp[0].detach().cpu())

                return hook

            # Register hooks
            for name, layer in linear_layers.items():
                hooks.append(layer.register_forward_hook(get_activation_hook(name)))

            # Forward pass for this batch
            with torch.no_grad():
                for sample in batch_samples:
                    try:
                        sample = sample.to(forward_device)
                        _ = model(sample)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            log_print(f"        ⚠ GPU OOM, switching to CPU")
                            torch.cuda.empty_cache()
                            forward_device = torch.device("cpu")
                            model = model.to(forward_device)
                            sample = sample.to(forward_device)
                            _ = model(sample)
                        else:
                            raise

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Update Hessian calculators with this batch's activations
            for name in linear_layers.keys():
                if activations[name]:
                    for act in activations[name]:
                        hessian_calcs[name].add_batch(act)

            # Clear activations to free memory
            del activations
            torch.cuda.empty_cache()

        log_print(
            f"    ✓ Processed all {len(calibration_data)} samples in {time.time()-start_time:.1f}s"
        )

        # Compute inverse Hessian diagonals from accumulated statistics
        log_print(
            f"    Computing inverse Hessian diagonals for {len(linear_layers)} layers..."
        )
        num_layers = len(linear_layers)
        for layer_idx, name in enumerate(linear_layers.keys()):
            if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
                log_print(f"      Layer {layer_idx+1}/{num_layers}: {name}")

            # Compute inverse diagonal
            h_inv_diag = hessian_calcs[name].get_inverse_hessian_diag(percdamp=0.01)
            hessian_inv_diags[name] = h_inv_diag

        log_print(f"    ✓ Computed Hessians for {len(hessian_inv_diags)} layers")
        return hessian_inv_diags

    # ============================================================================
    # STEP 4: Merge Models (Three Methods)
    # ============================================================================

    def merge_with_ties(self, use_sparsegpt: bool = False) -> AutoModelForCausalLM:
        """
        Merge models using TIES method.

        Args:
            use_sparsegpt: If True, use SparseGPT importance for trimming
                          If False, use magnitude-based trimming

        Returns:
            Merged model
        """
        method_name = "TIES-SparseGPT" if use_sparsegpt else "TIES-Magnitude"
        logger.info("=" * 60)
        logger.info(f"MERGING WITH {method_name}")
        logger.info("=" * 60)

        # Load base model
        logger.info("Loading base model...")
        merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.float16, device_map="cpu"
        )

        # Get layer names to merge
        linear_layers = self._find_linear_layers(merged_model)
        layer_names = list(linear_layers.keys())

        logger.info(f"Merging {len(layer_names)} layers...")

        # TIES merger
        merger = TIES()

        # PERFORMANCE OPTIMIZATION: Load mask files ONCE and cache them
        cached_mask_dicts = None
        if use_sparsegpt:
            logger.info("Pre-loading importance mask files...")
            cached_mask_dicts = []
            for idx in range(len(self.finetuned_model_paths)):
                mask_file = self.mask_dir / f"importance_mask_{idx}.pt"
                mask_dict = torch.load(mask_file, map_location="cpu", weights_only=False)
                cached_mask_dicts.append(mask_dict)
                logger.info(f"  Loaded mask file {idx}: {mask_file.name} ({len(mask_dict)} layers)")

        # Merge layer by layer
        log_print(f"\nMerging {len(layer_names)} layers...")
        merge_start = time.time()
        failed_layers = []
        for layer_idx, layer_name in enumerate(layer_names):
            if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
                elapsed = time.time() - merge_start
                rate = (layer_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(layer_names) - layer_idx - 1) / rate if rate > 0 else 0
                log_print(
                    f"  [{layer_idx+1}/{len(layer_names)}] {layer_name} | ETA: {eta:.0f}s"
                )

            # Get base layer parameters
            base_params = self._get_layer_params(merged_model, layer_name)

            # Compute task vectors on-the-fly (NO STORAGE!)
            task_vectors, model_device = self._compute_task_vectors_for_layer(
                layer_name
            )

            # Load importance masks if using SparseGPT
            # MEMORY OPTIMIZATION: Use cached mask dicts instead of reloading from disk!
            importance_masks = None
            if use_sparsegpt:
                importance_masks = []
                for idx, mask_dict in enumerate(cached_mask_dicts):
                    full_key = self._find_matching_key(mask_dict, layer_name)

                    if full_key:
                        # Extract this layer's mask data (no file I/O!)
                        mask_data = mask_dict[full_key]

                        # Check if it's the new format (dict with indices)
                        if isinstance(mask_data, dict) and "indices" in mask_data:
                            # New format: reconstruct mask from indices
                            indices = mask_data["indices"]
                            shape = mask_data["shape"]

                            # OPTIMIZED: Pre-allocate on correct device, scatter indices
                            dense_mask = torch.zeros(
                                shape, dtype=torch.float32, device=model_device
                            )
                            # Flatten, set indices, reshape in-place
                            dense_mask.view(-1)[indices] = 1.0
                        else:
                            # Old format: sparse tensor or dense mask
                            if hasattr(mask_data, "to_dense") and mask_data.is_sparse:
                                dense_mask = mask_data.to_dense()
                            else:
                                dense_mask = mask_data
                            # Move to device
                            dense_mask = dense_mask.to(dtype=torch.float32, device=model_device)

                        importance_masks.append(dense_mask)
                    else:
                        log_print(
                            f"    ⚠ Mask for {layer_name} not found in model {idx}, using all-ones mask"
                        )
                        # Create dummy mask (all ones) as float
                        importance_masks.append(
                            torch.ones(
                                base_params.shape,
                                dtype=torch.float32,
                                device=model_device,
                            ).flatten()
                        )

                # DO NOT apply masks here - merger will handle it!
                # Applying masks here AND in merger causes double-trimming:
                # Result: 0.2 * 0.2 = 0.04 = only 4% of weights survive!

            # Merge this layer
            # IMPORTANT: Pass importance_masks to merger, don't pre-apply!
            try:
                # Use same device as models for consistency
                merge_device = model_device

                try:
                    # Pass task vectors directly - merger will apply trimming
                    # If importance_masks is not None, use them; otherwise use magnitude
                    merged_params = merger.merge(
                        weights=[1.0] * len(task_vectors),
                        base_model_parameters=base_params,
                        ft_models_parameters=[base_params + tv for tv in task_vectors],
                        densities=[self.density] * len(task_vectors),
                        device=merge_device,
                        importance_masks=importance_masks,  # Pass masks to merger
                        use_sparsegpt=use_sparsegpt,  # Use correct flag
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        if merge_device.type == "cuda":
                            log_print(
                                f"      ⚠ GPU OOM on layer {layer_name}, moving to CPU"
                            )
                            torch.cuda.empty_cache()
                            merge_device = torch.device("cpu")
                            # Move all tensors to CPU
                            base_params_cpu = base_params.to("cpu")
                            task_vectors_cpu = [tv.to("cpu") for tv in task_vectors]
                            importance_masks_cpu = None
                            if importance_masks is not None:
                                importance_masks_cpu = [
                                    m.to("cpu") for m in importance_masks
                                ]
                            merged_params = merger.merge(
                                weights=[1.0] * len(task_vectors_cpu),
                                base_model_parameters=base_params_cpu,
                                ft_models_parameters=[
                                    base_params_cpu + tv for tv in task_vectors_cpu
                                ],
                                densities=[self.density] * len(task_vectors_cpu),
                                device=merge_device,
                                importance_masks=importance_masks_cpu,
                                use_sparsegpt=use_sparsegpt,
                            )
                        else:
                            raise
                    else:
                        raise  # Update merged model
                self._set_layer_params(merged_model, layer_name, merged_params)
            except Exception as e:
                log_print(f"    ❌ Error merging {layer_name}: {e}")
                logger.error(f"    Error merging {layer_name}: {e}")
                failed_layers.append((layer_name, str(e)))
                continue

        # Report failures
        if failed_layers:
            log_print(f"\n⚠ Warning: {len(failed_layers)} layers failed to merge:")
            for layer, error in failed_layers[:5]:  # Show first 5
                log_print(f"  - {layer}: {error}")
            if len(failed_layers) > 5:
                log_print(f"  ... and {len(failed_layers) - 5} more")

        # Cleanup cached mask dictionaries to free memory
        if cached_mask_dicts is not None:
            del cached_mask_dicts
            gc.collect()

        # Cleanup base model cache
        self._clear_base_model_cache()

        log_print(f"\n✓ {method_name} merge complete!")
        return merged_model

    def _clear_base_model_cache(self):
        """
        Clear cached base and ft models to free memory.
        Called after each merging method completes.
        """
        if hasattr(self, "_base_model_cache"):
            del self._base_model_cache
        if hasattr(self, "_ft_models_cache"):
            del self._ft_models_cache
        if hasattr(self, "_model_device"):
            del self._model_device
        gc.collect()
        torch.cuda.empty_cache()

    def merge_with_dare(self, use_sparsegpt: bool = False) -> AutoModelForCausalLM:
        """
        Merge models using DARE method.

        Args:
            use_sparsegpt: If True, use SparseGPT importance for dropout
                          If False, use random dropout

        Returns:
            Merged model
        """
        method_name = "DARE-SparseGPT" if use_sparsegpt else "DARE-Random"
        logger.info("=" * 60)
        logger.info(f"MERGING WITH {method_name}")
        logger.info("=" * 60)

        # Load base model
        logger.info("Loading base model...")
        merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.float16, device_map="cpu"
        )

        # Get layer names to merge
        linear_layers = self._find_linear_layers(merged_model)
        layer_names = list(linear_layers.keys())

        logger.info(f"Merging {len(layer_names)} layers...")

        # DARE merger
        merger = DARE()

        # PERFORMANCE OPTIMIZATION: Load mask files ONCE and cache them
        cached_mask_dicts = None
        if use_sparsegpt:
            logger.info("Pre-loading importance mask files...")
            cached_mask_dicts = []
            for idx in range(len(self.finetuned_model_paths)):
                mask_file = self.mask_dir / f"importance_mask_{idx}.pt"
                mask_dict = torch.load(mask_file, map_location="cpu", weights_only=False)
                cached_mask_dicts.append(mask_dict)
                logger.info(f"  Loaded mask file {idx}: {mask_file.name} ({len(mask_dict)} layers)")

        # Merge layer by layer
        log_print(f"\nMerging {len(layer_names)} layers...")
        merge_start = time.time()
        failed_layers = []
        for layer_idx, layer_name in enumerate(layer_names):
            if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
                elapsed = time.time() - merge_start
                rate = (layer_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(layer_names) - layer_idx - 1) / rate if rate > 0 else 0
                log_print(
                    f"  [{layer_idx+1}/{len(layer_names)}] {layer_name} | ETA: {eta:.0f}s"
                )

            # Get base layer parameters
            base_params = self._get_layer_params(merged_model, layer_name)

            # Compute task vectors on-the-fly (NO STORAGE!)
            task_vectors, model_device = self._compute_task_vectors_for_layer(
                layer_name
            )

            # Load importance masks if using SparseGPT
            # MEMORY OPTIMIZATION: Use cached mask dicts instead of reloading from disk!
            importance_masks = None
            if use_sparsegpt:
                importance_masks = []
                for idx, mask_dict in enumerate(cached_mask_dicts):
                    full_key = self._find_matching_key(mask_dict, layer_name)

                    if full_key:
                        # Extract this layer's mask data (no file I/O!)
                        mask_data = mask_dict[full_key]

                        # Check if it's the new format (dict with indices)
                        if isinstance(mask_data, dict) and "indices" in mask_data:
                            # New format: reconstruct mask from indices
                            indices = mask_data["indices"]
                            shape = mask_data["shape"]

                            # OPTIMIZED: Pre-allocate on correct device, scatter indices
                            dense_mask = torch.zeros(
                                shape, dtype=torch.float32, device=model_device
                            )
                            # Flatten, set indices, reshape in-place
                            dense_mask.view(-1)[indices] = 1.0
                        else:
                            # Old format: sparse tensor or dense mask
                            if hasattr(mask_data, "to_dense") and mask_data.is_sparse:
                                dense_mask = mask_data.to_dense()
                            else:
                                dense_mask = mask_data
                            # Move to device
                            dense_mask = dense_mask.to(dtype=torch.float32, device=model_device)

                        importance_masks.append(dense_mask)
                    else:
                        log_print(
                            f"    ⚠ Mask for {layer_name} not found in model {idx}, using all-ones mask"
                        )
                        # Create dummy mask (all ones) as float
                        importance_masks.append(
                            torch.ones(
                                base_params.shape,
                                dtype=torch.float32,
                                device=model_device,
                            ).flatten()
                        )

                # DO NOT apply masks here!

            # Merge this layer
            try:
                # Use same device as models for consistency
                merge_device = model_device

                try:
                    merged_params = merger.merge(
                        weights=[1.0] * len(task_vectors),
                        base_model_parameters=base_params,
                        ft_models_parameters=[base_params + tv for tv in task_vectors],
                        densities=[self.density] * len(task_vectors),
                        device=merge_device,
                        importance_masks=importance_masks,
                        use_sparsegpt=use_sparsegpt,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        if merge_device.type == "cuda":
                            log_print(
                                f"      ⚠ GPU OOM on layer {layer_name}, moving to CPU"
                            )
                            torch.cuda.empty_cache()
                            merge_device = torch.device("cpu")
                            # Move all tensors to CPU
                            base_params_cpu = base_params.to("cpu")
                            task_vectors_cpu = [tv.to("cpu") for tv in task_vectors]
                            merged_params = merger.merge(
                                weights=[1.0] * len(task_vectors_cpu),
                                base_model_parameters=base_params_cpu,
                                ft_models_parameters=[
                                    base_params_cpu + tv for tv in task_vectors_cpu
                                ],
                                densities=[self.density] * len(task_vectors_cpu),
                                device=merge_device,
                                hessian_inv_diags=None,
                                use_sparsegpt=False,
                            )
                        else:
                            raise
                    else:
                        raise

                # Update merged model
                self._set_layer_params(merged_model, layer_name, merged_params)
            except Exception as e:
                log_print(f"    ❌ Error merging {layer_name}: {e}")
                logger.error(f"    Error merging {layer_name}: {e}")
                failed_layers.append((layer_name, str(e)))
                continue

        # Report failures
        if failed_layers:
            log_print(f"\n⚠ Warning: {len(failed_layers)} layers failed to merge:")
            for layer, error in failed_layers[:5]:  # Show first 5
                log_print(f"  - {layer}: {error}")
            if len(failed_layers) > 5:
                log_print(f"  ... and {len(failed_layers) - 5} more")

        # Cleanup cached mask dictionaries to free memory
        if cached_mask_dicts is not None:
            del cached_mask_dicts
            gc.collect()

        # Cleanup base model cache
        self._clear_base_model_cache()

        log_print(f"\n✓ {method_name} merge complete!")
        return merged_model

    def _find_matching_key(self, state_dict: Dict, layer_name: str) -> Optional[str]:
        """
        Find full key in state dict that matches layer name.

        Handles variations like:
        - 'model.layers.0.mlp.gate_proj.weight' in state dict
        - 'model.layers.0.mlp.gate_proj' as layer name
        """
        # Direct match
        if layer_name in state_dict:
            return layer_name

        # Try with .weight suffix
        weight_key = f"{layer_name}.weight"
        if weight_key in state_dict:
            return weight_key

        # Try with .bias suffix
        bias_key = f"{layer_name}.bias"
        if bias_key in state_dict:
            return bias_key

        # Fuzzy match
        for key in state_dict.keys():
            if layer_name in key and "weight" in key:
                return key

        return None

    def _get_layer_params(self, model: nn.Module, layer_name: str) -> torch.Tensor:
        """Get parameters of a specific layer."""
        parts = layer_name.split(".")
        module = model
        try:
            for i, part in enumerate(parts):
                if not hasattr(module, part):
                    raise AttributeError(
                        f"Module does not have attribute '{part}' (at position {i} in path {layer_name})"
                    )
                module = getattr(module, part)

            if not hasattr(module, "weight"):
                raise AttributeError(
                    f"Layer '{layer_name}' does not have a weight attribute"
                )

            return module.weight.data.clone()
        except Exception as e:
            logger.error(f"Error getting layer params for {layer_name}: {e}")
            raise

    def _set_layer_params(
        self, model: nn.Module, layer_name: str, params: torch.Tensor
    ):
        """Set parameters of a specific layer."""
        parts = layer_name.split(".")
        module = model
        try:
            for i, part in enumerate(parts):
                if not hasattr(module, part):
                    raise AttributeError(
                        f"Module does not have attribute '{part}' (at position {i} in path {layer_name})"
                    )
                module = getattr(module, part)

            if not hasattr(module, "weight"):
                raise AttributeError(
                    f"Layer '{layer_name}' does not have a weight attribute"
                )

            # Ensure params are on same device as module and correct dtype
            params = params.to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.data = params
        except Exception as e:
            logger.error(f"Error setting layer params for {layer_name}: {e}")
            raise

    # ============================================================================
    # STEP 5: Evaluation and Comparison
    # ============================================================================

    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        eval_dataset: str = "wikitext",
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate model perplexity on test set.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            eval_dataset: Dataset name for evaluation
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating model on {eval_dataset}...")

        model.eval()
        model.to(self.device)

        try:
            # Load evaluation data
            if eval_dataset == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                text_field = "text"
            else:
                dataset = load_dataset(eval_dataset, split="test")
                text_field = self._detect_text_field(dataset)

            # Compute perplexity
            total_loss = 0.0
            total_tokens = 0

            with torch.no_grad():
                for i in range(min(num_samples, len(dataset))):
                    try:
                        text = dataset[i][text_field]
                        if not text or len(text) < 10:
                            continue

                        # Tokenize
                        inputs = tokenizer(
                            text, max_length=512, truncation=True, return_tensors="pt"
                        ).to(self.device)

                        # Forward pass
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss

                        total_loss += loss.item() * inputs["input_ids"].size(1)
                        total_tokens += inputs["input_ids"].size(1)

                    except Exception as e:
                        logger.warning(f"Error on sample {i}: {e}")
                        continue

            # Compute perplexity
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics = {
                "perplexity": perplexity,
                "avg_loss": avg_loss,
                "num_samples": num_samples,
                "num_tokens": total_tokens,
            }

            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Avg Loss: {avg_loss:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"perplexity": float("inf"), "error": str(e)}
        finally:
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    def merge_all_methods(self) -> Dict[str, Dict]:
        """
        Run all three merging methods and compare results.

        Hybrid optimization strategy:
        1. Compute task vectors once, cache to disk (~2-6GB total)
           - Used ONLY for importance mask generation (1x disk read)
        2. During merging: Load models into RAM, compute task vectors on-the-fly
           - Much faster than 112 disk reads per model!
        3. Importance masks cached to disk (~20MB each, indices-only)

        Why this hybrid approach?
        - Importance masks: Need full task vectors once → cache to disk
        - Merging (TIES/DARE): Need per-layer task vectors 112 times → keep models in RAM

        Returns:
            Dictionary with results for each method
        """
        log_print("\n" + "=" * 60)
        log_print("COMPREHENSIVE MODEL MERGING COMPARISON")
        log_print("=" * 60)
        log_print("MEMORY-EFFICIENT MODE:")
        log_print("  - Task vectors: Temporarily cached, deleted after mask generation")
        log_print("  - Importance masks: ~20MB each (indices-only, kept)")
        log_print("  - Models loaded on-demand during merging")
        log_print(
            f"  - Peak disk usage: ~{len(self.finetuned_model_paths) * 2000}MB (temporary)"
        )
        log_print(
            f"  - Final disk cache: ~{len(self.finetuned_model_paths) * 20}MB (masks only)"
        )
        log_print("=" * 60)

        # Step 1: Compute task vectors (temporary, for mask generation)
        self.compute_and_save_task_vectors()

        # Step 2: Compute importance masks using cached task vectors
        self.compute_and_cache_importance_masks()

        # Step 3: DELETE task vectors to free disk space (we only need masks now!)
        log_print(
            "\n[CLEANUP] Removing temporary task vector cache to save disk space..."
        )
        for idx in range(len(self.finetuned_model_paths)):
            task_vector_file = self.task_vector_dir / f"task_vector_{idx}.pt"
            if task_vector_file.exists():
                file_size_mb = task_vector_file.stat().st_size / (1024**2)
                task_vector_file.unlink()
                log_print(f"  ✓ Deleted task_vector_{idx}.pt ({file_size_mb:.1f}MB)")
        log_print(f"  ✓ Freed ~{len(self.finetuned_model_paths) * 2000}MB disk space!")
        log_print(
            f"  Final cache: ~{len(self.finetuned_model_paths) * 20}MB (importance masks only)\n"
        )

        # Load tokenizer for evaluation
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = {}

        # Method 1: Simple TIES (magnitude) - NO HESSIANS NEEDED
        log_print("\n" + "=" * 60)
        log_print("METHOD 1: TIES with Magnitude-Based Trimming")
        log_print("  (Computing task vectors on-the-fly)")
        log_print("=" * 60)
        start_time = time.time()
        ties_model = self.merge_with_ties(use_sparsegpt=False)
        ties_time = time.time() - start_time

        # Save
        ties_path = self.output_dir / "ties_magnitude"
        ties_model.save_pretrained(ties_path)
        tokenizer.save_pretrained(ties_path)
        log_print(f"Saved to: {ties_path}")

        # Evaluate
        ties_metrics = self.evaluate_model(ties_model, tokenizer)
        ties_metrics["merge_time"] = ties_time
        results["TIES-Magnitude"] = ties_metrics

        # Free memory
        del ties_model
        gc.collect()
        torch.cuda.empty_cache()

        # Method 2: Simple DARE (random)
        log_print("\n" + "=" * 60)
        log_print("METHOD 2: DARE with Random Dropout")
        log_print("=" * 60)
        start_time = time.time()
        dare_model = self.merge_with_dare(use_sparsegpt=False)
        dare_time = time.time() - start_time

        # Save
        dare_path = self.output_dir / "dare_random"
        dare_model.save_pretrained(dare_path)
        tokenizer.save_pretrained(dare_path)
        log_print(f"Saved to: {dare_path}")

        # Evaluate
        dare_metrics = self.evaluate_model(dare_model, tokenizer)
        dare_metrics["merge_time"] = dare_time
        results["DARE-Random"] = dare_metrics

        # Free memory
        del dare_model
        gc.collect()
        torch.cuda.empty_cache()

        # Method 3: SparseGPT-based (can use either TIES or DARE)
        log_print("\n" + "=" * 60)
        log_print("METHOD 3: TIES with SparseGPT Importance")
        log_print("=" * 60)
        start_time = time.time()
        sparsegpt_model = self.merge_with_ties(use_sparsegpt=True)
        sparsegpt_time = time.time() - start_time

        # Save
        sparsegpt_path = self.output_dir / "ties_sparsegpt"
        sparsegpt_model.save_pretrained(sparsegpt_path)
        tokenizer.save_pretrained(sparsegpt_path)
        log_print(f"Saved to: {sparsegpt_path}")

        # Evaluate
        sparsegpt_metrics = self.evaluate_model(sparsegpt_model, tokenizer)
        sparsegpt_metrics["merge_time"] = sparsegpt_time
        results["TIES-SparseGPT"] = sparsegpt_metrics

        # Free memory
        del sparsegpt_model
        gc.collect()
        torch.cuda.empty_cache()

        # Print comparison
        self._print_comparison(results)

        # Save results
        results_file = self.output_dir / "comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")

        return results

    def _print_comparison(self, results: Dict[str, Dict]):
        """Print comparison table."""
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 60)

        print(
            "\n{:<20} {:>12} {:>12} {:>12}".format(
                "Method", "Perplexity", "Avg Loss", "Time (s)"
            )
        )
        print("-" * 60)

        for method, metrics in results.items():
            print(
                "{:<20} {:>12.2f} {:>12.4f} {:>12.1f}".format(
                    method,
                    metrics.get("perplexity", float("inf")),
                    metrics.get("avg_loss", float("inf")),
                    metrics.get("merge_time", 0),
                )
            )

        print("\n" + "=" * 60)

        # Find best method
        best_method = min(
            results.items(), key=lambda x: x[1].get("perplexity", float("inf"))
        )
        logger.info(f"🏆 BEST METHOD: {best_method[0]}")
        logger.info(f"   Perplexity: {best_method[1]['perplexity']:.2f}")


# ============================================================================
# Command-Line Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Merge LLaMA models using TIES, DARE, and SparseGPT methods"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model",
    )

    parser.add_argument(
        "--finetuned_models",
        type=str,
        nargs="+",
        required=True,
        help="Paths or HuggingFace IDs of fine-tuned models",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names for calibration (one per fine-tuned model)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_models",
        help="Directory to save merged models",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./merge_cache",
        help="Directory for intermediate files",
    )

    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="Fraction of weights to keep (default: 0.2)",
    )

    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.finetuned_models) != len(args.datasets):
        raise ValueError(
            f"Number of models ({len(args.finetuned_models)}) must match "
            f"number of datasets ({len(args.datasets)})"
        )

    # Create merger
    merger = LLaMAMerger(
        base_model_path=args.base_model,
        finetuned_model_paths=args.finetuned_models,
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        density=args.density,
        num_calibration_samples=args.num_calibration_samples,
        device=args.device,
    )

    # Run all methods and compare
    results = merger.merge_all_methods()

    logger.info("\n✓ ALL DONE!")


if __name__ == "__main__":
    main()
