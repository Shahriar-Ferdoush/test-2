"""
LLaMA Model Merging with SparseGPT Sequential Processing

This file orchestrates the SparseGPT-based merging process, following the
pattern from sparsegpt/llama.py. It handles:
1. Calibration data capture (Catcher pattern)
2. Sequential layer processing
3. Hessian accumulation from current model state
4. Task vector pruning with error correction
5. Base model weight updates between layers (KEY INNOVATION!)

The key insight: After pruning task vectors for layer i, we add them back to
the base model BEFORE processing layer i+1. This ensures the Hessian for
layer i+1 is computed with accurate inputs that reflect the pruned model state.

Reference: Frantar & Alistarh (2023) - "SparseGPT: Massive Language Models
           Can Be Accurately Pruned in One-Shot"
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsegpt_task_vector import SparseGPTTaskVector


def find_layers(module: nn.Module, layer_types: List = None) -> Dict[str, nn.Module]:
    """
    Find all layers of specified types in a module.

    Matches modelutils.py::find_layers() from SparseGPT.
    """
    if layer_types is None:
        layer_types = [nn.Linear]

    layers = {}
    for name, mod in module.named_modules():
        if any(isinstance(mod, t) for t in layer_types):
            layers[name] = mod
    return layers


@torch.no_grad()
def llama_sparse_merge_sequential(
    base_model: AutoModelForCausalLM,
    finetuned_models: List[AutoModelForCausalLM],
    calibration_data: List[torch.Tensor],
    density: float = 0.2,
    device: str = "cuda",
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> AutoModelForCausalLM:
    """
    Merge fine-tuned models into base model using SparseGPT importance.

    Follows llama_sequential() pattern from sparsegpt/llama.py:
    1. Capture inputs to first transformer layer (Catcher pattern)
    2. For each layer sequentially:
       a. Create SparseGPTTaskVector pruner
       b. Accumulate Hessian from current base model state
       c. Compute task vectors (fine-tuned - base)
       d. Prune task vectors using SparseGPT algorithm
       e. Add pruned task vectors to base model (UPDATE!)
       f. Propagate outputs to next layer
    3. Return merged base model

    Args:
        base_model: Base model to merge into
        finetuned_models: List of fine-tuned models (all must have same architecture)
        calibration_data: List of tokenized input tensors [batch_size, seq_len]
        density: Fraction of task vector weights to keep (0.2 = keep 20%)
        device: Device for computation ('cuda' or 'cpu')
        blocksize: Block size for SparseGPT algorithm
        percdamp: Dampening factor for Hessian inversion

    Returns:
        Merged base model (modified in-place)
    """
    print("=" * 80)
    print("SPARSEGPT-BASED MODEL MERGING (Sequential Processing)")
    print("=" * 80)
    print(f"Base model: {type(base_model).__name__}")
    print(f"Fine-tuned models: {len(finetuned_models)}")
    print(f"Calibration samples: {len(calibration_data)}")
    print(f"Density: {density:.1%} (pruning {1-density:.1%} of task vector)")
    print(f"Device: {device}")
    print("=" * 80)

    dev = torch.device(device)

    # === Step 0: Validate model compatibility ===
    print("\n[0/3] Validating model architecture compatibility...")
    base_num_layers = len(base_model.model.layers)
    for idx, ft_model in enumerate(finetuned_models):
        ft_num_layers = len(ft_model.model.layers)
        if ft_num_layers != base_num_layers:
            raise ValueError(
                f"Fine-tuned model {idx} has {ft_num_layers} layers, "
                f"but base model has {base_num_layers} layers. "
                "Models must have same architecture."
            )
    print(f"  ✓ All models have {base_num_layers} transformer layers")

    # === Step 1: Disable cache and get model structure ===
    use_cache = base_model.config.use_cache
    base_model.config.use_cache = False

    layers = base_model.model.layers  # Transformer layers
    dtype = next(iter(base_model.parameters())).dtype

    # === Step 2: Move embeddings and first layer to device ===
    base_model.model.embed_tokens = base_model.model.embed_tokens.to(dev)
    base_model.model.norm = base_model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    # === Step 3: Capture inputs to first layer (Catcher pattern) ===
    print("\n[1/3] Capturing calibration inputs to first transformer layer...")
    nsamples = len(calibration_data)
    seqlen = (
        calibration_data[0].shape[1]
        if len(calibration_data[0].shape) > 1
        else base_model.config.max_position_embeddings
    )
    hidden_size = base_model.config.hidden_size

    # Pre-allocate storage for first layer inputs
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        """Intercepts inputs to first transformer layer."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError  # Stop forward pass after capturing input

    # Replace first layer with Catcher
    layers[0] = Catcher(layers[0])

    # Run calibration data to capture inputs
    for batch in calibration_data:
        try:
            batch = batch.to(dev)
            base_model(batch)
        except ValueError:
            pass  # Expected - Catcher stops forward pass

    # Restore original first layer
    layers[0] = layers[0].module

    # Free memory
    layers[0] = layers[0].cpu()
    base_model.model.embed_tokens = base_model.model.embed_tokens.cpu()
    base_model.model.norm = base_model.model.norm.cpu()
    torch.cuda.empty_cache()

    print(f"  ✓ Captured inputs: {inps.shape}")

    # === Step 4: Move fine-tuned models to same device (for task vector computation) ===
    print("\n[2/3] Loading fine-tuned models...")
    for idx, ft_model in enumerate(finetuned_models):
        finetuned_models[idx] = ft_model.to(dev)
        finetuned_models[idx].eval()
    print(f"  ✓ Loaded {len(finetuned_models)} fine-tuned models")

    # Pre-allocate storage for layer outputs
    outs = torch.zeros_like(inps)
    attention_mask = cache.get("attention_mask", None)

    # Create attention mask if not captured
    if attention_mask is None:
        # Create causal attention mask
        attention_mask = torch.ones((nsamples, seqlen), dtype=torch.long, device=dev)

    print("\n[3/3] Processing layers sequentially...")
    print("=" * 80)

    # === Step 5: Process each transformer layer sequentially ===
    num_layers = len(layers)
    for i in range(num_layers):
        print(f"\n[Layer {i+1}/{num_layers}]")

        # Move current layer to device
        layer = layers[i].to(dev)

        # Find all linear layers in this transformer block
        full = find_layers(layer)

        # Get corresponding layers from fine-tuned models
        ft_layers = []
        for ft_model in finetuned_models:
            ft_layer = ft_model.model.layers[i].to(dev)
            ft_layers.append(ft_layer)

        # Process each linear layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        for layer_name in full.keys():
            print(f"  {layer_name}...", end=" ", flush=True)

            base_linear = full[layer_name]
            layer_shape = base_linear.weight.shape

            # === Step 5a: Create SparseGPT pruner for this layer ===
            pruner = SparseGPTTaskVector(layer_shape, device=dev)

            # === Step 5b: Accumulate Hessian from CURRENT base model state ===
            # This is critical! We use the base model AFTER previous layers were merged
            def add_batch_hook(module, inp, out):
                pruner.add_batch(inp[0].data)

            handle = base_linear.register_forward_hook(add_batch_hook)

            # Run calibration samples through current base model
            for j in range(nsamples):
                # Get attention mask slice for this sample
                attn_mask_slice = (
                    attention_mask[j : j + 1] if attention_mask is not None else None
                )
                output = layer(inps[j].unsqueeze(0), attention_mask=attn_mask_slice)
                # Layer returns tuple (hidden_states, ) or just hidden_states
                if output is None:
                    raise RuntimeError(
                        f"Layer {i} forward pass returned None. "
                        "This may indicate an issue with the model or attention mask."
                    )
                if isinstance(output, tuple):
                    _ = output[0]
                else:
                    _ = output

            handle.remove()

            # === Step 5c: Compute task vectors (fine-tuned - base) ===
            task_vectors = []
            for ft_layer in ft_layers:
                ft_linear_layers = find_layers(ft_layer)
                if layer_name not in ft_linear_layers:
                    raise RuntimeError(
                        f"Layer '{layer_name}' not found in fine-tuned model. "
                        f"Available layers: {list(ft_linear_layers.keys())}"
                    )
                ft_linear = ft_linear_layers[layer_name]
                task_vector = ft_linear.weight.data - base_linear.weight.data
                task_vectors.append(task_vector)

            # === Step 5d: Prune each task vector using SparseGPT ===
            pruned_task_vectors = []
            for tv in task_vectors:
                pruned_tv = pruner.fasterprune(
                    task_vector=tv,
                    density=density,
                    blocksize=blocksize,
                    percdamp=percdamp,
                )
                pruned_task_vectors.append(pruned_tv)

            # === Step 5e: Merge pruned task vectors (simple average) ===
            if len(pruned_task_vectors) > 0:
                merged_tv = torch.stack(pruned_task_vectors).mean(dim=0)

                # === Step 5f: ADD TO BASE MODEL (CRITICAL!) ===
                # This updates the base model so next layer sees correct inputs
                base_linear.weight.data = base_linear.weight.data + merged_tv

            # Free memory
            pruner.free()

            print("✓")

        # === Step 6: Compute outputs from updated layer (inputs for next layer) ===
        for j in range(nsamples):
            # Get attention mask slice for this sample
            attn_mask_slice = (
                attention_mask[j : j + 1] if attention_mask is not None else None
            )
            output = layer(inps[j].unsqueeze(0), attention_mask=attn_mask_slice)
            # Layer returns tuple (hidden_states, ) or just hidden_states
            if output is None:
                raise RuntimeError(
                    f"Layer {i} forward pass returned None during output computation. "
                    "This may indicate an issue with the model or attention mask."
                )
            if isinstance(output, tuple):
                outs[j] = output[0]
            else:
                outs[j] = output

        # Move processed layer back to CPU
        layers[i] = layer.cpu()
        for ft_layer in ft_layers:
            ft_layer.cpu()
        del layer, ft_layers
        torch.cuda.empty_cache()

        # Swap: outputs become inputs for next layer
        inps, outs = outs, inps

    # === Step 7: Cleanup ===
    base_model.config.use_cache = use_cache

    print("\n" + "=" * 80)
    print("✓ SEQUENTIAL MERGING COMPLETE!")
    print("=" * 80)

    return base_model


def load_calibration_data(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    num_samples: int = 128,
    seq_length: int = 512,
) -> List[torch.Tensor]:
    """
    Load and tokenize calibration data from the same dataset used for fine-tuning.

    IMPORTANT: Use the SAME datasets that were used to fine-tune the models!
    This ensures the Hessian is computed on the correct data distribution.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "ruslanmv/ai-medical-chatbot")
        tokenizer: Tokenizer for the model
        num_samples: Number of calibration samples
        seq_length: Maximum sequence length

    Returns:
        List of tokenized tensors [batch_size=1, seq_len]
    """
    print(f"Loading calibration data from {dataset_name}...")
    print(f"  ℹ️  Using same dataset as fine-tuning (CRITICAL for accurate Hessian)")

    try:
        # Load only the first num_samples to avoid downloading entire dataset
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")

        # Auto-detect text field (try common field names)
        text_field = None
        possible_fields = [
            "text",  # Standard text field
            "content",  # Alternative text field
            "conversation",  # Chat/conversation datasets
            "messages",  # Chat datasets
            "prompt",  # Instruction datasets
            "input",  # Instruction datasets
            "Patient",  # Medical datasets (ruslanmv/ai-medical-chatbot)
            "Context",  # Mental health datasets (Amod/mental_health_counseling_conversations)
        ]

        for field in possible_fields:
            if field in dataset.column_names:
                text_field = field
                print(f"  ✓ Using text field: '{text_field}'")
                break

        if text_field is None:
            # Fallback: use first string field
            for field in dataset.column_names:
                if isinstance(dataset[0][field], (str, list)):
                    text_field = field
                    print(f"  ⚠️  Auto-detected text field: '{text_field}'")
                    break

        if text_field is None:
            raise ValueError(
                f"Could not find text field in dataset {dataset_name}. "
                f"Available columns: {dataset.column_names}"
            )

        # Sample and tokenize
        calibration_data = []
        skipped = 0

        for i in range(len(dataset)):
            if len(calibration_data) >= num_samples:
                break

            try:
                text = dataset[i][text_field]

                # Handle conversation format (list of messages)
                if isinstance(text, list):
                    # Join conversation turns
                    text = " ".join(str(msg) for msg in text if msg)

                # Skip empty texts
                if not text or len(str(text).strip()) == 0:
                    skipped += 1
                    continue

                # Tokenize
                tokens = tokenizer(
                    str(text),
                    return_tensors="pt",
                    max_length=seq_length,
                    truncation=True,
                    padding="max_length",
                )

                calibration_data.append(tokens.input_ids)

                if (len(calibration_data)) % 20 == 0:
                    print(
                        f"    Progress: {len(calibration_data)}/{num_samples} samples tokenized"
                    )

            except Exception as e:
                skipped += 1
                if skipped <= 3:  # Only show first few errors
                    print(f"    ⚠️  Skipping sample {i}: {e}")
                continue

        if skipped > 0:
            print(f"  ℹ️  Skipped {skipped} invalid samples")

        print(f"  ✓ Loaded {len(calibration_data)} calibration samples")

        if len(calibration_data) < num_samples:
            print(f"  ⚠️  Only got {len(calibration_data)}/{num_samples} samples")
            print(f"     Hessian quality may be reduced")

        return calibration_data

    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        print(f"     Dataset: {dataset_name}")
        print(f"     This usually means:")
        print(f"       1. Dataset name is incorrect")
        print(f"       2. Dataset requires authentication")
        print(f"       3. Network connection issue")
        print(f"\n  ⚠️  FALLING BACK TO DUMMY DATA (Hessian will be INACCURATE!)")

        # Generate dummy data as last resort
        dummy_data = []
        for _ in range(num_samples):
            tokens = torch.randint(0, tokenizer.vocab_size, (1, seq_length))
            dummy_data.append(tokens)

        return dummy_data


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LLaMA models using SparseGPT")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_models", type=str, nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load models
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )

    finetuned_models = []
    for ft_path in args.finetuned_models:
        print(f"Loading fine-tuned model: {ft_path}")
        ft_model = AutoModelForCausalLM.from_pretrained(
            ft_path, torch_dtype=torch.float16
        )
        finetuned_models.append(ft_model)

    # Load calibration data
    calibration_data = load_calibration_data(args.dataset, tokenizer, args.num_samples)

    # Merge
    merged_model = llama_sparse_merge_sequential(
        base_model=base_model,
        finetuned_models=finetuned_models,
        calibration_data=calibration_data,
        density=args.density,
        device=args.device,
    )

    # Save
    print(f"\nSaving merged model to {args.output}")
    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✓ Done!")
