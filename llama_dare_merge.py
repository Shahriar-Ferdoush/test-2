"""
LLaMA Model Merging with DARE (Drop And REscale)

Simple layer-by-layer merging using DARE method:
1. Iterate through each layer
2. Compute task vectors (fine-tuned - base)
3. Apply DARE: Random dropout + rescale
4. No sequential dependency - can process layers independently

Reference: Yu et al. (2023) - "Language Models are Super Mario:
           Absorbing Abilities from Homologous Models as a Free Lunch"
"""

import time
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from dare_utils import DARE


def find_layers(module: nn.Module, layer_types: List = None) -> dict:
    """Find all layers of specified types."""
    if layer_types is None:
        layer_types = [nn.Linear]

    layers = {}
    for name, mod in module.named_modules():
        if any(isinstance(mod, t) for t in layer_types):
            layers[name] = mod
    return layers


@torch.no_grad()
def llama_dare_merge(
    base_model: AutoModelForCausalLM,
    finetuned_models: List[AutoModelForCausalLM],
    density: float = 0.2,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    """
    Merge fine-tuned models using DARE method.

    Args:
        base_model: Base model to merge into
        finetuned_models: List of fine-tuned models
        density: Fraction of task vector weights to keep (0.2 = keep 20%)
        device: Device for computation

    Returns:
        Merged base model
    """
    print("=" * 80)
    print("DARE-BASED MODEL MERGING (Random Dropout + Rescale)")
    print("=" * 80)
    print(f"Base model: {type(base_model).__name__}")
    print(f"Fine-tuned models: {len(finetuned_models)}")
    print(f"Density: {density:.1%} (dropping {1-density:.1%} randomly)")
    print(f"Device: {device}")
    print("=" * 80)

    dev = torch.device(device)

    # Validate model compatibility
    print("\nValidating model architecture...")
    base_layer_count = len(list(base_model.model.layers))
    for idx, ft_model in enumerate(finetuned_models):
        ft_layer_count = len(list(ft_model.model.layers))
        if ft_layer_count != base_layer_count:
            raise ValueError(
                f"Fine-tuned model {idx} has {ft_layer_count} layers, "
                f"but base model has {base_layer_count} layers."
            )
    print(f"  ✓ All models have {base_layer_count} transformer layers")

    # Move models to device
    base_model = base_model.to(dev)
    base_model.eval()

    finetuned_models = [ft.to(dev).eval() for ft in finetuned_models]

    # Find all linear layers in base model
    base_layers = find_layers(base_model)

    print(f"\nProcessing {len(base_layers)} layers...")
    start_time = time.time()

    # Process each layer independently
    for idx, (layer_name, base_layer) in enumerate(base_layers.items()):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(base_layers)}] {layer_name}")

        # Get base weights
        base_weights = base_layer.weight.data

        # Compute task vectors from all fine-tuned models
        ft_weights_list = []
        for ft_model in finetuned_models:
            ft_layers = find_layers(ft_model)
            if layer_name in ft_layers:
                ft_weights_list.append(ft_layers[layer_name].weight.data)

        if not ft_weights_list:
            continue  # Skip if layer not found in fine-tuned models

        # Models already on device, no need to move weights again
        # Apply DARE merging
        dare_merger = DARE()
        merged_weights = dare_merger.merge(
            weights=[1.0] * len(ft_weights_list),  # Equal weights for all models
            base_model_parameters=base_weights,  # Already on dev
            ft_models_parameters=ft_weights_list,  # Already on dev
            densities=[density] * len(ft_weights_list),  # Same density for all
            device=dev,
            use_sparsegpt=False,  # Random dropout
        )

        # Update base model (merged_weights already on dev, same as base)
        base_layer.weight.data = merged_weights

    elapsed = time.time() - start_time
    print(f"\n✓ Merging complete in {elapsed:.1f}s")
    print("=" * 80)

    return base_model


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LLaMA models using DARE")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_models", type=str, nargs="+", required=True)
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

    # Merge
    merged_model = llama_dare_merge(
        base_model=base_model,
        finetuned_models=finetuned_models,
        density=args.density,
        device=args.device,
    )

    # Save
    print(f"\nSaving merged model to {args.output}")
    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✓ Done!")
