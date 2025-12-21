"""
Example: Sequential Task Vector Pruning with Accurate Input Propagation

This demonstrates the new SparseGPTTaskVector class and sequential_layer_pruning
function that match the SparseGPT architecture.

Key Difference from Parallel Processing:
----------------------------------------
- Parallel: All layers see inputs from unchanged base model (INACCURATE)
- Sequential: Each layer sees inputs after previous layer was pruned (ACCURATE)

This matters for deep models where early layer changes affect later layers!
"""

import torch
import torch.nn as nn

from sparsegpt_task_vector import (
    SparseGPTTaskVector,
    compute_task_vector,
    find_layers,
    sequential_layer_pruning,
)


def create_simple_model(input_dim=512, hidden_dim=256, num_layers=3):
    """Create a simple MLP for demonstration."""
    layers = []
    in_dim = input_dim
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, 10))  # Output layer
    return nn.Sequential(*layers)


def create_calibration_data(num_samples=128, seq_len=32, input_dim=512):
    """Create dummy calibration data."""
    return [torch.randn(1, seq_len, input_dim) for _ in range(num_samples)]


# ============================================================================
# Example 1: Single Layer Pruning (Direct SparseGPTTaskVector Usage)
# ============================================================================


def example_single_layer():
    """
    Demonstrate pruning a single layer using SparseGPTTaskVector class.
    Follows the exact pattern from SparseGPT.
    """
    print("=" * 80)
    print("EXAMPLE 1: Single Layer Pruning")
    print("=" * 80)

    # Create models
    base_model = create_simple_model()
    ft_model = create_simple_model()  # Fine-tuned (simulated)

    # Get first linear layer
    base_layer = base_model[0]  # First Linear layer
    ft_layer = ft_model[0]

    # Create calibration data
    calibration_data = create_calibration_data(num_samples=64, input_dim=512)

    print(f"\nLayer shape: {base_layer.weight.shape}")
    print(f"Calibration samples: {len(calibration_data)}")

    # === Step 1: Create SparseGPTTaskVector instance ===
    layer_shape = (base_layer.out_features, base_layer.in_features)
    pruner = SparseGPTTaskVector(layer_shape, device="cpu")

    # === Step 2: Accumulate Hessian ===
    print("\n[1/4] Accumulating Hessian from calibration data...")
    for batch in calibration_data:
        # Get inputs to this layer
        inputs = batch[:, 0, :]  # [1, input_dim]
        pruner.add_batch(inputs)
    print(f"  ✓ Hessian accumulated from {pruner.nsamples} samples")

    # === Step 3: Compute task vector ===
    print("\n[2/4] Computing task vector...")
    task_vector = compute_task_vector(ft_layer, base_layer)
    print(f"  Task vector shape: {task_vector.shape}")
    print(f"  Task vector norm: {task_vector.norm():.4f}")

    # === Step 4: Prune with error correction ===
    print("\n[3/4] Pruning with SparseGPT error correction...")
    pruned_tv = pruner.fasterprune(
        task_vector=task_vector,
        density=0.2,  # Keep 20%, prune 80%
        blocksize=128,
        percdamp=0.01,
        rescale=False,
    )

    # === Step 5: Verify sparsity ===
    print("\n[4/4] Verifying results...")
    sparsity = (pruned_tv == 0).float().mean()
    print(f"  Actual sparsity: {sparsity:.1%}")
    print(f"  Pruned TV norm: {pruned_tv.norm():.4f}")

    # Free memory
    pruner.free()

    print("\n✓ Single layer pruning complete!\n")


# ============================================================================
# Example 2: Sequential Multi-Layer Pruning (NEW FEATURE!)
# ============================================================================


def example_sequential_layers():
    """
    Demonstrate sequential layer pruning with accurate input propagation.
    This is the key innovation - each layer sees inputs from the updated model!
    """
    print("=" * 80)
    print("EXAMPLE 2: Sequential Multi-Layer Pruning")
    print("=" * 80)

    # Create models
    base_model = create_simple_model(num_layers=3)
    ft_model1 = create_simple_model(num_layers=3)
    ft_model2 = create_simple_model(num_layers=3)

    # Create calibration data
    calibration_data = create_calibration_data(num_samples=32, input_dim=512)

    # Find all linear layers
    linear_layers = find_layers(base_model, [nn.Linear])
    layer_names = list(linear_layers.keys())[:3]  # First 3 linear layers

    print(f"\nModels: 1 base + 2 fine-tuned")
    print(f"Layers to prune: {layer_names}")
    print(f"Calibration samples: {len(calibration_data)}")

    # === Sequential pruning with accurate input propagation ===
    print("\n" + "=" * 80)
    print("Starting sequential pruning...")
    print("=" * 80)

    pruned_tvs = sequential_layer_pruning(
        base_model=base_model,
        finetuned_models=[ft_model1, ft_model2],
        calibration_loader=calibration_data,
        layer_names=layer_names,
        density=0.2,  # Keep 20%
        blocksize=64,
        percdamp=0.01,
        rescale=False,
    )

    # === Verify results ===
    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    for layer_name in layer_names:
        tvs = pruned_tvs[layer_name]
        if len(tvs) > 0:
            avg_sparsity = sum((tv == 0).float().mean() for tv in tvs) / len(tvs)
            avg_norm = sum(tv.norm() for tv in tvs) / len(tvs)
            print(f"\n{layer_name}:")
            print(f"  Avg sparsity: {avg_sparsity:.1%}")
            print(f"  Avg norm: {avg_norm:.4f}")

    print("\n✓ Sequential multi-layer pruning complete!\n")


# ============================================================================
# Example 3: Comparison - Sequential vs Parallel Processing
# ============================================================================


def example_compare_sequential_vs_parallel():
    """
    Compare sequential (accurate) vs parallel (inaccurate) processing.
    Shows why sequential matters for model quality.
    """
    print("=" * 80)
    print("EXAMPLE 3: Sequential vs Parallel Comparison")
    print("=" * 80)

    # TODO: Implement parallel processing for comparison
    # This would show that parallel processing gives different (worse) results
    # because later layers see inaccurate inputs

    print("\n⚠ Parallel processing demonstration coming soon!")
    print("Key point: Sequential processing ensures each layer sees")
    print("inputs from the updated model, giving accurate Hessians.")
    print()


if __name__ == "__main__":
    # Run examples
    example_single_layer()
    example_sequential_layers()
    example_compare_sequential_vs_parallel()

    print("=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. SparseGPTTaskVector class matches SparseGPT architecture exactly")
    print("2. Sequential processing ensures accurate Hessian computation")
    print("3. Each layer sees inputs after previous layers were pruned")
    print("4. This is critical for maintaining model quality!")
    print()
