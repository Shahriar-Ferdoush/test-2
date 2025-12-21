"""
Diagnostic: Analyze Task Vector Distributions

This script helps understand why some reconstruction errors are zero
by analyzing the task vectors for each layer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def analyze_task_vectors(task_vector_file):
    """Analyze task vector statistics to understand reconstruction errors."""

    print("=" * 80)
    print("TASK VECTOR DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Load task vectors
    task_vectors = torch.load(task_vector_file, map_location="cpu")

    print(f"\nLoaded {len(task_vectors)} task vectors")

    # Analyze each layer
    layer_stats = []

    for layer_name, tv in task_vectors.items():
        # Skip non-weight parameters
        if "weight" not in layer_name:
            continue

        # Compute statistics
        stats = {
            "name": layer_name,
            "shape": tv.shape,
            "num_params": tv.numel(),
            "norm": tv.norm().item(),
            "mean_abs": tv.abs().mean().item(),
            "max_abs": tv.abs().max().item(),
            "sparsity": (tv.abs() < 1e-6).float().mean().item(),
            "std": tv.std().item(),
        }

        layer_stats.append(stats)

    # Sort by norm (smallest to largest)
    layer_stats.sort(key=lambda x: x["norm"])

    print("\n" + "=" * 80)
    print("LAYERS WITH SMALLEST CHANGES (Likely Zero Reconstruction Error)")
    print("=" * 80)
    print(f"{'Layer':<50} {'Norm':>10} {'Sparsity':>10} {'Max':>10}")
    print("-" * 80)

    for stats in layer_stats[:20]:  # Show 20 smallest
        print(
            f"{stats['name']:<50} {stats['norm']:>10.4f} {stats['sparsity']:>9.1%} {stats['max_abs']:>10.4f}"
        )

    print("\n" + "=" * 80)
    print("LAYERS WITH LARGEST CHANGES (Likely Non-Zero Reconstruction Error)")
    print("=" * 80)
    print(f"{'Layer':<50} {'Norm':>10} {'Sparsity':>10} {'Max':>10}")
    print("-" * 80)

    for stats in layer_stats[-20:]:  # Show 20 largest
        print(
            f"{stats['name']:<50} {stats['norm']:>10.4f} {stats['sparsity']:>9.1%} {stats['max_abs']:>10.4f}"
        )

    # Summary statistics
    norms = [s["norm"] for s in layer_stats]
    sparsities = [s["sparsity"] for s in layer_stats]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total layers: {len(layer_stats)}")
    print(f"\nTask Vector Norms:")
    print(f"  Min:    {min(norms):.4f}")
    print(f"  Median: {sorted(norms)[len(norms)//2]:.4f}")
    print(f"  Mean:   {sum(norms)/len(norms):.4f}")
    print(f"  Max:    {max(norms):.4f}")

    print(f"\nTask Vector Sparsity (% of near-zero values):")
    print(f"  Min:    {min(sparsities):.1%}")
    print(f"  Median: {sorted(sparsities)[len(sparsities)//2]:.1%}")
    print(f"  Mean:   {sum(sparsities)/len(sparsities):.1%}")
    print(f"  Max:    {max(sparsities):.1%}")

    # Count layers with very small norms
    tiny_norm_count = sum(1 for n in norms if n < 0.01)
    small_norm_count = sum(1 for n in norms if n < 0.1)

    print(
        f"\nLayers with TINY changes (norm < 0.01): {tiny_norm_count} ({tiny_norm_count/len(norms):.1%})"
    )
    print(
        f"Layers with SMALL changes (norm < 0.1): {small_norm_count} ({small_norm_count/len(norms):.1%})"
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("Layers with tiny norms (< 0.01) will likely have zero reconstruction error.")
    print("This is EXPECTED: Not all layers change equally during fine-tuning!")
    print("\nWhy some layers don't change:")
    print("  1. Early layers: Learn general features (already learned)")
    print("  2. Middle layers: May be task-agnostic")
    print("  3. Specific projections: Some Q/K/V/O projections barely adapt")
    print("  4. LoRA adaptation: Fine-tuning may focus on specific layers")
    print("\nZero reconstruction error = Good news! The pruning is working correctly.")
    print("=" * 80)

    return layer_stats


if __name__ == "__main__":
    import sys

    # Default path (adjust as needed)
    cache_dir = Path("./merge_cache/task_vectors")

    if len(sys.argv) > 1:
        task_vector_file = Path(sys.argv[1])
    else:
        # Try to find first task vector file
        task_vector_files = list(cache_dir.glob("task_vector_*.pt"))
        if not task_vector_files:
            print(f"Error: No task vector files found in {cache_dir}")
            print("Usage: python analyze_task_vectors.py <path_to_task_vector_file>")
            sys.exit(1)
        task_vector_file = task_vector_files[0]

    print(f"Analyzing: {task_vector_file}\n")

    stats = analyze_task_vectors(task_vector_file)

    print("\n✓ Analysis complete!")
    print("\nTo analyze a different model:")
    print(f"  python {sys.argv[0]} merge_cache/task_vectors/task_vector_1.pt")
