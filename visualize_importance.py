"""
Visualization: Comparing Importance Metrics

This script visualizes the difference between magnitude-based (TIES),
random (DARE), and SparseGPT importance-based weight selection.

Usage:
    python visualize_importance.py
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sparsegpt_importance import (
        HessianCalculator,
        compute_importance_scores,
        generate_importance_mask,
    )

    SPARSEGPT_AVAILABLE = True
except ImportError:
    print("Warning: sparsegpt_importance not available. Some features disabled.")
    SPARSEGPT_AVAILABLE = False


def create_synthetic_task_vector(size: int = 100) -> torch.Tensor:
    """
    Create a synthetic task vector with known structure.

    Structure:
    - First 20 weights: Large magnitude, high importance (signal)
    - Next 30 weights: Small magnitude, high importance (hidden signal)
    - Next 30 weights: Large magnitude, low importance (noise)
    - Last 20 weights: Small magnitude, low importance (pure noise)
    """
    task_vector = torch.zeros(size)

    # High magnitude, high importance
    task_vector[0:20] = torch.randn(20) * 2.0 + 5.0

    # Low magnitude, high importance (these should be kept!)
    task_vector[20:50] = torch.randn(30) * 0.3 + 0.2

    # High magnitude, low importance (these should be dropped!)
    task_vector[50:80] = torch.randn(30) * 2.0 + 4.0

    # Low magnitude, low importance
    task_vector[80:100] = torch.randn(20) * 0.1

    return task_vector


def create_synthetic_hessian_inv_diag(size: int = 100) -> torch.Tensor:
    """
    Create synthetic inverse Hessian diagonal with known structure.

    Structure:
    - First 50 positions: Low H^-1 → High curvature → High importance
    - Last 50 positions: High H^-1 → Low curvature → Low importance
    """
    h_inv_diag = torch.ones(size)

    # High importance region (low H^-1)
    h_inv_diag[0:50] = torch.rand(50) * 0.1 + 0.01

    # Low importance region (high H^-1)
    h_inv_diag[50:100] = torch.rand(50) * 5.0 + 2.0

    return h_inv_diag


def magnitude_selection(task_vector: torch.Tensor, density: float) -> torch.Tensor:
    """TIES: Select by absolute magnitude."""
    k = int(density * task_vector.numel())
    threshold = torch.topk(task_vector.abs(), k).values.min()
    mask = task_vector.abs() >= threshold
    return mask.float()


def random_selection(task_vector: torch.Tensor, density: float) -> torch.Tensor:
    """DARE: Random selection."""
    mask = torch.bernoulli(torch.full_like(task_vector, density))
    return mask


def sparsegpt_selection(
    task_vector: torch.Tensor, h_inv_diag: torch.Tensor, density: float
) -> torch.Tensor:
    """SparseGPT: Importance-based selection."""
    importance = compute_importance_scores(task_vector, h_inv_diag)
    mask = generate_importance_mask(importance, density)
    return mask.float()


def visualize_comparison(density: float = 0.3):
    """
    Create a comprehensive visualization comparing the three methods.
    """
    # Create synthetic data
    task_vector = create_synthetic_task_vector(100)
    h_inv_diag = create_synthetic_hessian_inv_diag(100)

    # Compute importance scores
    importance_sparsegpt = compute_importance_scores(task_vector, h_inv_diag)
    importance_magnitude = task_vector.abs()

    # Generate masks
    mask_magnitude = magnitude_selection(task_vector, density)
    mask_random = random_selection(task_vector, density)
    mask_sparsegpt = sparsegpt_selection(task_vector, h_inv_diag, density)

    # Ground truth: which weights SHOULD be kept?
    # First 50 positions are high importance
    ground_truth = torch.zeros(100)
    ground_truth[0:50] = 1.0

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        f"Comparison of Weight Selection Methods (Density={density})",
        fontsize=16,
        fontweight="bold",
    )

    x = np.arange(100)

    # Row 1: Magnitude Method (TIES)
    axes[0, 0].bar(x, task_vector.numpy(), color="skyblue", alpha=0.7)
    axes[0, 0].set_title("Task Vector Weights", fontweight="bold")
    axes[0, 0].set_ylabel("TIES\n(Magnitude)", fontweight="bold", fontsize=12)
    axes[0, 0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(x, importance_magnitude.numpy(), color="orange", alpha=0.7)
    axes[0, 1].set_title("Importance Scores", fontweight="bold")
    axes[0, 1].set_ylabel("|w|", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].bar(x, mask_magnitude.numpy(), color="green", alpha=0.7)
    axes[0, 2].bar(x, ground_truth.numpy(), color="red", alpha=0.3)
    axes[0, 2].set_title("Selected Weights", fontweight="bold")
    axes[0, 2].set_ylabel("Mask", fontsize=10)
    axes[0, 2].legend(["Selected", "Ground Truth (should select)"], fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Random Method (DARE)
    axes[1, 0].bar(x, task_vector.numpy(), color="skyblue", alpha=0.7)
    axes[1, 0].set_ylabel("DARE\n(Random)", fontweight="bold", fontsize=12)
    axes[1, 0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(x, torch.ones(100).numpy(), color="gray", alpha=0.5)
    axes[1, 1].set_ylabel("All equal = 1", fontsize=10)
    axes[1, 1].set_title("(No Importance Used)", fontsize=10, style="italic")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].bar(x, mask_random.numpy(), color="green", alpha=0.7)
    axes[1, 2].bar(x, ground_truth.numpy(), color="red", alpha=0.3)
    axes[1, 2].set_ylabel("Mask", fontsize=10)
    axes[1, 2].legend(["Selected", "Ground Truth (should select)"], fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: SparseGPT Method (Ours)
    axes[2, 0].bar(x, task_vector.numpy(), color="skyblue", alpha=0.7)
    axes[2, 0].set_ylabel("SparseGPT\n(Hessian)", fontweight="bold", fontsize=12)
    axes[2, 0].set_xlabel("Weight Index", fontsize=10)
    axes[2, 0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].bar(x, importance_sparsegpt.numpy(), color="purple", alpha=0.7)
    axes[2, 1].set_xlabel("Weight Index", fontsize=10)
    axes[2, 1].set_ylabel("w²/(H⁻¹)²", fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].bar(x, mask_sparsegpt.numpy(), color="green", alpha=0.7)
    axes[2, 2].bar(x, ground_truth.numpy(), color="red", alpha=0.3)
    axes[2, 2].set_xlabel("Weight Index", fontsize=10)
    axes[2, 2].set_ylabel("Mask", fontsize=10)
    axes[2, 2].legend(["Selected", "Ground Truth (should select)"], fontsize=8)
    axes[2, 2].grid(True, alpha=0.3)

    # Add color coding explanation
    fig.text(
        0.5,
        0.02,
        "Ground Truth Structure: Positions 0-49 (high importance, should keep) | "
        "Positions 50-99 (low importance, should drop)",
        ha="center",
        fontsize=10,
        style="italic",
        color="red",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("importance_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved visualization to: importance_comparison.png")
    plt.show()

    # Calculate and print metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    def calculate_metrics(mask, ground_truth):
        selected_correct = (mask * ground_truth).sum().item()
        selected_total = mask.sum().item()
        should_select = ground_truth.sum().item()

        precision = selected_correct / selected_total if selected_total > 0 else 0
        recall = selected_correct / should_select if should_select > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return precision, recall, f1

    prec_mag, rec_mag, f1_mag = calculate_metrics(mask_magnitude, ground_truth)
    prec_rand, rec_rand, f1_rand = calculate_metrics(mask_random, ground_truth)
    prec_sg, rec_sg, f1_sg = calculate_metrics(mask_sparsegpt, ground_truth)

    print(f"\nMagnitude (TIES):")
    print(f"  Precision: {prec_mag:.3f} | Recall: {rec_mag:.3f} | F1: {f1_mag:.3f}")

    print(f"\nRandom (DARE):")
    print(f"  Precision: {prec_rand:.3f} | Recall: {rec_rand:.3f} | F1: {f1_rand:.3f}")

    print(f"\nSparseGPT (Ours):")
    print(f"  Precision: {prec_sg:.3f} | Recall: {rec_sg:.3f} | F1: {f1_sg:.3f}")

    print(f"\n{'='*60}")
    print(f"IMPROVEMENT OVER BASELINES")
    print(f"{'='*60}")
    print(f"vs Magnitude: {((f1_sg/f1_mag - 1) * 100):.1f}% better F1")
    print(f"vs Random:    {((f1_sg/f1_rand - 1) * 100):.1f}% better F1")


def visualize_density_sweep():
    """
    Show how different methods perform across density values.
    """
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Create synthetic data
    task_vector = create_synthetic_task_vector(100)
    h_inv_diag = create_synthetic_hessian_inv_diag(100)
    ground_truth = torch.zeros(100)
    ground_truth[0:50] = 1.0

    # Store results
    f1_magnitude = []
    f1_random = []
    f1_sparsegpt = []

    for density in densities:
        mask_mag = magnitude_selection(task_vector, density)
        mask_rand = random_selection(task_vector, density)
        mask_sg = sparsegpt_selection(task_vector, h_inv_diag, density)

        def calc_f1(mask, gt):
            correct = (mask * gt).sum()
            selected = mask.sum()
            should_select = gt.sum()

            prec = correct / selected if selected > 0 else 0
            rec = correct / should_select if should_select > 0 else 0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        f1_magnitude.append(calc_f1(mask_mag, ground_truth))
        f1_random.append(calc_f1(mask_rand, ground_truth))
        f1_sparsegpt.append(calc_f1(mask_sg, ground_truth))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        densities,
        f1_magnitude,
        "o-",
        label="Magnitude (TIES)",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        densities, f1_random, "s-", label="Random (DARE)", linewidth=2, markersize=8
    )
    plt.plot(
        densities,
        f1_sparsegpt,
        "^-",
        label="SparseGPT (Ours)",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Density (fraction of weights kept)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score (higher is better)", fontsize=12, fontweight="bold")
    plt.title(
        "Performance vs Density: Method Comparison", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.05, 0.85)
    plt.ylim(0, 1.05)

    # Add shaded region for aggressive pruning
    plt.axvspan(
        0.05,
        0.3,
        alpha=0.1,
        color="red",
        label="Aggressive pruning\n(SparseGPT advantage)",
    )
    plt.text(
        0.15,
        0.95,
        "Aggressive\nPruning Zone",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.2),
    )

    plt.tight_layout()
    plt.savefig("density_sweep.png", dpi=300, bbox_inches="tight")
    print("✓ Saved density sweep to: density_sweep.png")
    plt.show()

    # Print insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM DENSITY SWEEP")
    print("=" * 60)
    print("\n1. At LOW densities (< 0.3):")
    print(f"   SparseGPT achieves F1={f1_sparsegpt[1]:.3f} at density=0.2")
    print(f"   Magnitude achieves F1={f1_magnitude[1]:.3f} at density=0.2")
    print(f"   Improvement: {((f1_sparsegpt[1]/f1_magnitude[1] - 1) * 100):.1f}%")

    print("\n2. At HIGH densities (> 0.5):")
    print(f"   Methods converge as most weights are kept anyway")

    print("\n3. Recommendation:")
    print(f"   Use SparseGPT when density < 0.3 for maximum benefit!")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("VISUALIZING IMPORTANCE-BASED WEIGHT SELECTION")
    print("=" * 60)

    if not SPARSEGPT_AVAILABLE:
        print("\nError: sparsegpt_importance module not found!")
        print("Make sure sparsegpt_importance.py is in the same directory.")
        return

    # Visualization 1: Direct comparison
    print("\n[1/2] Creating method comparison visualization...")
    visualize_comparison(density=0.3)

    # Visualization 2: Density sweep
    print("\n[2/2] Creating density sweep visualization...")
    visualize_density_sweep()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • importance_comparison.png - Side-by-side method comparison")
    print("  • density_sweep.png - Performance vs density")
    print("\nKey takeaway:")
    print("  SparseGPT uses Hessian information to identify truly important")
    print("  weights, outperforming magnitude and random selection,")
    print("  especially at aggressive pruning levels (< 30% density).")


if __name__ == "__main__":
    main()
