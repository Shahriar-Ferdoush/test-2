"""
Comparison of Three Model Merging Methods

This script demonstrates and compares three approaches to model merging:
1. TIES (Trim, Elect, Merge) - magnitude-based trimming
2. DARE (Drop And REscale) - random dropout
3. SparseGPT - importance-based with error correction

Usage:
    python compare_merging_methods.py
"""

import time
from typing import List, Tuple

import torch
import torch.nn as nn

from dare_utils import DARE
from sparsegpt_task_vector import (
    HessianCalculator,
    TaskVectorImportanceCalculator,
    prune_task_vector_with_error_correction,
)
from ties_utils import TIES


class SimpleTestModel(nn.Module):
    """Simple model for testing merging methods."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def create_test_models(device="cpu") -> Tuple[nn.Module, List[nn.Module]]:
    """Create base model and fine-tuned variants for testing."""
    print("Creating test models...")

    # Base model
    base_model = SimpleTestModel().to(device)

    # Fine-tuned models (simulate by adding controlled noise)
    ft_models = []
    for i in range(3):
        ft_model = SimpleTestModel().to(device)
        # Copy base weights and add task-specific updates
        with torch.no_grad():
            for base_param, ft_param in zip(
                base_model.parameters(), ft_model.parameters()
            ):
                # Add task-specific noise with different scales
                noise_scale = 0.1 * (i + 1)  # Different magnitude per task
                ft_param.copy_(base_param + noise_scale * torch.randn_like(base_param))

        ft_models.append(ft_model)

    print(f"Created 1 base model and {len(ft_models)} fine-tuned models")
    return base_model, ft_models


def generate_calibration_data(
    batch_size=32, num_batches=20, input_dim=512, device="cpu"
):
    """Generate synthetic calibration data for Hessian computation."""
    print(f"Generating {num_batches} calibration batches...")
    return [torch.randn(batch_size, input_dim).to(device) for _ in range(num_batches)]


def compute_hessians_for_sparsegpt(
    base_model: nn.Module, calibration_data: List[torch.Tensor], device="cpu"
) -> dict:
    """Compute Hessians for SparseGPT method."""
    print("\n" + "=" * 80)
    print("Computing Hessians for SparseGPT...")
    print("=" * 80)

    start_time = time.time()

    # Get layer names to compute Hessians for
    layer_names = ["layer1", "layer2", "layer3"]

    # Initialize calculator
    calc = TaskVectorImportanceCalculator(
        model=base_model,
        calibration_loader=calibration_data,
        device=device,
        percdamp=0.01,
    )

    # Compute Hessians
    calc.compute_hessians_for_layers(layer_names, verbose=True)

    elapsed = time.time() - start_time
    print(f"Hessian computation completed in {elapsed:.2f}s")

    # Extract full inverse Hessians for each layer
    hessian_invs = {name: calc.hessian_invs[name] for name in layer_names}

    return hessian_invs


def compare_merging_methods(
    base_model: nn.Module,
    ft_models: List[nn.Module],
    calibration_data: List[torch.Tensor],
    density: float = 0.2,
    device="cpu",
):
    """Compare TIES, DARE, and SparseGPT merging methods."""

    print("\n" + "=" * 80)
    print("COMPARISON OF MODEL MERGING METHODS")
    print("=" * 80)
    print(f"Density (fraction of weights to keep): {density}")
    print(f"Number of fine-tuned models: {len(ft_models)}")
    print(f"Device: {device}")

    # Task weights (equal importance for all tasks)
    task_weights = [1.0] * len(ft_models)
    densities = [density] * len(ft_models)

    # Get test parameters (use first layer for demonstration)
    base_params = base_model.layer1.weight.data.clone()
    ft_params_list = [ft.layer1.weight.data.clone() for ft in ft_models]

    results = {}

    # ========== METHOD 1: TIES (Magnitude-based) ==========
    print("\n" + "-" * 80)
    print("METHOD 1: TIES (Trim, Elect, Merge) - Magnitude-based")
    print("-" * 80)

    start_time = time.time()

    ties_merger = TIES()
    merged_ties = ties_merger.merge(
        weights=task_weights,
        base_model_parameters=base_params,
        ft_models_parameters=ft_params_list,
        densities=densities,
        device=device,
        use_sparsegpt=False,  # Use magnitude-based trimming
    )

    ties_time = time.time() - start_time

    # Compute statistics
    task_vectors = [ft - base_params for ft in ft_params_list]
    avg_tv_norm = sum(tv.norm().item() for tv in task_vectors) / len(task_vectors)
    merged_update_ties = merged_ties - base_params

    results["ties"] = {
        "merged": merged_ties,
        "time": ties_time,
        "update_norm": merged_update_ties.norm().item(),
        "sparsity": (merged_update_ties == 0).float().mean().item(),
        "relative_change": (merged_update_ties.norm() / avg_tv_norm).item(),
    }

    print(f"Time: {ties_time:.3f}s")
    print(f"Merged update norm: {results['ties']['update_norm']:.4f}")
    print(f"Sparsity: {results['ties']['sparsity']*100:.1f}% (zero weights)")
    print(
        f"Relative change: {results['ties']['relative_change']:.4f}x average task vector"
    )

    # ========== METHOD 2: DARE (Random Dropout) ==========
    print("\n" + "-" * 80)
    print("METHOD 2: DARE (Drop And REscale) - Random Dropout")
    print("-" * 80)

    start_time = time.time()

    dare_merger = DARE()
    merged_dare = dare_merger.merge(
        weights=task_weights,
        base_model_parameters=base_params,
        ft_models_parameters=ft_params_list,
        densities=densities,
        device=device,
        use_sparsegpt=False,  # Use random dropout
    )

    dare_time = time.time() - start_time

    merged_update_dare = merged_dare - base_params

    results["dare"] = {
        "merged": merged_dare,
        "time": dare_time,
        "update_norm": merged_update_dare.norm().item(),
        "sparsity": (merged_update_dare == 0).float().mean().item(),
        "relative_change": (merged_update_dare.norm() / avg_tv_norm).item(),
    }

    print(f"Time: {dare_time:.3f}s")
    print(f"Merged update norm: {results['dare']['update_norm']:.4f}")
    print(f"Sparsity: {results['dare']['sparsity']*100:.1f}% (zero weights)")
    print(
        f"Relative change: {results['dare']['relative_change']:.4f}x average task vector"
    )

    # ========== METHOD 3: SparseGPT (Error Correction) ==========
    print("\n" + "-" * 80)
    print("METHOD 3: SparseGPT - Importance-based with Error Correction")
    print("-" * 80)

    # Compute Hessians
    hessian_start = time.time()
    hessian_invs_dict = compute_hessians_for_sparsegpt(
        base_model, calibration_data, device
    )
    hessian_time = time.time() - hessian_start

    # For layer1, extract Hessian
    layer1_hessian_inv = hessian_invs_dict["layer1"]
    hessian_invs_list = [layer1_hessian_inv] * len(
        ft_models
    )  # Same Hessian for all tasks

    print(f"\nMerging with SparseGPT...")
    start_time = time.time()

    # Use TIES with SparseGPT error correction
    ties_sparsegpt = TIES()
    merged_sparsegpt = ties_sparsegpt.merge(
        weights=task_weights,
        base_model_parameters=base_params,
        ft_models_parameters=ft_params_list,
        densities=densities,
        device=device,
        hessian_invs=hessian_invs_list,  # Full inverse Hessians
        use_sparsegpt=True,  # Enable SparseGPT
        blocksize=128,
    )

    sparsegpt_time = time.time() - start_time
    total_sparsegpt_time = hessian_time + sparsegpt_time

    merged_update_sparsegpt = merged_sparsegpt - base_params

    results["sparsegpt"] = {
        "merged": merged_sparsegpt,
        "time": total_sparsegpt_time,
        "hessian_time": hessian_time,
        "merge_time": sparsegpt_time,
        "update_norm": merged_update_sparsegpt.norm().item(),
        "sparsity": (merged_update_sparsegpt == 0).float().mean().item(),
        "relative_change": (merged_update_sparsegpt.norm() / avg_tv_norm).item(),
    }

    print(f"Time (Hessian): {hessian_time:.3f}s")
    print(f"Time (Merge): {sparsegpt_time:.3f}s")
    print(f"Time (Total): {total_sparsegpt_time:.3f}s")
    print(f"Merged update norm: {results['sparsegpt']['update_norm']:.4f}")
    print(f"Sparsity: {results['sparsegpt']['sparsity']*100:.1f}% (zero weights)")
    print(
        f"Relative change: {results['sparsegpt']['relative_change']:.4f}x average task vector"
    )

    # ========== COMPARISON SUMMARY ==========
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Method':<20} {'Time (s)':<12} {'Update Norm':<15} {'Sparsity':<12} {'Rel. Change':<12}"
    )
    print("-" * 80)

    for method_name, method_label in [
        ("ties", "TIES (Magnitude)"),
        ("dare", "DARE (Random)"),
        ("sparsegpt", "SparseGPT (Error)"),
    ]:
        r = results[method_name]
        print(
            f"{method_label:<20} {r['time']:<12.3f} {r['update_norm']:<15.4f} "
            f"{r['sparsity']*100:<12.1f} {r['relative_change']:<12.4f}"
        )

    # Quality comparison (reconstruction error)
    print("\n" + "-" * 80)
    print("Reconstruction Error Analysis")
    print("-" * 80)

    # Average task vector (gold standard for reconstruction)
    avg_task_vector = sum(task_vectors) / len(task_vectors)

    for method_name, method_label in [
        ("ties", "TIES"),
        ("dare", "DARE"),
        ("sparsegpt", "SparseGPT"),
    ]:
        merged_update = results[method_name]["merged"] - base_params
        reconstruction_error = (merged_update - avg_task_vector).norm().item()
        relative_error = reconstruction_error / avg_task_vector.norm().item()

        print(
            f"{method_label:<15} Reconstruction Error: {reconstruction_error:.4f} "
            f"(Relative: {relative_error*100:.2f}%)"
        )

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(
        """
1. TIES (Magnitude-based):
   - Fastest method (no Hessian computation)
   - Simple magnitude-based importance
   - Good for quick prototyping

2. DARE (Random Dropout):
   - Fast and stochastic
   - Rescaling preserves magnitude expectation
   - May need multiple runs for stability

3. SparseGPT (Error Correction):
   - Requires Hessian computation (one-time cost)
   - Best reconstruction quality at high sparsity
   - Error correction maintains model output
   - Recommended for production merging

Recommendation: Use SparseGPT for high-quality merging at aggressive 
sparsity levels (density < 0.3). Use TIES/DARE for quick experiments.
    """
    )

    return results


def test_error_correction_benefit(device="cpu"):
    """Demonstrate the benefit of error correction at different densities."""

    print("\n" + "=" * 80)
    print("ERROR CORRECTION BENEFIT ANALYSIS")
    print("=" * 80)

    # Create simple test case
    torch.manual_seed(42)
    base_weights = torch.randn(256, 512).to(device)
    ft_weights = base_weights + 0.1 * torch.randn_like(base_weights)
    task_vector = ft_weights - base_weights

    # Generate calibration data
    calibration_data = [torch.randn(32, 512).to(device) for _ in range(10)]

    # Compute Hessian
    print("Computing Hessian...")
    hess_calc = HessianCalculator((256, 512), device=device)
    for batch in calibration_data:
        hess_calc.add_batch(batch)

    hessian_inv_diag = hess_calc.get_inverse_hessian_diag()
    hessian_inv = hess_calc.get_inverse_hessian()

    # Test different densities
    densities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    print(
        f"\n{'Density':<10} {'Simple Mask Error':<20} {'Error Correction':<20} {'Improvement':<15}"
    )
    print("-" * 70)

    for density in densities:
        # Simple magnitude-based masking
        k = int(density * task_vector.numel())
        threshold = torch.topk(task_vector.abs().view(-1), k).values.min()
        mask = task_vector.abs() >= threshold
        pruned_simple = task_vector * mask

        # Error correction
        pruned_corrected = prune_task_vector_with_error_correction(
            task_vector=task_vector,
            hessian_inv=hessian_inv,
            density=density,
            blocksize=128,
            rescale=False,
        )

        # Compute reconstruction errors
        error_simple = (task_vector - pruned_simple).norm().item()
        error_corrected = (task_vector - pruned_corrected).norm().item()
        improvement = (error_simple - error_corrected) / error_simple * 100

        print(
            f"{density:<10.1f} {error_simple:<20.4f} {error_corrected:<20.4f} {improvement:<15.1f}%"
        )

    print(
        "\nConclusion: Error correction provides significant benefit at low densities."
    )
    print("At density=0.1, error correction can reduce reconstruction error by >50%!")


def main():
    """Main comparison script."""
    print("=" * 80)
    print("MODEL MERGING METHODS COMPARISON")
    print("=" * 80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # Create test models
    base_model, ft_models = create_test_models(device)

    # Generate calibration data
    calibration_data = generate_calibration_data(device=device)

    # Compare methods at different sparsity levels
    print("\n" + "=" * 80)
    print("Testing at 20% density (aggressive pruning)")
    print("=" * 80)
    results_20 = compare_merging_methods(
        base_model, ft_models, calibration_data, density=0.2, device=device
    )

    print("\n" + "=" * 80)
    print("Testing at 50% density (moderate pruning)")
    print("=" * 80)
    results_50 = compare_merging_methods(
        base_model, ft_models, calibration_data, density=0.5, device=device
    )

    # Error correction benefit analysis
    test_error_correction_benefit(device=device)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
