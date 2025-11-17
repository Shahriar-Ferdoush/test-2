"""
Example: Using SparseGPT Importance for Model Merging

This script demonstrates how to merge fine-tuned models using
SparseGPT's Hessian-based importance calculation instead of
random dropout (DARE) or magnitude-based trimming (TIES).

Usage:
    python example_sparsegpt_merge.py --method ties --density 0.2
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from dare_utils import DARE
from sparsegpt_importance import HessianCalculator, TaskVectorImportanceCalculator

# Import merging utilities
from ties_utils import TIES


def create_dummy_model(hidden_dim: int = 256, num_layers: int = 2):
    """
    Create a simple dummy model for demonstration.
    Replace this with your actual model loading.
    """

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
            return self.norm(x)

    return DummyModel()


def create_dummy_calibration_data(
    num_samples: int = 128, seq_len: int = 32, hidden_dim: int = 256
) -> List[torch.Tensor]:
    """
    Create dummy calibration data.
    Replace with actual validation data from your fine-tuning task.
    """
    calibration_data = []
    for _ in range(num_samples):
        # Random input data
        data = torch.randn(1, seq_len, hidden_dim)
        calibration_data.append(data)
    return calibration_data


def get_layer_inputs(model, layer_name, input_data):
    """
    Hook to capture inputs to a specific layer.
    """
    activations = []

    def hook(module, inp, out):
        activations.append(inp[0].detach())

    # Get the layer
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)

    # Register hook
    handle = layer.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(input_data)

    # Remove hook
    handle.remove()

    return activations[0] if activations else None


def compute_hessians_for_model(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device,
    percdamp: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Compute Hessian inverse diagonals for all linear layers in the model.

    Args:
        model: The model to compute Hessians for
        calibration_data: List of calibration input tensors
        device: Device to use
        percdamp: Damping factor for Hessian inversion

    Returns:
        Dictionary mapping layer names to inverse Hessian diagonals
    """
    hessian_inv_diags = {}

    # Find all linear layers
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_names.append(name)

    print(f"Found {len(layer_names)} linear layers: {layer_names}")

    model.eval()
    model.to(device)

    # Compute Hessian for each layer
    for layer_name in layer_names:
        print(f"\nComputing Hessian for layer: {layer_name}")

        # Get the layer
        parts = layer_name.split(".")
        layer = model
        for part in parts:
            layer = getattr(layer, part)

        # Initialize Hessian calculator
        weight_shape = layer.weight.shape
        calc = HessianCalculator(weight_shape, device=device)

        # Collect activations from calibration data
        print(f"  Processing {len(calibration_data)} calibration samples...")
        for i, data in enumerate(calibration_data):
            if (i + 1) % 32 == 0:
                print(f"    Processed {i+1}/{len(calibration_data)}")

            data = data.to(device)
            inputs = get_layer_inputs(model, layer_name, data)
            if inputs is not None:
                calc.add_batch(inputs)

        # Compute inverse Hessian diagonal
        print(f"  Computing inverse Hessian diagonal...")
        h_inv_diag = calc.get_inverse_hessian_diag(percdamp=percdamp)
        hessian_inv_diags[layer_name] = h_inv_diag

        print(f"  Done! Shape: {h_inv_diag.shape}")

    return hessian_inv_diags


def merge_models_per_layer(
    base_model: nn.Module,
    ft_models: List[nn.Module],
    method: str = "ties",
    weights: List[float] = None,
    densities: List[float] = None,
    use_sparsegpt: bool = True,
    calibration_data: List[torch.Tensor] = None,
    device: torch.device = None,
) -> nn.Module:
    """
    Merge multiple fine-tuned models layer by layer.

    Args:
        base_model: Base model before fine-tuning
        ft_models: List of fine-tuned models
        method: Merging method ('ties' or 'dare')
        weights: Weights for each task vector
        densities: Densities for trimming each task vector
        use_sparsegpt: Whether to use SparseGPT importance
        calibration_data: Calibration data for Hessian computation
        device: Device to use

    Returns:
        Merged model
    """
    if device is None:
        device = next(base_model.parameters()).device

    if weights is None:
        weights = [1.0] * len(ft_models)

    if densities is None:
        densities = [0.2] * len(ft_models)

    # Initialize merger
    if method == "ties":
        merger = TIES()
    elif method == "dare":
        merger = DARE()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute Hessians if using SparseGPT
    hessian_inv_diags_dict = {}
    if use_sparsegpt:
        if calibration_data is None:
            raise ValueError("calibration_data is required when use_sparsegpt=True")

        print("\n" + "=" * 60)
        print("COMPUTING HESSIANS FOR SPARSEGPT IMPORTANCE")
        print("=" * 60)
        hessian_inv_diags_dict = compute_hessians_for_model(
            base_model, calibration_data, device
        )
        print("\nHessian computation complete!")

    # Create merged model (copy of base model)
    import copy

    merged_model = copy.deepcopy(base_model)

    # Find all linear layers
    layer_names = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            layer_names.append(name)

    print("\n" + "=" * 60)
    print(f"MERGING {len(layer_names)} LAYERS USING {method.upper()}")
    print(f"SparseGPT Importance: {use_sparsegpt}")
    print("=" * 60)

    # Merge each layer
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n[{layer_idx+1}/{len(layer_names)}] Merging layer: {layer_name}")

        # Get layer from each model
        base_layer = dict(base_model.named_modules())[layer_name]
        ft_layers = [
            dict(ft_model.named_modules())[layer_name] for ft_model in ft_models
        ]
        merged_layer = dict(merged_model.named_modules())[layer_name]

        # Get parameters
        base_params = base_layer.weight.data
        ft_params_list = [layer.weight.data for layer in ft_layers]

        # Prepare kwargs for merging
        merge_kwargs = {
            "weights": weights,
            "base_model_parameters": base_params,
            "ft_models_parameters": ft_params_list,
            "densities": densities,
            "device": device,
        }

        # Add SparseGPT importance if enabled
        if use_sparsegpt:
            h_inv_diag = hessian_inv_diags_dict[layer_name]
            merge_kwargs["hessian_inv_diags"] = [h_inv_diag] * len(ft_models)
            merge_kwargs["use_sparsegpt"] = True

        # Merge
        merged_weight = merger.merge(**merge_kwargs)

        # Update merged model
        merged_layer.weight.data = merged_weight

        print(f"  ✓ Merged weight shape: {merged_weight.shape}")

    return merged_model


def main():
    parser = argparse.ArgumentParser(
        description="Merge models with SparseGPT importance"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ties",
        choices=["ties", "dare"],
        help="Merging method",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="Density for trimming (fraction of weights to keep)",
    )
    parser.add_argument(
        "--use_sparsegpt",
        action="store_true",
        help="Use SparseGPT importance (vs magnitude/random)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension for dummy model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in dummy model"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dummy models (replace with your actual model loading)
    print("\n" + "=" * 60)
    print("CREATING MODELS")
    print("=" * 60)

    base_model = create_dummy_model(args.hidden_dim, args.num_layers)
    print("✓ Created base model")

    # Create "fine-tuned" models by adding noise to base model
    ft_models = []
    for i in range(3):
        ft_model = create_dummy_model(args.hidden_dim, args.num_layers)
        # Copy base weights and add small perturbations
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(
                base_model.named_parameters(), ft_model.named_parameters()
            ):
                p2.data = p1.data + 0.1 * torch.randn_like(p1.data)
        ft_models.append(ft_model)
        print(f"✓ Created fine-tuned model {i+1}")

    # Create calibration data (replace with actual validation data)
    calibration_data = None
    if args.use_sparsegpt:
        print("\n" + "=" * 60)
        print("CREATING CALIBRATION DATA")
        print("=" * 60)
        calibration_data = create_dummy_calibration_data(
            num_samples=args.num_calibration_samples,
            seq_len=32,
            hidden_dim=args.hidden_dim,
        )
        print(f"✓ Created {len(calibration_data)} calibration samples")

    # Merge models
    print("\n" + "=" * 60)
    print("STARTING MERGE")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    print(f"Density: {args.density}")
    print(f"SparseGPT: {args.use_sparsegpt}")

    merged_model = merge_models_per_layer(
        base_model=base_model,
        ft_models=ft_models,
        method=args.method,
        weights=[1.0, 1.0, 1.0],
        densities=[args.density, args.density, args.density],
        use_sparsegpt=args.use_sparsegpt,
        calibration_data=calibration_data,
        device=device,
    )

    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)

    # Simple validation
    print("\nValidating merged model...")
    test_input = torch.randn(1, 32, args.hidden_dim).to(device)

    with torch.no_grad():
        base_output = base_model.to(device)(test_input)
        merged_output = merged_model.to(device)(test_input)

        # Check that merged output is different from base
        diff = torch.abs(merged_output - base_output).mean().item()
        print(f"Mean difference between base and merged output: {diff:.6f}")

        if diff > 1e-6:
            print("✓ Merge appears successful (outputs differ from base model)")
        else:
            print("⚠ Warning: Merged output very similar to base (may indicate issue)")

    # Compare methods if requested
    if args.use_sparsegpt:
        print("\n" + "=" * 60)
        print("COMPARING WITH BASELINE METHOD")
        print("=" * 60)

        baseline_model = merge_models_per_layer(
            base_model=base_model,
            ft_models=ft_models,
            method=args.method,
            weights=[1.0, 1.0, 1.0],
            densities=[args.density, args.density, args.density],
            use_sparsegpt=False,  # Use baseline
            calibration_data=None,
            device=device,
        )

        with torch.no_grad():
            baseline_output = baseline_model.to(device)(test_input)

            sparsegpt_diff = torch.abs(merged_output - base_output).mean().item()
            baseline_diff = torch.abs(baseline_output - base_output).mean().item()

            print(f"\nMean difference from base model:")
            print(f"  SparseGPT: {sparsegpt_diff:.6f}")
            print(f"  Baseline:  {baseline_diff:.6f}")

            # The differences show how much each method changes from base
            # Larger difference = more task vector information preserved


if __name__ == "__main__":
    main()
