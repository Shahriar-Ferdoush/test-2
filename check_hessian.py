"""
Diagnostic: Analyze Hessian Impact on Importance Scores (Memory-Efficient)

This script computes Hessians on-the-fly (no caching) to show:
1. Distribution of H⁻¹ values (are they varied enough?)
2. How much importance scores differ from magnitude-based
3. Whether SparseGPT actually selects different parameters

Note: Hessians are NOT cached (saves ~12GB). We compute them in memory only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_hessian_for_layer(model, tokenizer, layer_name, num_samples=128):
    """Compute Hessian for a specific layer using calibration data."""
    print(f"📊 Computing Hessian for {layer_name} (in memory)...")

    # Load calibration data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Get layer module
    parts = layer_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)

    # Hook to capture inputs
    inputs_list = []

    def hook(module, input, output):
        inputs_list.append(input[0].detach().cpu())

    handle = module.register_forward_hook(hook)

    # Collect inputs
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample["text"]
            if len(text.strip()) == 0:
                continue
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            model(**tokens.to(model.device))

    handle.remove()

    # Compute Hessian H = E[x·x^T]
    all_inputs = torch.cat(inputs_list, dim=0)  # [N, seq_len, hidden_dim]
    all_inputs = all_inputs.reshape(-1, all_inputs.size(-1))  # [N*seq_len, hidden_dim]

    H = torch.matmul(all_inputs.T, all_inputs) / all_inputs.size(0)
    H = H.to(torch.float32)

    # Add damping for numerical stability
    damping = 1e-5
    H = H + damping * torch.eye(H.size(0))

    print(f"   ✓ Hessian computed: {H.shape}")
    return H


def analyze_hessian_impact(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    ft_model_paths=None,
    cache_dir="./merge_cache",
    model_idx=0,
    layer_name="model.layers.0.self_attn.q_proj",
):
    """Analyze how Hessian changes importance scores."""

    print("=" * 80)
    print("DIAGNOSTIC: Hessian Impact Analysis (Memory-Efficient)")
    print("=" * 80)
    print(f"⚠️  NOTE: Computing Hessian on-the-fly (not loading cache)")
    print(f"   This saves ~12GB storage but takes a few minutes...")
    print()

    if ft_model_paths is None:
        ft_model_paths = [
            "./fine_tuned_models/doctor_model",
            "./fine_tuned_models/mental_health_model",
        ]

    # Load base model
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned model
    ft_model_path = ft_model_paths[model_idx]
    print(f"Loading fine-tuned model: {ft_model_path}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path, torch_dtype=torch.float16, device_map="cpu"
    )

    # Compute Hessian
    H = compute_hessian_for_layer(base_model, tokenizer, layer_name, num_samples=128)
    H_inv_diag = torch.diag(torch.linalg.inv(H))

    # Get task vector from models
    print(f"\n[1] COMPUTING TASK VECTOR")
    base_weight = None
    ft_weight = None

    parts = layer_name.split(".")
    module = base_model
    for part in parts:
        module = getattr(module, part)
    base_weight = module.weight.data.clone().to(torch.float32)

    module = ft_model
    for part in parts:
        module = getattr(module, part)
    ft_weight = module.weight.data.clone().to(torch.float32)

    task_vector = ft_weight - base_weight
    print(f"    Task vector shape: {task_vector.shape}")
    print(f"    Task vector range: [{task_vector.min():.6f}, {task_vector.max():.6f}]")

    # Free models
    del base_model, ft_model
    torch.cuda.empty_cache()

    print(f"\n[2] HESSIAN STATISTICS for {layer_name}")
    print(f"    Shape: {H.shape}")
    print(f"    H⁻¹ diagonal range: [{H_inv_diag.min():.6f}, {H_inv_diag.max():.6f}]")
    print(f"    H⁻¹ diagonal mean: {H_inv_diag.mean():.6f}")
    print(f"    H⁻¹ diagonal std: {H_inv_diag.std():.6f}")

    # Check if Hessian is too uniform (bad calibration)
    variance_ratio = H_inv_diag.std() / H_inv_diag.mean()
    print(f"    Coefficient of variation: {variance_ratio:.4f}")

    if variance_ratio < 0.1:
        print(f"    ⚠️  WARNING: Hessian is very uniform!")
        print(f"        This suggests poor calibration data.")
        print(f"        Expected: 0.3-1.0, Got: {variance_ratio:.4f}")
    else:
        print(f"    ✓ Hessian shows good variance (importance will differ)")

    print(f"\n[3] IMPORTANCE COMPARISON")

    # For layers with 2D weights (Linear layers), compute importance per-element
    if len(task_vector.shape) == 2:
        # Expand H_inv_diag to match weight shape [out_dim, in_dim]
        # H is computed on inputs, so H_inv_diag corresponds to input dimension
        H_inv_diag_expanded = H_inv_diag.unsqueeze(0).expand_as(task_vector)
        task_vector_flat = task_vector.flatten()
    else:
        # For 1D or other shapes, flatten
        H_inv_diag_expanded = H_inv_diag
        task_vector_flat = task_vector.flatten()

    # Compute importance both ways
    # Magnitude-based (TIES baseline)
    importance_magnitude = task_vector_flat.abs()

    # Hessian-based (SparseGPT)
    eps = 1e-10
    if len(task_vector.shape) == 2:
        importance_sparsegpt = task_vector.pow(2) / (H_inv_diag_expanded.pow(2) + eps)
        importance_sparsegpt = importance_sparsegpt.flatten()
    else:
        importance_sparsegpt = task_vector_flat.pow(2) / (H_inv_diag.pow(2) + eps)

    # Normalize for comparison
    importance_magnitude_norm = importance_magnitude / importance_magnitude.sum()
    importance_sparsegpt_norm = importance_sparsegpt / importance_sparsegpt.sum()

    # Compare top-k selections
    k = int(0.5 * len(task_vector_flat))  # Keep 50%

    top_k_magnitude = torch.topk(importance_magnitude, k).indices
    top_k_sparsegpt = torch.topk(importance_sparsegpt, k).indices

    overlap = len(set(top_k_magnitude.tolist()) & set(top_k_sparsegpt.tolist()))
    overlap_pct = 100 * overlap / k

    print(f"    Top-50% overlap: {overlap_pct:.1f}%")

    if overlap_pct > 90:
        print(f"    ❌ HIGH OVERLAP: SparseGPT selects almost same parameters!")
        print(f"       This suggests Hessian is not providing useful information.")
        print(f"       Possible causes:")
        print(f"         1. Calibration data is too uniform (check dataset)")
        print(f"         2. Only 1 calibration sample (increase to 128+)")
        print(f"         3. Hessian computed on wrong distribution")
    elif overlap_pct < 50:
        print(f"    ✓ LOW OVERLAP: SparseGPT selects DIFFERENT parameters!")
        print(f"      This is expected—Hessian is working correctly.")
    else:
        print(f"    ~ MODERATE OVERLAP: SparseGPT makes some difference.")

    # Compute importance ratio distribution
    print(f"\n[4] IMPORTANCE RATIO DISTRIBUTION")
    importance_ratio = importance_sparsegpt / (importance_magnitude + eps)

    print(
        f"    Ratio range: [{importance_ratio.min():.2e}, {importance_ratio.max():.2e}]"
    )
    print(f"    Ratio median: {importance_ratio.median():.2e}")

    # Count how many parameters change by more than 10×
    changed_significantly = (importance_ratio < 0.1) | (importance_ratio > 10.0)
    pct_changed = 100 * changed_significantly.float().mean()
    print(f"    Parameters with >10× importance change: {pct_changed:.1f}%")

    if pct_changed < 10:
        print(f"    ❌ TOO FEW parameters significantly affected!")
        print(f"       Hessian is not providing enough differentiation.")
    else:
        print(f"    ✓ Good: {pct_changed:.1f}% of parameters significantly affected")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if variance_ratio < 0.1 and overlap_pct > 90:
        print("❌ PROBLEM DETECTED:")
        print("   Hessian is too uniform → SparseGPT ≈ TIES-Magnitude")
        print("\nSOLUTION:")
        print("   1. Increase calibration samples (use 128-256)")
        print("   2. Use calibration data from target domain")
        print("   3. Check if Hessian computation is correct")
    elif variance_ratio >= 0.1 and overlap_pct < 70:
        print("✓ HESSIAN IS WORKING:")
        print("   SparseGPT selects different parameters than magnitude!")
        print("   Lower parameter change % might be expected behavior.")
    else:
        print("~ INCONCLUSIVE:")
        print("   Run with actual task vectors to see real behavior.")


if __name__ == "__main__":
    # Configure paths for your setup
    analyze_hessian_impact(
        base_model_path="meta-llama/Llama-3.2-1B-Instruct",
        ft_model_paths=[
            "./fine_tuned_models/doctor_model",
            "./fine_tuned_models/mental_health_model",
        ],
        model_idx=0,  # Which fine-tuned model to analyze
        layer_name="model.layers.0.self_attn.q_proj",  # Which layer to analyze
    )
