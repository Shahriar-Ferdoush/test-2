"""
Diagnostic: Analyze Hessian Impact on Importance Scores

This script loads cached Hessians and task vectors to show:
1. Distribution of H⁻¹ values (are they varied enough?)
2. How much importance scores differ from magnitude-based
3. Whether SparseGPT actually selects different parameters
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_hessian_impact(
    cache_dir="./merge_cache",
    model_idx=0,
    layer_name="model.layers.0.self_attn.q_proj"
):
    """Analyze how Hessian changes importance scores."""
    
    print("="*80)
    print("DIAGNOSTIC: Hessian Impact Analysis")
    print("="*80)
    
    # Load Hessian
    hessian_path = Path(cache_dir) / "hessians" / f"hessian_model_{model_idx}.pt"
    if not hessian_path.exists():
        print(f"❌ Hessian not found: {hessian_path}")
        return
    
    hessians = torch.load(hessian_path)
    if layer_name not in hessians:
        print(f"❌ Layer {layer_name} not in Hessians")
        print(f"   Available layers: {list(hessians.keys())[:5]}...")
        return
    
    H = hessians[layer_name]
    H_inv_diag = torch.diag(torch.linalg.inv(H))
    
    print(f"\n[1] HESSIAN STATISTICS for {layer_name}")
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
    
    # Simulate a task vector
    print(f"\n[2] SIMULATING TASK VECTOR")
    task_vector = torch.randn_like(H_inv_diag) * 0.01  # Typical magnitude
    print(f"    Task vector range: [{task_vector.min():.6f}, {task_vector.max():.6f}]")
    
    # Compute importance both ways
    print(f"\n[3] IMPORTANCE COMPARISON")
    
    # Magnitude-based (TIES baseline)
    importance_magnitude = task_vector.abs()
    
    # Hessian-based (SparseGPT)
    eps = 1e-10
    importance_sparsegpt = task_vector.pow(2) / (H_inv_diag.pow(2) + eps)
    
    # Normalize for comparison
    importance_magnitude_norm = importance_magnitude / importance_magnitude.sum()
    importance_sparsegpt_norm = importance_sparsegpt / importance_sparsegpt.sum()
    
    # Compare top-k selections
    k = int(0.5 * len(task_vector))  # Keep 50%
    
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
    
    print(f"    Ratio range: [{importance_ratio.min():.2e}, {importance_ratio.max():.2e}]")
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
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    
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
    analyze_hessian_impact()