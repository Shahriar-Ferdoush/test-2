"""
SparseGPT-based Task Vector Pruning with Error Correction

This implements the SparseGPT algorithm for pruning task vectors in model merging.
Uses Optimal Brain Surgeon (OBS) with blockwise error propagation to maintain
model quality at high sparsity levels.

Key Innovation: Error Correction
--------------------------------
When a weight is pruned to zero, the error is propagated to remaining weights:
    error = (w_pruned - 0) / H_ii^(-1)
    W_remaining -= error @ H^(-1)[i, remaining]

This maintains model output quality even with 50-90% pruning.

Mathematical Foundation:
-----------------------
1. Hessian: H = (2/n) * X^T * X  (input activation covariance)
2. Importance: score(w_i) = w_i^2 / (H_ii^-1)^2  (considers magnitude & sensitivity)
3. Pruning: Blockwise OBS with error propagation to remaining weights

Reference: Frantar & Alistarh (2023) - "SparseGPT: Massive Language Models
           Can Be Accurately Pruned in One-Shot"
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class HessianCalculator:
    """
    Computes Hessian (input covariance matrix) for a layer using calibration data.

    The Hessian H = (2/n) * X^T * X captures correlations between input features,
    which determines weight sensitivity to changes.
    """

    def __init__(self, layer_shape: Tuple[int, int], device: torch.device = None):
        """
        Args:
            layer_shape: (out_features, in_features) for the layer
            device: Device for computation (CPU/GPU)
        """
        self.rows, self.columns = layer_shape
        self.device = device or torch.device("cpu")

        # Initialize Hessian accumulator [in_features, in_features]
        self.H = torch.zeros(
            (self.columns, self.columns), dtype=torch.float32, device=self.device
        )
        self.nsamples = 0  # Track total tokens/samples seen

    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate Hessian statistics from one batch of calibration data.

        Args:
            inp: Input activations [batch, seq_len, in_features] or [batch, in_features]
        """
        # Ensure 2D: [tokens, in_features]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])  # [batch*seq_len, in_features]

        tmp = inp.shape[0]  # Number of tokens in this batch

        # Running average update: scale old Hessian by its weight
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # Normalize and add covariance: H += (2/n) * X^T * X
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.t() @ inp  # [in_features, in_features]

        # Validate Hessian (catch numerical issues in calibration data)
        if torch.isnan(self.H).any():
            raise ValueError(
                f"NaN detected in Hessian after batch. Check calibration data."
            )
        if torch.isinf(self.H).any():
            raise ValueError(
                f"Inf detected in Hessian after batch. Check calibration data."
            )

    def get_inverse_hessian_diag(self, percdamp: float = 0.01) -> torch.Tensor:
        """
        Compute diagonal of inverse Hessian: [H^(-1)]_ii for each feature i.

        Used for importance scoring. Faster than full inverse but doesn't
        enable error correction.

        Args:
            percdamp: Dampening factor (1% of mean diagonal by default)

        Returns:
            Diagonal elements [in_features]
        """
        H = self.H.clone()

        # Handle dead features (never activated)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        # Add dampening for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag_indices = torch.arange(self.columns, device=self.device)
        H[diag_indices, diag_indices] += damp

        # Inverse via Cholesky: H = L @ L^T => H^(-1) = L^(-T) @ L^(-1)
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv_diag = torch.diag(H_inv)

            # Validate result
            if torch.isnan(H_inv_diag).any() or torch.isinf(H_inv_diag).any():
                raise ValueError("NaN/Inf in inverse Hessian diagonal")

            return H_inv_diag
        except (RuntimeError, ValueError) as e:
            # Fallback to diagonal approximation if Cholesky fails
            warnings.warn(f"Cholesky failed ({e}), using diagonal approximation")
            diag_inv = 1.0 / (torch.diag(H) + 1e-10)
            return diag_inv

    def get_inverse_hessian(self, percdamp: float = 0.01) -> torch.Tensor:
        """
        Compute FULL inverse Hessian for error correction.

        Returns upper triangular Cholesky factor of H^(-1).
        This enables efficient error propagation: H^(-1) @ error

        Args:
            percdamp: Dampening factor (1% of mean diagonal by default)

        Returns:
            Upper triangular matrix [in_features, in_features]
            Represents Cholesky factor U where H^(-1) = U^T @ U
        """
        H = self.H.clone()

        # Handle dead features
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        # Add dampening
        damp = percdamp * torch.mean(torch.diag(H))
        diag_indices = torch.arange(self.columns, device=self.device)
        H[diag_indices, diag_indices] += damp

        # Cholesky decomposition: H = L @ L^T
        try:
            L = torch.linalg.cholesky(H)
            # Inverse: H^(-1) = L^(-T) @ L^(-1)
            H_inv = torch.cholesky_inverse(L)

            # Validate H_inv before computing its Cholesky
            if torch.isnan(H_inv).any() or torch.isinf(H_inv).any():
                raise ValueError(
                    f"NaN/Inf in H_inv: NaN={torch.isnan(H_inv).sum()}, "
                    f"Inf={torch.isinf(H_inv).sum()}"
                )

            # Return Cholesky factor of inverse (upper triangular for efficiency)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

            # Final validation
            if torch.isnan(H_inv_chol).any() or torch.isinf(H_inv_chol).any():
                raise ValueError("NaN/Inf in Cholesky factor of H_inv")

            return H_inv_chol
        except (RuntimeError, ValueError) as e:
            raise ValueError(
                f"Failed to compute inverse Hessian: {e}. "
                f"Hessian condition: min_diag={torch.diag(H).min():.2e}, "
                f"max_diag={torch.diag(H).max():.2e}, mean={torch.diag(H).mean():.2e}"
            )


def compute_importance_scores(
    weights: torch.Tensor, hessian_inv_diag: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute SparseGPT importance scores for weights.

    Formula: importance(w_ij) = w_ij^2 / (H_jj^-1)^2

    Higher importance = more critical to preserve
    - Higher |w_ij| = larger magnitude
    - Lower H_jj^-1 = lower sensitivity (harder to prune)

    Args:
        weights: Weight tensor [out_features, in_features]
        hessian_inv_diag: Diagonal of H^(-1), shape [in_features]
        eps: Numerical stability constant

    Returns:
        Importance scores with same shape as weights
    """
    original_shape = weights.shape

    # Handle 1D weights (bias vectors)
    if len(original_shape) == 1:
        importance = weights**2 / ((hessian_inv_diag + eps) ** 2)
        return importance

    # Handle 2D weights (linear layers)
    elif len(original_shape) == 2:
        # Broadcast H_inv_diag across output dimension
        # Each input feature j has one H_jj^-1 value applied to all output neurons
        importance = weights**2 / ((hessian_inv_diag.unsqueeze(0) + eps) ** 2)
        return importance

    # Handle higher-dimensional weights (conv layers)
    else:
        # Flatten spatial dimensions, compute importance, reshape back
        flat_weights = weights.flatten(1)  # [out, in*H*W]
        # Repeat H_inv_diag for spatial dimensions
        in_features = hessian_inv_diag.shape[0]
        spatial_size = flat_weights.shape[1] // in_features
        expanded_hessian = hessian_inv_diag.repeat_interleave(spatial_size)

        importance = flat_weights**2 / ((expanded_hessian.unsqueeze(0) + eps) ** 2)
        importance = importance.reshape(original_shape)
        return importance


def generate_importance_mask(
    importance_scores: torch.Tensor, density: float
) -> torch.Tensor:
    """
    Generate binary mask keeping top-k% most important weights.

    Args:
        importance_scores: Importance scores (same shape as weights)
        density: Fraction of weights to keep (0.2 = keep 20%, prune 80%)

    Returns:
        Binary mask (1 = keep, 0 = prune) with same shape as importance_scores
    """
    if density >= 1.0:
        return torch.ones_like(importance_scores)
    if density <= 0.0:
        return torch.zeros_like(importance_scores)

    # Flatten for top-k selection
    flat_importance = importance_scores.flatten()
    num_params = flat_importance.numel()
    num_keep = int(num_params * density)
    num_keep = max(1, num_keep)  # Keep at least 1 weight

    # Get threshold value at top-k position
    threshold = torch.topk(flat_importance, num_keep, sorted=False)[0].min()

    # Create mask (>= threshold to handle ties)
    mask = (importance_scores >= threshold).float()

    return mask


def apply_importance_mask(
    task_vector: torch.Tensor,
    mask: torch.Tensor,
    rescale: bool = True,
    density: float = None,
) -> torch.Tensor:
    """
    Apply binary mask to task vector with optional rescaling.

    Args:
        task_vector: Task vector to prune
        mask: Binary mask (1 = keep, 0 = prune)
        rescale: If True, scale remaining weights by 1/density (DARE-style)
        density: Required if rescale=True

    Returns:
        Masked task vector
    """
    masked = task_vector * mask

    if rescale:
        if density is None:
            actual_density = mask.float().mean().item()
        else:
            actual_density = density

        if actual_density > 0:
            masked = masked / actual_density  # Scale to preserve magnitude

    return masked


def prune_task_vector_with_error_correction(
    task_vector: torch.Tensor,
    hessian_inv: torch.Tensor,
    density: float,
    blocksize: int = 128,
    rescale: bool = False,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """
    Prune task vector using SparseGPT with blockwise OBS error correction.

    Algorithm (per block):
    1. Compute importance scores for block columns
    2. For each column i in block:
        a. Prune low-importance weights to zero
        b. Compute reconstruction error: err = (w - w_pruned) / H_ii^(-1)
        c. Propagate error to remaining columns: W[:, i+1:] -= err @ H^(-1)[i, i+1:]
    3. After block: propagate accumulated errors to future blocks

    This maintains model output quality by compensating remaining weights
    for errors introduced by pruning.

    Args:
        task_vector: Weight tensor [out_features, in_features]
        hessian_inv: Upper triangular Cholesky factor of H^(-1) [in_features, in_features]
        density: Fraction of weights to keep (0.2 = 20%)
        blocksize: Columns per block (trade-off: larger = faster but more memory)
        rescale: If True, scale by 1/density after pruning (DARE-style)
        percdamp: Dampening factor (should match HessianCalculator.get_inverse_hessian)

    Returns:
        Pruned task vector with error correction applied
    """
    # Validate inputs
    if len(task_vector.shape) != 2:
        raise ValueError(f"Expected 2D task vector, got shape {task_vector.shape}")

    rows, columns = task_vector.shape
    if hessian_inv.shape != (columns, columns):
        raise ValueError(
            f"Hessian shape {hessian_inv.shape} doesn't match task vector columns {columns}"
        )

    device = task_vector.device
    dtype = task_vector.dtype

    # Work in float32 for numerical precision
    W = task_vector.clone().float()
    Hinv = hessian_inv.float()

    # Extract diagonal of H^(-1) for importance scoring
    # Hinv is upper triangular Cholesky factor: H^(-1) = Hinv^T @ Hinv
    # For upper triangular Cholesky: diag(H^(-1)) = diag(Hinv)^2
    # But we need the actual H^(-1) diagonal for proper scaling
    # Compute full H^(-1) diagonal from Cholesky factor
    Hinv_full = Hinv.t() @ Hinv  # Reconstruct H^(-1) from Cholesky factor
    Hinv_diag = torch.diag(Hinv_full)  # [columns]

    # Compute importance scores for entire weight matrix
    importance = compute_importance_scores(W, Hinv_diag, eps=1e-10)

    # Generate pruning mask (1 = keep, 0 = prune)
    mask = generate_importance_mask(importance, density)

    # Validate Hessian diagonal (catch numerical issues early)
    if torch.isnan(Hinv_diag).any() or torch.isinf(Hinv_diag).any():
        raise ValueError(
            f"Invalid Hessian diagonal: NaN={torch.isnan(Hinv_diag).sum()}, "
            f"Inf={torch.isinf(Hinv_diag).sum()}"
        )

    # === Blockwise pruning with error correction ===
    # Following original SparseGPT algorithm exactly
    Losses = torch.zeros(rows, device=device)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        # Extract block of weights [rows, blocksize]
        W1 = W[:, i1:i2].clone()

        # Pruned weights (Q1), errors (Err1) for this block
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)

        # Extract inverse Hessian block
        Hinv1 = Hinv[i1:i2, i1:i2]  # [blocksize, blocksize]

        # Extract mask block
        mask1 = mask[:, i1:i2]  # [rows, blocksize]

        # === Process each column in block ===
        for i in range(count):
            w = W1[:, i]  # Current column [rows]
            d = torch.diag(Hinv1)[i]  # Diagonal element of Hinv (Cholesky factor)

            # CRITICAL FIX: For proper scaling, use the actual H^(-1) diagonal
            # But for error propagation structure, use Cholesky diagonal
            # Reconstruct the proper inverse diagonal element
            d_inv = Hinv_diag[i1 + i]  # Actual H^(-1)_ii

            # Apply mask to get pruned weights
            q = w.clone()
            q[~mask1[:, i]] = 0  # Zero out pruned weights (where mask=False)

            # Store pruned weights
            Q1[:, i] = q

            # Track reconstruction error
            Losses1[:, i] = (w - q) ** 2 / (d_inv**2 + 1e-10)

            # === ERROR CORRECTION (key innovation!) ===
            # Compute scaled error using CHOLESKY diagonal (not H^(-1) diagonal)
            # This matches the original algorithm's structure
            err1 = (w - q) / (d + 1e-10)  # Scale by Cholesky factor diagonal

            # CRITICAL: Propagate to CURRENT and remaining columns (not just i+1)
            # This is the key difference from the buggy version
            # Update W1[:, i:] (from current column onwards)
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

            # Store error for global propagation
            Err1[:, i] = err1

        # Write pruned block back
        W[:, i1:i2] = Q1

        # Accumulate losses
        Losses += torch.sum(Losses1, 1) / 2

        # === GLOBAL error propagation to future blocks ===
        # Propagate accumulated errors from this block to all future blocks
        if i2 < columns:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # Validate no NaN/Inf after each block
        if torch.isnan(W).any() or torch.isinf(W).any():
            raise ValueError(
                f"NaN/Inf detected in weights after block {i1}-{i2}. "
                f"NaN count: {torch.isnan(W).sum()}, Inf count: {torch.isinf(W).sum()}"
            )

    # Optional DARE-style rescaling
    if rescale and density > 0:
        W = W / density

    # Final validation before returning
    if torch.isnan(W).any():
        raise ValueError(
            f"NaN in final pruned task vector. NaN count: {torch.isnan(W).sum()}"
        )
    if torch.isinf(W).any():
        raise ValueError(
            f"Inf in final pruned task vector. Inf count: {torch.isinf(W).sum()}"
        )

    # Convert back to original dtype
    return W.to(dtype)


class TaskVectorImportanceCalculator:
    """
    High-level interface for computing Hessians and pruning task vectors.

    Usage:
        1. Initialize with base model and calibration data
        2. Compute Hessians for layers: compute_hessians_for_layers()
        3. Prune task vectors: prune_task_vector_with_error_correction()
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: List[Tuple[torch.Tensor, ...]],
        device: torch.device = None,
        percdamp: float = 0.01,
    ):
        """
        Args:
            model: Base model (for registering hooks)
            calibration_loader: List of input batches for Hessian computation
            device: Device for computation
            percdamp: Dampening factor for Hessian inversion
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device or next(model.parameters()).device
        self.percdamp = percdamp

        # Storage for computed Hessians
        self.hessian_calculators: Dict[str, HessianCalculator] = {}
        self.hessian_inv_diags: Dict[str, torch.Tensor] = {}
        self.hessian_invs: Dict[str, torch.Tensor] = {}

    def compute_hessians_for_layers(
        self,
        layer_names: List[str],
        verbose: bool = True,
    ):
        """
        Compute Hessians for specified layers using calibration data.

        Args:
            layer_names: List of layer names (e.g., ['model.layers.0.self_attn.q_proj'])
            verbose: Print progress
        """
        # Register forward hooks to capture inputs
        hooks = []
        inputs_captured = {name: [] for name in layer_names}

        def make_hook(name):
            def hook(module, inp, out):
                # Capture input activations
                inputs_captured[name].append(inp[0].detach().cpu())

            return hook

        # Attach hooks
        for name in layer_names:
            layer = self._get_layer_by_name(name)
            if layer is None:
                warnings.warn(f"Layer {name} not found")
                continue
            hook = layer.register_forward_hook(make_hook(name))
            hooks.append(hook)

        # Run calibration data through model
        if verbose:
            print(f"Computing Hessians for {len(layer_names)} layers...")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.calibration_loader):
                if verbose and batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.calibration_loader)}")

                # Forward pass (hooks will capture inputs)
                if isinstance(batch, (tuple, list)):
                    inp = batch[0].to(self.device)
                else:
                    inp = batch.to(self.device)

                self.model(inp)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute Hessians from captured inputs
        for name in layer_names:
            if name not in inputs_captured or len(inputs_captured[name]) == 0:
                warnings.warn(f"No inputs captured for layer {name}")
                continue

            if verbose:
                print(f"Processing layer: {name}")

            # Get layer shape
            layer = self._get_layer_by_name(name)
            if hasattr(layer, "weight"):
                layer_shape = layer.weight.shape
            else:
                warnings.warn(f"Layer {name} has no weight attribute")
                continue

            # Initialize Hessian calculator
            calc = HessianCalculator(layer_shape, device=self.device)

            # Add all captured batches
            for inp in inputs_captured[name]:
                calc.add_batch(inp.to(self.device))

            # Store calculator and compute inverses
            self.hessian_calculators[name] = calc
            self.hessian_inv_diags[name] = calc.get_inverse_hessian_diag(self.percdamp)
            self.hessian_invs[name] = calc.get_inverse_hessian(self.percdamp)

        if verbose:
            print("Hessian computation complete!")

    def prune_task_vector_with_error_correction(
        self,
        layer_name: str,
        task_vector: torch.Tensor,
        density: float,
        blocksize: int = 128,
        rescale: bool = False,
    ) -> torch.Tensor:
        """
        Prune task vector for a specific layer using precomputed Hessian.

        Args:
            layer_name: Name of the layer
            task_vector: Task vector to prune [out_features, in_features]
            density: Fraction of weights to keep
            blocksize: Columns per block
            rescale: DARE-style rescaling

        Returns:
            Pruned task vector with error correction
        """
        if layer_name not in self.hessian_invs:
            raise ValueError(f"Hessian not computed for layer {layer_name}")

        hessian_inv = self.hessian_invs[layer_name]

        return prune_task_vector_with_error_correction(
            task_vector=task_vector,
            hessian_inv=hessian_inv,
            density=density,
            blocksize=blocksize,
            rescale=rescale,
            percdamp=self.percdamp,
        )

    def get_importance_scores(
        self,
        layer_name: str,
        task_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Get importance scores for task vector."""
        if layer_name not in self.hessian_inv_diags:
            raise ValueError(f"Hessian not computed for layer {layer_name}")

        return compute_importance_scores(
            task_vector, self.hessian_inv_diags[layer_name]
        )

    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """Get layer by dotted name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module


class TaskVectorImportanceCalculatorPerTask:
    """
    Compute per-task Hessians using fine-tuned models instead of base model.

    This captures task-specific feature importance by computing Hessians
    from each fine-tuned model's calibration data.

    Usage:
        1. Initialize with fine-tuned models and their calibration data
        2. Compute Hessians: compute_hessians_for_all_tasks()
        3. Get per-task importance scores or masks
    """

    def __init__(
        self,
        ft_models: List[nn.Module],
        calibration_loaders: List[List[Tuple[torch.Tensor, ...]]],
        device: torch.device = None,
        percdamp: float = 0.01,
    ):
        """
        Args:
            ft_models: List of fine-tuned models
            calibration_loaders: List of calibration data loaders (one per model)
            device: Device for computation
            percdamp: Dampening factor
        """
        self.ft_models = ft_models
        self.calibration_loaders = calibration_loaders
        self.device = device or next(ft_models[0].parameters()).device
        self.percdamp = percdamp
        self.num_tasks = len(ft_models)

        # Storage: task_idx -> layer_name -> Hessian data
        self.task_hessians: Dict[int, Dict[str, HessianCalculator]] = {}
        self.task_hessian_inv_diags: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_hessian_invs: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_hessians_for_all_tasks(
        self,
        layer_names: List[str],
        verbose: bool = True,
    ):
        """
        Compute Hessians for all tasks and specified layers.

        Args:
            layer_names: List of layer names
            verbose: Print progress
        """
        for task_idx in range(self.num_tasks):
            if verbose:
                print(f"\n=== Task {task_idx + 1}/{self.num_tasks} ===")

            model = self.ft_models[task_idx]
            calib_data = self.calibration_loaders[task_idx]

            # Use TaskVectorImportanceCalculator for this task
            calc = TaskVectorImportanceCalculator(
                model=model,
                calibration_loader=calib_data,
                device=self.device,
                percdamp=self.percdamp,
            )

            calc.compute_hessians_for_layers(layer_names, verbose=verbose)

            # Store results
            self.task_hessians[task_idx] = calc.hessian_calculators
            self.task_hessian_inv_diags[task_idx] = calc.hessian_inv_diags
            self.task_hessian_invs[task_idx] = calc.hessian_invs

    def prune_task_vectors(
        self,
        layer_name: str,
        task_vectors: List[torch.Tensor],
        densities: List[float],
        blocksize: int = 128,
        rescale: bool = False,
    ) -> List[torch.Tensor]:
        """
        Prune task vectors using their corresponding task-specific Hessians.

        Args:
            layer_name: Layer name
            task_vectors: List of task vectors (one per task)
            densities: List of densities (one per task)
            blocksize: Columns per block
            rescale: DARE-style rescaling

        Returns:
            List of pruned task vectors
        """
        pruned = []

        for task_idx, (tv, density) in enumerate(zip(task_vectors, densities)):
            if layer_name not in self.task_hessian_invs[task_idx]:
                raise ValueError(
                    f"Hessian not computed for task {task_idx}, layer {layer_name}"
                )

            hessian_inv = self.task_hessian_invs[task_idx][layer_name]

            pruned_tv = prune_task_vector_with_error_correction(
                task_vector=tv,
                hessian_inv=hessian_inv,
                density=density,
                blocksize=blocksize,
                rescale=rescale,
                percdamp=self.percdamp,
            )

            pruned.append(pruned_tv)

        return pruned
