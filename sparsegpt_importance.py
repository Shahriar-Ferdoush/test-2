"""
SparseGPT-based Importance Calculation for Task Vector Pruning

This module implements the SparseGPT importance calculation algorithm
for pruning task vectors in model merging. Instead of using random dropout (DARE)
or simple magnitude-based trimming (TIES), this approach uses Hessian-based
importance scores that consider weight sensitivity to changes.

Mathematical Foundation:
-----------------------
1. Hessian Computation:
   H = (2/n) * X^T * X
   where X are the input activations from calibration data

2. Weight Importance Score:
   importance(w_i) = w_i^2 / (H_ii^-1)^2

   This measures the reconstruction error if weight w_i is removed.
   Higher score = more important weight.

3. Mask Generation:
   - Sort weights by importance score
   - Keep top (density * 100)% most important weights
   - Zero out the rest

Key Advantages over Naive Methods:
-----------------------------------
- TIES uses |w| which ignores weight sensitivity
- DARE uses random dropout which ignores all structure
- SparseGPT uses w^2/H_ii^-1 which captures:
  * Weight magnitude (numerator)
  * Sensitivity to changes (denominator via Hessian)
  * Second-order interaction effects

Usage:
------
1. Collect calibration data from validation set of fine-tuning task
2. Compute Hessian for each layer
3. Calculate importance scores for task vectors
4. Apply importance-based masking instead of random/top-k
"""

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class HessianCalculator:
    """
    Computes the Hessian matrix for a layer using calibration data.

    The Hessian H approximates the second derivative of the loss with respect
    to the layer's input activations. This captures weight importance information.
    """

    def __init__(self, layer_shape: Tuple[int, int], device: torch.device = None):
        """
        Initialize Hessian calculator.

        Args:
            layer_shape: Shape of the weight matrix (rows, columns)
            device: Device to store Hessian on
        """
        self.rows, self.columns = layer_shape
        self.device = device if device is not None else torch.device("cpu")
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Add a batch of input activations to accumulate Hessian.

        Mathematical Background:
        -----------------------
        We compute H = (1/n) * Σ(x_i * x_i^T) where x_i are input activations.
        This approximates the Fisher Information Matrix, which captures the
        sensitivity of the layer's output to weight changes.

        Why This Works:
        --------------
        - High H[j,j] → input feature j has high variance → important feature
        - High H[j,k] → features j and k are correlated
        - H^{-1}[j,j] → how much error propagates when weight w_*j changes

        Args:
            inp: Input activations of shape (batch_size, seq_len, hidden_dim)
                 or (batch_size, hidden_dim)
                 Example: [1, 2048, 4096] for LLaMA layer
        """
        # STEP 1: Handle different input shapes
        # Ensure we have at least 3D tensor: [batch, seq, features]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)  # [seq, features] → [1, seq, features]

        tmp = inp.shape[0]  # Batch size for averaging

        # STEP 2: Flatten batch and sequence dimensions
        # [batch, seq, features] → [batch*seq, features]
        # This treats each token as an independent sample
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        # Example: [1, 2048, 4096] → [2048, 4096]

        # STEP 3: Transpose for efficient outer product computation
        # [num_samples, features] → [features, num_samples]
        inp = inp.t()
        # Example: [2048, 4096] → [4096, 2048]

        # STEP 4: Update running average (incremental mean)
        # This allows processing data in batches without storing everything
        # Formula: H_new = (n/(n+m)) * H_old + (m/(n+m)) * X_new @ X_new^T
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # STEP 5: Scale inputs for numerical stability
        # The sqrt(2/n) factor prevents overflow/underflow as n grows
        # This is a variance stabilization technique
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        # STEP 6: Accumulate Hessian: H += X @ X^T
        # Result shape: [features, features]
        # H[i,j] = correlation between feature i and feature j
        # Example: [4096, 2048] @ [2048, 4096] = [4096, 4096]
        self.H += inp.matmul(inp.t())

    def get_hessian(self) -> torch.Tensor:
        """
        Get the computed Hessian matrix.

        Returns:
            Hessian matrix of shape (columns, columns)
        """
        return self.H

    def get_inverse_hessian_diag(self, percdamp: float = 0.01) -> torch.Tensor:
        """
        Compute diagonal of inverse Hessian using Cholesky decomposition.

        Mathematical Background:
        -----------------------
        H^{-1}[j,j] measures how much the loss changes when we perturb
        weights connected to input feature j.

        - LOW H^{-1}[j,j] → small sensitivity → safe to prune weights from feature j
        - HIGH H^{-1}[j,j] → high sensitivity → keep weights from feature j

        Why Cholesky?
        -------------
        - H is positive semi-definite (by construction: H = X^T X)
        - Cholesky decomposition: H = L L^T (L is lower triangular)
        - More numerically stable than direct inversion
        - Faster than general matrix inversion

        Args:
            percdamp: Damping factor (default: 0.01)
                     Adds percdamp * mean(diag(H)) to diagonal
                     Prevents singular matrix / numerical instability

        Returns:
            Diagonal of H^-1 of shape (columns,)
            Each element H^{-1}[j,j] corresponds to input feature j
        """
        H = self.H.clone()

        # STEP 1: Handle dead neurons (zero activations)
        # If a feature never activates, H[j,j] = 0
        # Set to 1 to avoid division by zero (will zero out these weights anyway)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        # STEP 2: Add damping for numerical stability
        # Damping formula: H → H + λI where λ = percdamp * mean(diag(H))
        # This ensures H is positive definite (invertible)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp

        # STEP 3: Compute inverse via Cholesky decomposition
        # H = L @ L^T (Cholesky factorization)
        # H^{-1} = (L^T)^{-1} @ L^{-1}
        try:
            L = torch.linalg.cholesky(H)  # Compute L
            H_inv = torch.cholesky_inverse(L)  # Compute H^{-1}
            H_inv = torch.linalg.cholesky(H_inv, upper=True)  # Stabilize
        except RuntimeError as e:
            # Fallback: Use pseudo-inverse if Cholesky fails
            # This can happen with highly correlated features
            warnings.warn(f"Cholesky decomposition failed: {e}. Using pseudo-inverse.")
            H_inv = torch.linalg.pinv(H)

        # STEP 4: Extract diagonal elements
        # These are the values we need for importance scoring
        inv_diag = torch.diag(H_inv)
        # Shape: [in_features]
        # inv_diag[j] = H^{-1}[j,j] = sensitivity of feature j

        return inv_diag


def compute_importance_scores(
    weights: torch.Tensor, hessian_inv_diag: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute SparseGPT importance scores for weights.

    Formula: importance(w_ij) = w_ij^2 / (H_jj^-1)^2

    This formula comes from the SparseGPT paper (Frantar & Alistarh, 2023).
    For a weight matrix W with shape [out_features, in_features]:
    - W[i,j] connects input feature j to output neuron i
    - H_jj^-1 is the Hessian inverse diagonal for INPUT feature j
    - The same H_jj^-1 applies to ALL connections from feature j (all output neurons)

    Higher importance = weight is more critical to preserve.
    Higher w_ij = larger weight magnitude → more important
    Higher H_jj^-1 = higher sensitivity to changes → easier to remove

    Args:
        weights: Weight tensor, typically shape [out_features, in_features]
                 For linear layer: [output_dim, input_dim]
        hessian_inv_diag: Diagonal of inverse Hessian, shape [in_features]
                          One value per INPUT feature (column of weight matrix)
        eps: Small constant for numerical stability (default: 1e-10)

    Returns:
        Importance scores with same shape as weights

    Example:
        For weight matrix [4096, 11008] and H_inv_diag [11008]:
        - W[0, 100] has importance: W[0,100]^2 / H_inv[100]^2
        - W[1, 100] has importance: W[1,100]^2 / H_inv[100]^2
        - Both use the SAME H_inv[100] (feature 100's sensitivity)
    """
    # Validate input dimensions
    original_shape = weights.shape

    # Handle different weight tensor shapes
    if len(original_shape) == 1:
        # 1D tensor (e.g., bias or vector)
        if hessian_inv_diag.numel() != weights.numel():
            raise ValueError(
                f"For 1D weights, hessian_inv_diag must have same size. "
                f"Got weights: {weights.shape}, hessian_inv_diag: {hessian_inv_diag.shape}"
            )
        # Direct element-wise computation
        importance = (weights**2) / ((hessian_inv_diag + eps) ** 2)

    elif len(original_shape) == 2:
        # 2D tensor (weight matrix): [out_features, in_features]
        out_features, in_features = original_shape

        if hessian_inv_diag.numel() != in_features:
            raise ValueError(
                f"For 2D weights [out, in], hessian_inv_diag must have size [in]. "
                f"Got weights: {original_shape}, hessian_inv_diag: {hessian_inv_diag.shape}"
            )

        # Broadcast H_inv_diag across output dimension
        # Shape: [1, in_features] → broadcasts to [out_features, in_features]
        hessian_inv_diag_broadcasted = hessian_inv_diag.reshape(1, -1)

        # Compute importance: w_ij^2 / (H_jj^-1)^2
        # This matches SparseGPT's algorithm where each INPUT feature j
        # has the same H_jj^-1 for all OUTPUT neurons i
        importance = (weights**2) / ((hessian_inv_diag_broadcasted + eps) ** 2)

    else:
        # Higher dimensional tensors (rare, but handle gracefully)
        # Assume last dimension corresponds to input features
        if hessian_inv_diag.numel() != original_shape[-1]:
            raise ValueError(
                f"For {len(original_shape)}D weights, hessian_inv_diag must match last dim. "
                f"Got weights: {original_shape}, hessian_inv_diag: {hessian_inv_diag.shape}"
            )

        # Reshape for broadcasting: [..., 1, in_features]
        broadcast_shape = [1] * (len(original_shape) - 1) + [original_shape[-1]]
        hessian_inv_diag_broadcasted = hessian_inv_diag.reshape(broadcast_shape)

        importance = (weights**2) / ((hessian_inv_diag_broadcasted + eps) ** 2)

    return importance


def generate_importance_mask(
    importance_scores: torch.Tensor, density: float
) -> torch.Tensor:
    """
    Generate binary mask based on importance scores.

    Algorithm:
    ---------
    1. Compute top-k where k = density * num_weights
    2. Find threshold score at position k
    3. Keep all weights with score >= threshold
    4. Drop all weights with score < threshold

    Keeps top (density * 100)% most important weights.

    Example:
    -------
    scores = [10, 5, 3, 8, 1, 9]
    density = 0.5 (keep 50%)
    k = 3 weights
    threshold = 8 (3rd highest score)
    mask = [1, 0, 0, 1, 0, 1] (keep scores >= 8)

    Args:
        importance_scores: Importance scores for each weight
                          Shape: [out_features, in_features] typically
                          Higher score = more important
        density: Fraction of weights to keep (0.0 to 1.0)
                 0.2 = keep top 20% most important
                 0.8 = keep top 80% most important

    Returns:
        Binary mask with same shape as importance_scores
        True = keep weight, False = drop weight
    """
    # Edge case: Keep everything
    if density >= 1.0:
        return torch.ones_like(importance_scores, dtype=torch.bool)

    # Edge case: Drop everything
    if density <= 0.0:
        return torch.zeros_like(importance_scores, dtype=torch.bool)

    # STEP 1: Calculate number of elements to keep
    k = int(density * importance_scores.numel())
    # Example: 4096*11008 weights * 0.2 density = ~9M weights to keep

    # STEP 2: Get threshold for top-k most important weights
    # topk returns (values, indices) sorted in descending order
    # We take the minimum value (kth largest) as our threshold
    threshold = torch.topk(importance_scores.flatten(), k).values.min()

    # STEP 3: Create binary mask
    # Keep all weights with importance >= threshold
    mask = importance_scores >= threshold
    # mask[i,j] = True if weight w[i,j] should be kept
    # mask[i,j] = False if weight w[i,j] should be dropped

    return mask


def apply_importance_mask(
    task_vector: torch.Tensor,
    mask: torch.Tensor,
    rescale: bool = True,
    density: float = None,
) -> torch.Tensor:
    """
    Apply importance-based mask to task vector.

    Args:
        task_vector: Task vector to mask
        mask: Binary mask indicating which weights to keep
        rescale: Whether to rescale remaining weights (like DARE)
        density: Density for rescaling (required if rescale=True)

    Returns:
        Masked (and optionally rescaled) task vector
    """
    masked_tv = task_vector * mask.to(task_vector.dtype)

    if rescale and density is not None and density > 0.0:
        masked_tv = masked_tv / density

    return masked_tv


class TaskVectorImportanceCalculator:
    """
    High-level interface for computing importance-based masks for task vectors.

    This class manages Hessian computation across multiple layers and provides
    a simple interface for importance-based pruning of task vectors.

    NOTE: This version computes Hessian from a SINGLE model (typically base model).
    For per-task Hessians from fine-tuned models, use TaskVectorImportanceCalculatorPerTask.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: List[Tuple[torch.Tensor, ...]],
        device: torch.device = None,
        percdamp: float = 0.01,
    ):
        """
        Initialize importance calculator.

        Args:
            model: The model to compute importance for (typically the base or fine-tuned model)
            calibration_loader: List of calibration data batches
            device: Device to perform computations on
            percdamp: Damping factor for Hessian inversion
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device if device is not None else next(model.parameters()).device
        self.percdamp = percdamp
        self.hessian_calculators: Dict[str, HessianCalculator] = {}
        self.hessian_inv_diags: Dict[str, torch.Tensor] = {}

    def compute_hessians_for_layers(self, layer_names: List[str], verbose: bool = True):
        """
        Compute Hessians for specified layers using calibration data.

        Args:
            layer_names: List of layer names to compute Hessians for
            verbose: Whether to print progress
        """
        # Register hooks to capture activations
        hooks = []
        activations = {name: [] for name in layer_names}

        def get_activation_hook(name):
            def hook(module, inp, out):
                activations[name].append(inp[0].detach().cpu())

            return hook

        # Register hooks
        for name in layer_names:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                hooks.append(layer.register_forward_hook(get_activation_hook(name)))

        # Run calibration data through model
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.calibration_loader):
                if verbose:
                    print(
                        f"Processing calibration batch {i+1}/{len(self.calibration_loader)}"
                    )

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inp = batch[0]
                else:
                    inp = batch

                inp = inp.to(self.device)
                _ = self.model(inp)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute Hessians from activations
        for name in layer_names:
            if verbose:
                print(f"Computing Hessian for layer: {name}")

            layer = self._get_layer_by_name(name)
            if layer is None:
                warnings.warn(f"Layer {name} not found in model")
                continue

            # Get layer shape
            if hasattr(layer, "weight"):
                weight_shape = layer.weight.shape
            else:
                warnings.warn(f"Layer {name} has no weight attribute")
                continue

            # Initialize Hessian calculator
            calc = HessianCalculator(weight_shape, device=self.device)

            # Add all activation batches
            for act in activations[name]:
                act = act.to(self.device)
                calc.add_batch(act)

            self.hessian_calculators[name] = calc

            # Compute inverse Hessian diagonal
            if verbose:
                print(f"Computing inverse Hessian diagonal for layer: {name}")
            self.hessian_inv_diags[name] = calc.get_inverse_hessian_diag(self.percdamp)

    def get_importance_mask_for_task_vector(
        self, layer_name: str, task_vector: torch.Tensor, density: float
    ) -> torch.Tensor:
        """
        Get importance-based mask for a task vector.

        Args:
            layer_name: Name of the layer this task vector corresponds to
            task_vector: The task vector to mask
            density: Fraction of weights to keep

        Returns:
            Binary mask
        """
        if layer_name not in self.hessian_inv_diags:
            raise ValueError(
                f"No Hessian computed for layer {layer_name}. Run compute_hessians_for_layers first."
            )

        hessian_inv_diag = self.hessian_inv_diags[layer_name]

        # Compute importance scores
        importance_scores = compute_importance_scores(task_vector, hessian_inv_diag)

        # Generate mask
        mask = generate_importance_mask(importance_scores, density)

        return mask

    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a layer by its name in the model."""
        try:
            parts = name.split(".")
            layer = self.model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except AttributeError:
            return None


# ============================================================================
# Simplified Interface Functions for Integration with TIES/DARE
# ============================================================================


def trim_with_sparsegpt_importance(
    task_vector: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
    density: float,
    rescale: bool = False,
) -> torch.Tensor:
    """
    Simplified function to trim task vector using SparseGPT importance.

    This is a drop-in replacement for the TIES trim() function.

    Args:
        task_vector: Task vector to trim
        hessian_inv_diag: Diagonal of inverse Hessian for this layer
        density: Fraction of weights to keep
        rescale: Whether to rescale (for DARE-style behavior)

    Returns:
        Trimmed task vector
    """
    importance_scores = compute_importance_scores(task_vector, hessian_inv_diag)
    mask = generate_importance_mask(importance_scores, density)
    return apply_importance_mask(task_vector, mask, rescale=rescale, density=density)


def drop_and_rescale_with_sparsegpt_importance(
    task_vector: torch.Tensor,
    hessian_inv_diag: torch.Tensor,
    density: float,
    rescale: bool = True,
) -> torch.Tensor:
    """
    Simplified function for DARE-style drop with SparseGPT importance.

    This is a drop-in replacement for the DARE drop_and_rescale() function.

    Args:
        task_vector: Task vector to drop weights from
        hessian_inv_diag: Diagonal of inverse Hessian for this layer
        density: Fraction of weights to keep
        rescale: Whether to rescale remaining weights

    Returns:
        Masked and rescaled task vector
    """
    return trim_with_sparsegpt_importance(
        task_vector, hessian_inv_diag, density, rescale
    )


# ============================================================================
# Per-Task Hessian Calculator (Using Fine-Tuned Models)
# ============================================================================


class TaskVectorImportanceCalculatorPerTask:
    """
    Advanced interface for computing task-specific importance using Hessians
    from FINE-TUNED models rather than base model.

    Key Difference from TaskVectorImportanceCalculator:
    - This computes separate Hessian for each fine-tuned model
    - More accurate task-specific curvature information
    - Theoretically optimal: H_ft captures task-specific loss landscape

    Usage:
        calc = TaskVectorImportanceCalculatorPerTask(
            ft_models=[ft_model_1, ft_model_2],
            calibration_loaders=[cal_data_1, cal_data_2],  # Task-specific data
            device='cuda'
        )

        # Compute Hessians for each task
        calc.compute_hessians_for_all_tasks(layer_names)

        # Get task-specific masks
        masks = calc.get_importance_masks_for_task_vectors(
            layer_name='layer.0.weight',
            task_vectors=[tv1, tv2],
            densities=[0.2, 0.2]
        )
    """

    def __init__(
        self,
        ft_models: List[nn.Module],
        calibration_loaders: List[List[Tuple[torch.Tensor, ...]]],
        device: torch.device = None,
        percdamp: float = 0.01,
    ):
        """
        Initialize per-task importance calculator.

        Args:
            ft_models: List of fine-tuned models (one per task)
            calibration_loaders: List of calibration data loaders (one per task)
            device: Device to perform computations on
            percdamp: Damping factor for Hessian inversion
        """
        if len(ft_models) != len(calibration_loaders):
            raise ValueError(
                f"Number of models ({len(ft_models)}) must match "
                f"number of calibration loaders ({len(calibration_loaders)})"
            )

        self.ft_models = ft_models
        self.calibration_loaders = calibration_loaders
        self.num_tasks = len(ft_models)
        self.device = (
            device if device is not None else next(ft_models[0].parameters()).device
        )
        self.percdamp = percdamp

        # Store Hessians per task: {task_idx: {layer_name: h_inv_diag}}
        self.hessian_inv_diags_per_task: Dict[int, Dict[str, torch.Tensor]] = {
            i: {} for i in range(self.num_tasks)
        }

    def compute_hessians_for_all_tasks(
        self, layer_names: List[str], verbose: bool = True
    ):
        """
        Compute Hessians for all tasks using their respective fine-tuned models.

        This is the KEY difference: each task gets its own Hessian computed
        from its fine-tuned model, capturing task-specific loss landscape.

        Args:
            layer_names: List of layer names to compute Hessians for
            verbose: Whether to print progress
        """
        for task_idx in range(self.num_tasks):
            if verbose:
                print(f"\n{'='*60}")
                print(f"COMPUTING HESSIANS FOR TASK {task_idx + 1}/{self.num_tasks}")
                print(f"{'='*60}")

            model = self.ft_models[task_idx]
            cal_loader = self.calibration_loaders[task_idx]

            # Compute Hessian for this task's fine-tuned model
            hessians = self._compute_hessians_for_model(
                model, cal_loader, layer_names, verbose
            )

            self.hessian_inv_diags_per_task[task_idx] = hessians

            if verbose:
                print(
                    f"\n✓ Completed Task {task_idx + 1}: {len(hessians)} layers computed"
                )

    def _compute_hessians_for_model(
        self,
        model: nn.Module,
        calibration_loader: List[Tuple[torch.Tensor, ...]],
        layer_names: List[str],
        verbose: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Hessians for a single model (internal method).
        """
        hessian_inv_diags = {}

        # Register hooks to capture activations
        hooks = []
        activations = {name: [] for name in layer_names}

        def get_activation_hook(name):
            def hook(module, inp, out):
                activations[name].append(inp[0].detach().cpu())

            return hook

        # Register hooks
        for name in layer_names:
            layer = self._get_layer_by_name(model, name)
            if layer is not None:
                hooks.append(layer.register_forward_hook(get_activation_hook(name)))

        # Run calibration data through model
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if verbose and (i + 1) % 32 == 0:
                    print(
                        f"  Processing calibration batch {i+1}/{len(calibration_loader)}"
                    )

                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inp = batch[0]
                else:
                    inp = batch

                inp = inp.to(self.device)
                _ = model(inp)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute Hessians from activations
        for name in layer_names:
            if verbose:
                print(f"  Computing Hessian for layer: {name}")

            layer = self._get_layer_by_name(model, name)
            if layer is None:
                warnings.warn(f"Layer {name} not found in model")
                continue

            # Get layer shape
            if hasattr(layer, "weight"):
                weight_shape = layer.weight.shape
            else:
                warnings.warn(f"Layer {name} has no weight attribute")
                continue

            # Initialize Hessian calculator
            calc = HessianCalculator(weight_shape, device=self.device)

            # Add all activation batches
            for act in activations[name]:
                act = act.to(self.device)
                calc.add_batch(act)

            # Compute inverse Hessian diagonal
            h_inv_diag = calc.get_inverse_hessian_diag(self.percdamp)
            hessian_inv_diags[name] = h_inv_diag

        return hessian_inv_diags

    def get_importance_masks_for_task_vectors(
        self, layer_name: str, task_vectors: List[torch.Tensor], densities: List[float]
    ) -> List[torch.Tensor]:
        """
        Get importance-based masks for task vectors using task-specific Hessians.

        This is where the magic happens: each task vector gets masked using
        the Hessian from its corresponding fine-tuned model!

        Args:
            layer_name: Name of the layer
            task_vectors: List of task vectors (one per task)
            densities: List of densities (one per task)

        Returns:
            List of binary masks (one per task)
        """
        if len(task_vectors) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task vectors, got {len(task_vectors)}"
            )

        if len(densities) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} densities, got {len(densities)}"
            )

        masks = []
        for task_idx in range(self.num_tasks):
            if layer_name not in self.hessian_inv_diags_per_task[task_idx]:
                raise ValueError(
                    f"No Hessian computed for task {task_idx}, layer {layer_name}"
                )

            # Get task-specific Hessian
            h_inv_diag = self.hessian_inv_diags_per_task[task_idx][layer_name]

            # Compute importance using task-specific Hessian
            importance = compute_importance_scores(task_vectors[task_idx], h_inv_diag)

            # Generate mask
            mask = generate_importance_mask(importance, densities[task_idx])
            masks.append(mask)

        return masks

    def get_hessian_for_task(self, task_idx: int, layer_name: str) -> torch.Tensor:
        """
        Get the Hessian inverse diagonal for a specific task and layer.

        Args:
            task_idx: Index of the task (0 to num_tasks-1)
            layer_name: Name of the layer

        Returns:
            Hessian inverse diagonal tensor
        """
        if task_idx not in self.hessian_inv_diags_per_task:
            raise ValueError(f"Invalid task index: {task_idx}")

        if layer_name not in self.hessian_inv_diags_per_task[task_idx]:
            raise ValueError(
                f"No Hessian computed for task {task_idx}, layer {layer_name}"
            )

        return self.hessian_inv_diags_per_task[task_idx][layer_name]

    def _get_layer_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a layer by its name in the model."""
        try:
            parts = name.split(".")
            layer = model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except AttributeError:
            return None


# ============================================================================
# Convenience Functions for Per-Task Hessians
# ============================================================================


def compute_per_task_hessians(
    ft_models: List[nn.Module],
    calibration_loaders: List[List[Tuple[torch.Tensor, ...]]],
    layer_names: List[str],
    device: torch.device = None,
    percdamp: float = 0.01,
    verbose: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Convenience function to compute per-task Hessians.

    Args:
        ft_models: List of fine-tuned models
        calibration_loaders: List of calibration data (one per task)
        layer_names: List of layer names
        device: Device to use
        percdamp: Damping factor
        verbose: Print progress

    Returns:
        Dictionary: {task_idx: {layer_name: h_inv_diag}}
    """
    calc = TaskVectorImportanceCalculatorPerTask(
        ft_models, calibration_loaders, device, percdamp
    )
    calc.compute_hessians_for_all_tasks(layer_names, verbose)
    return calc.hessian_inv_diags_per_task


def merge_with_per_task_hessians(
    base_model_params: torch.Tensor,
    ft_models_params: List[torch.Tensor],
    hessian_inv_diags: List[torch.Tensor],  # One per task
    weights: List[float],
    densities: List[float],
    method: str = "ties",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Merge models using per-task Hessians for importance calculation.

    Key difference: Each task vector uses its own Hessian from the fine-tuned model.

    Args:
        base_model_params: Base model parameters
        ft_models_params: List of fine-tuned model parameters
        hessian_inv_diags: List of Hessian diagonals (one per task)
        weights: Merge weights for each task
        densities: Densities for each task
        method: 'ties' or 'dare'
        device: Device to use

    Returns:
        Merged parameters
    """
    if method == "ties":
        from ties_utils import TIES

        merger = TIES()
    elif method == "dare":
        from dare_utils import DARE

        merger = DARE()
    else:
        raise ValueError(f"Unknown method: {method}")

    return merger.merge(
        weights=weights,
        base_model_parameters=base_model_params,
        ft_models_parameters=ft_models_params,
        densities=densities,
        device=device,
        hessian_inv_diags=hessian_inv_diags,
        use_sparsegpt=True,
    )
