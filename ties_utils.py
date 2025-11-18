from typing import Dict, List, Literal, Optional, Tuple

import torch

try:
    from sparsegpt_importance import (
        apply_importance_mask,
        compute_importance_scores,
        generate_importance_mask,
        trim_with_sparsegpt_importance,
    )

    SPARSEGPT_AVAILABLE = True
except ImportError:
    SPARSEGPT_AVAILABLE = False
    import warnings

    warnings.warn(
        "SparseGPT importance module not found. Falling back to magnitude-based trimming."
    )


def get_task_vector(
    base_model_parameters: torch.Tensor,
    ft_models_parameters: List[torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Computes the task vector as the difference between fine-tuned and base model parameters.

    Mathematical Definition:
    -----------------------
    Task Vector τ = θ_finetuned - θ_base

    This captures the "direction" and "magnitude" of the fine-tuning update.

    Why Task Vectors?
    ----------------
    - Base model: θ_base (pretrained, general knowledge)
    - Fine-tuned model: θ_ft = θ_base + τ (specialized for task)
    - Task vector τ represents task-specific knowledge

    Merging Multiple Tasks:
    ----------------------
    θ_merged = θ_base + Σ(weight_i * τ_i)

    This allows combining multiple specialized models while
    preserving the base model's general capabilities.

    Args:
        base_model_parameters: The parameters of the base model (pretrained)
                              Shape: [out_features, in_features] for linear layers
        ft_models_parameters: List of parameters from different fine-tuned models
                            Each has same shape as base_model_parameters
                            Example: [model_task1_params, model_task2_params, ...]
        device: Device to perform computations on. If None, uses current device

    Returns:
        Tuple containing:
        1. List of task vectors (one per fine-tuned model)
           task_vectors[i] = ft_models_parameters[i] - base_model_parameters
        2. Base model parameters (possibly moved to specified device)

    Example:
    -------
    base = [1.0, 2.0, 3.0]
    ft1 = [1.5, 2.5, 3.5]  # Fine-tuned for sentiment analysis
    ft2 = [0.5, 1.5, 2.5]  # Fine-tuned for question answering

    task_vec1 = [0.5, 0.5, 0.5]  # Sentiment-specific update
    task_vec2 = [-0.5, -0.5, -0.5]  # QA-specific update
    """
    # Move parameters to specified device if provided
    if device is not None:
        base_model_parameters = base_model_parameters.to(device)
        ft_models_parameters = [params.to(device) for params in ft_models_parameters]

    # STEP 1: Stack fine-tuned parameters for vectorized computation
    # Convert list [ft1, ft2, ...] to tensor of shape [num_tasks, ...]
    ft_params_stacked = torch.stack(ft_models_parameters)
    # Example: 3 models with [4096, 11008] → [3, 4096, 11008]

    # STEP 2: Compute task vectors via broadcasting
    # Subtract base from each fine-tuned model
    # base shape: [4096, 11008]
    # ft_stacked shape: [3, 4096, 11008]
    # After unsqueeze: [1, 4096, 11008] - [3, 4096, 11008] = [3, 4096, 11008]
    task_vectors_stacked = ft_params_stacked - base_model_parameters.unsqueeze(0)

    # STEP 3: Convert back to list of individual task vectors
    # [3, 4096, 11008] → [task_vec1, task_vec2, task_vec3]
    task_vectors = list(task_vectors_stacked.unbind(dim=0))

    return task_vectors, base_model_parameters


def trim(
    task_vector: torch.Tensor,
    density: float,
    hessian_inv_diag: Optional[torch.Tensor] = None,
    use_sparsegpt: bool = False,
) -> torch.Tensor:
    """
    Trims the task vector to retain only the top-k% of its elements.

    Two Modes:
    ----------
    1. Magnitude-based trimming (original TIES):
       - Score = |weight|
       - Keep weights with largest absolute values
       - Simple but ignores weight sensitivity

    2. SparseGPT importance-based trimming:
       - Score = weight^2 / (H^-1)^2
       - Keep weights with highest importance scores
       - Considers both magnitude and Hessian sensitivity

    Why Trim Task Vectors?
    ----------------------
    Task vectors often contain noise and conflicting updates.
    Trimming keeps only the most important changes, which:
    - Reduces interference between tasks
    - Improves merged model performance
    - Removes low-confidence updates

    Mathematical Comparison:
    -----------------------
    TIES (magnitude):
      score_ij = |τ_ij|

    SparseGPT (importance):
      score_ij = τ_ij^2 / (H_jj^-1)^2
      where H_jj^-1 = sensitivity of input feature j

    Key Difference from DARE:
    -------------------------
    - DARE: drop_and_rescale (rescales remaining weights)
    - TIES: trim (no rescaling, just masking)

    Args:
        task_vector: The task vector to be trimmed
                    Shape: [out_features, in_features]
                    Represents: fine_tuned_weights - base_weights
        density: Fraction of elements to retain (between 0 and 1)
                0.2 = keep top 20% most important weights
                0.8 = keep top 80% most important weights
        hessian_inv_diag: Diagonal of inverse Hessian [in_features]
                         Required if use_sparsegpt=True
                         One value per INPUT feature (column)
        use_sparsegpt: If True, use SparseGPT importance-based trimming
                      If False, use magnitude-based trimming (default)

    Returns:
        torch.Tensor: The trimmed task vector (same shape, but sparsified)

    Example:
    -------
    task_vec = [[1.0, -2.0, 0.5],
                [0.3, 1.5, -0.8]]
    density = 0.5 (keep 50%, drop 50%)

    Magnitude mode:
    - |values| = [[1.0, 2.0, 0.5], [0.3, 1.5, 0.8]]
    - threshold = 1.0 (3rd largest)
    - mask = [[1, 1, 0], [0, 1, 0]]
    - result = [[1.0, -2.0, 0], [0, 1.5, 0]]

    SparseGPT mode:
    - importance considers H^-1 diagonal
    - different weights may be kept
    """
    # Edge case: No trimming needed
    if density >= 1.0:
        return task_vector

    # ========== MODE 1: SparseGPT Importance-Based Trimming ==========
    if use_sparsegpt:
        if not SPARSEGPT_AVAILABLE:
            raise ImportError(
                "SparseGPT importance module not available. Install sparsegpt_importance.py"
            )
        if hessian_inv_diag is None:
            raise ValueError("hessian_inv_diag is required when use_sparsegpt=True")

        # Delegate to SparseGPT implementation (no rescaling for TIES)
        return trim_with_sparsegpt_importance(
            task_vector, hessian_inv_diag, density, rescale=False
        )

    # ========== MODE 2: Magnitude-Based Trimming (Original TIES) ==========

    # STEP 1: Calculate number of elements to retain
    k = int(density * task_vector.numel())
    if k < 0:
        raise ValueError("Density must be between 0 and 1.")

    # STEP 2: Get threshold value for top-k by magnitude
    # Use absolute value to consider both positive and negative weights
    threshold = torch.topk(task_vector.abs().view(-1), k).values.min()
    # Example: Keep all weights where |weight| >= threshold

    # STEP 3: Create mask for elements to retain
    mask = task_vector.abs() >= threshold
    # mask[i,j] = True if |task_vector[i,j]| >= threshold

    # STEP 4: Apply mask (zero out small weights, keep large ones)
    return task_vector * mask


def get_elect_mask(
    task_vectors_stacked: torch.Tensor,
    method: Literal["sum", "count"] = "sum",  # TIES-merging uses "sum"
    mask_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Computes a boolean mask indicating where each task vector's sign matches the elected sign.

    This is the "sign election" step in TIES merging, which resolves conflicts
    between task vectors that push parameters in opposite directions.

    Mathematical Background:
    -----------------------
    When merging multiple task vectors, they may have conflicting signs:
    - Task 1: τ_1[i] = +2.0 (wants to increase weight)
    - Task 2: τ_2[i] = -1.5 (wants to decrease weight)
    - Task 3: τ_3[i] = +0.8 (wants to increase weight)

    Sign election resolves this by:
    1. Computing aggregate sign (majority vote or sum)
    2. Discarding updates that disagree with elected sign
    3. Averaging only the agreeing updates

    Two Methods:
    -----------
    1. "sum" (TIES default):
       elected_sign[i] = sign(Σ τ_j[i])
       Example: sum = +2.0 - 1.5 + 0.8 = +1.3 → positive elected

    2. "count":
       elected_sign[i] = sign(count(positive) - count(negative))
       Example: 2 positive, 1 negative → positive elected

    Why This Helps:
    --------------
    Without sign election, conflicting updates would partially cancel out,
    losing information. With election, we keep the "consensus" direction
    and discard outliers.

    Args:
        task_vectors_stacked: Stacked task vectors of shape (num_tasks, num_params)
                             Example: [3, 4096, 11008] for 3 tasks
                             task_vectors_stacked[i] = task vector for task i
        method: Method to compute the sign mask
                "sum" - sums the task vectors (default, used by TIES)
                "count" - counts number of positive vs negative signs
        mask_dtype: Desired data type of output mask. If None, uses input dtype

    Returns:
        torch.Tensor: Boolean mask of shape (num_tasks, num_params)
                     mask[i, j] = True if task i's sign matches elected sign at position j
                     mask[i, j] = False if task i disagrees with elected sign

    Example:
    -------
    task_vectors = [[+1.0, -2.0, +0.5],
                    [-0.5, +1.0, +0.3],
                    [+0.8, -0.5, -0.2]]

    Method "sum":
    - sum = [+1.3, -1.5, +0.6]
    - elected_sign = [+1, -1, +1]
    - mask[0] = [T, T, T]  (all match)
    - mask[1] = [F, F, T]  (first two disagree)
    - mask[2] = [T, T, F]  (last one disagrees)

    Only weights that "agree" with the elected sign contribute to final merge.
    """
    if mask_dtype is None:
        mask_dtype = task_vectors_stacked.dtype

    # STEP 1: Get sign of each element (-1 or +1)
    # sign converts: positive → +1, negative → -1, zero → 0
    sign = task_vectors_stacked.sign().to(mask_dtype)
    # Example shape: [3, 4096, 11008] → [3, 4096, 11008]

    # STEP 2: Compute elected sign based on method
    if method == "sum":
        # Sum all task vectors and take sign
        # Elected sign = sign(Σ_i τ_i)
        elected_sign = (task_vectors_stacked.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
        # .sum(dim=0) → [4096, 11008] (aggregate across tasks)
        # >= 0 → boolean [4096, 11008]
        # * 2 - 1 → convert True/False to +1/-1

    elif method == "count":
        # Count number of positive vs negative signs
        # Elected sign = sign(count(positive) - count(negative))
        elected_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
        # sign.sum(dim=0) → counts net positive signs
        # Example: [+1, -1, +1] → sum = +1 → elected = +1

    else:
        raise ValueError(f"Unknown method: {method}. Use 'sum' or 'count'.")

    # STEP 3: Create mask where sign matches elected sign
    # True where task vector agrees with consensus
    # False where task vector disagrees
    return sign == elected_sign  # Broadcasting: [3, H, W] == [H, W] → [3, H, W]


# FOR TIES MERGING
# TESTING CODE AT THE BOTTOM
class TIES:
    def __init__(self, config=None):
        if config is not None:
            self.config = config

    def merge(
        self,
        weights: List[float],  # weights for each task vector
        base_model_parameters: torch.Tensor,
        ft_models_parameters: List[torch.Tensor],
        densities: List[float],
        device: Optional[torch.device] = None,
        hessian_inv_diags: Optional[List[torch.Tensor]] = None,
        importance_masks: Optional[
            List[torch.Tensor]
        ] = None,  # NEW: Pre-computed masks
        use_sparsegpt: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Merges multiple task vectors using the TIES (Trim, Elect, Merge) method.

        TIES Algorithm (Yadav et al., 2023):
        ------------------------------------
        1. TRIM: Remove low-importance weights from each task vector
           - Original: Keep top-k by magnitude |τ_i|
           - SparseGPT: Keep top-k by importance τ_i^2 / (H^-1)^2

        2. ELECT: Resolve sign conflicts via majority voting
           - Compute elected sign: sign(Σ τ_i)
           - Discard updates that disagree with elected sign

        3. MERGE: Weighted average of agreeing updates
           - θ_merged = θ_base + Σ(w_i * τ_i * elect_mask) / Σ(w_i * elect_mask)

        Why TIES Works:
        --------------
        - TRIM removes noise and low-confidence updates
        - ELECT resolves conflicts (prevents cancellation)
        - MERGE averages only consistent updates

        Result: Better performance than naive averaging, especially with
        conflicting task vectors.

        Mathematical Formula:
        --------------------
        θ_merged[j] = θ_base[j] +
                      Σ_i (w_i * τ_i[j] * 1_{sign(τ_i[j]) = sign(Σ_k τ_k[j])}) /
                      Σ_i (w_i * 1_{sign(τ_i[j]) = sign(Σ_k τ_k[j])})

        Args:
            weights: Weights for each task vector (how much to trust each task)
                    Example: [1.0, 0.5, 1.0] = trust task 1 and 3 equally, task 2 half as much
            base_model_parameters: The pretrained base model parameters
                                  Shape: [out_features, in_features] for linear layers
            ft_models_parameters: List of fine-tuned model parameters (one per task)
                                 Each has same shape as base_model_parameters
            densities: List of densities for trimming each task vector
                      Example: [0.2, 0.2, 0.2] = keep top 20% for all tasks
                      Can be different per task: [0.1, 0.3, 0.2]
            device: Device to perform computations on. If None, uses current device
            hessian_inv_diags: List of inverse Hessian diagonals (one per task)
                              Required if use_sparsegpt=True
                              Each has shape [in_features]
            use_sparsegpt: If True, use SparseGPT importance for TRIM step
                          If False, use magnitude-based trimming (default)

        Returns:
            torch.Tensor: The merged model parameters
            Shape same as base_model_parameters

        Example Workflow:
        ----------------
        Input:
        - base = [1, 2, 3, 4]
        - ft1 = [2, 3, 2, 5] → τ1 = [+1, +1, -1, +1]
        - ft2 = [0, 1, 4, 3] → τ2 = [-1, -1, +1, -1]
        - ft3 = [2, 3, 2, 5] → τ3 = [+1, +1, -1, +1]
        - densities = [1.0, 1.0, 1.0] (no trim for example)

        Step 1 - TRIM: (skipped, density=1.0)

        Step 2 - ELECT:
        - sum = [+1, +1, -1, +1]
        - elected = [+1, +1, -1, +1]
        - elect_mask1 = [T, T, T, T] (all agree)
        - elect_mask2 = [F, F, F, F] (all disagree!)
        - elect_mask3 = [T, T, T, T] (all agree)

        Step 3 - MERGE (with weights=[1, 1, 1]):
        - merged[0] = base[0] + (1*1 + 1*(-1)*0 + 1*1) / (1 + 0 + 1) = 1 + 1 = 2
        - merged[1] = base[1] + (1*1 + 1*(-1)*0 + 1*1) / (1 + 0 + 1) = 2 + 1 = 3
        - etc.

        Task 2 is completely discarded due to disagreement!
        """
        if device is None:
            device = base_model_parameters.device

        # ========== STEP 0: COMPUTE TASK VECTORS ==========
        # τ_i = θ_finetuned_i - θ_base
        task_vectors, base_model_parameters = get_task_vector(
            base_model_parameters,
            ft_models_parameters,
            device=device,
        )
        # task_vectors is list of [out_features, in_features] tensors

        # ========== VALIDATE INPUTS FOR SPARSEGPT MODE ==========
        # Two modes: importance_masks (pre-computed) OR hessian_inv_diags (compute on-the-fly)
        if use_sparsegpt:
            if importance_masks is not None:
                # Mode 1: Use pre-computed importance masks
                if len(importance_masks) != len(task_vectors):
                    raise ValueError(
                        f"Number of importance_masks ({len(importance_masks)}) must match number of task vectors ({len(task_vectors)})"
                    )
            elif hessian_inv_diags is not None:
                # Mode 2: Compute masks from Hessian diagonals
                if len(hessian_inv_diags) != len(task_vectors):
                    raise ValueError(
                        f"Number of hessian_inv_diags ({len(hessian_inv_diags)}) must match number of task vectors ({len(task_vectors)})"
                    )
            else:
                raise ValueError(
                    "Either importance_masks or hessian_inv_diags is required when use_sparsegpt=True"
                )

        # ========== STEP 1: TRIM TASK VECTORS ==========
        # Keep only top-k% most important weights in each task vector
        if use_sparsegpt and importance_masks is not None:
            # Mode 1: Apply pre-computed importance masks directly
            # IMPORTANT: Masks are already computed with correct density, just apply them
            trimmed_task_vectors = []
            for tv, mask in zip(task_vectors, importance_masks):
                # Ensure mask is same shape as task vector
                if mask.numel() == tv.numel():
                    mask_reshaped = mask.reshape(tv.shape)
                else:
                    raise ValueError(
                        f"Mask size ({mask.numel()}) doesn't match task vector size ({tv.numel()})"
                    )
                # Apply mask (mask is float, 0.0 or 1.0)
                trimmed_task_vectors.append(tv * mask_reshaped)
        elif use_sparsegpt and hessian_inv_diags is not None:
            # Mode 2: SparseGPT mode with Hessian - compute importance on-the-fly
            trimmed_task_vectors = [
                trim(tv, density, hessian_inv_diag=h_inv, use_sparsegpt=True)
                for tv, density, h_inv in zip(
                    task_vectors, densities, hessian_inv_diags
                )
            ]
        else:
            # Magnitude mode: importance = |τ|
            trimmed_task_vectors = [
                trim(tv, density) for tv, density in zip(task_vectors, densities)
            ]
        # trimmed_task_vectors: list of sparse task vectors

        # ========== STEP 2: STACK FOR VECTORIZED OPERATIONS ==========
        # Convert list to tensor for efficient computation
        task_vectors_stacked = torch.stack(trimmed_task_vectors).to(device)
        # Shape: [num_tasks, out_features, in_features]
        # Example: [3, 4096, 11008]

        # ========== STEP 3: ELECT - COMPUTE SIGN CONSENSUS ==========
        # Determine which direction (positive or negative) wins for each parameter
        elect_mask = get_elect_mask(task_vectors_stacked, method="sum").to(
            device if device is not None else task_vectors_stacked.device
        )
        # elect_mask[i, j, k] = True if task i's sign at [j,k] matches elected sign
        # Shape: [num_tasks, out_features, in_features]

        # ========== STEP 4: PREPARE TASK WEIGHTS ==========
        # Convert weights list to tensor and add dimensions for broadcasting
        weights = torch.tensor(
            weights,
            dtype=task_vectors_stacked.dtype,
            device=device if device is not None else task_vectors_stacked.device,
        )
        # Shape: [num_tasks]

        # Add dimensions to match task_vectors_stacked shape
        while len(weights.shape) < len(task_vectors_stacked.shape):
            weights = weights.unsqueeze(-1)
        # Shape: [num_tasks, 1, 1] → broadcasts to [num_tasks, out_feat, in_feat]

        # ========== STEP 5: MERGE - WEIGHTED AVERAGE OF AGREEING UPDATES ==========
        # Apply both weights and election mask
        # Only task vectors that agree with elected sign contribute
        weighted_task_vectors = task_vectors_stacked * weights * elect_mask
        # Shape: [num_tasks, out_features, in_features]

        # Sum across tasks (dim=0)
        merged_update = weighted_task_vectors.sum(dim=0)
        # Shape: [out_features, in_features]

        # ========== STEP 6: NORMALIZE BY SUM OF AGREEING WEIGHTS ==========
        # Compute normalization factor: sum of weights where election mask is True
        # This ensures proper averaging (not just summing)
        normalization_factor = (
            (weights * elect_mask.to(weights.dtype)).sum(dim=0).clamp(min=1e-10)
        )  # Clamp to avoid division by zero
        # Shape: [out_features, in_features]

        # Divide to get weighted average
        normalized_merged_update = merged_update / normalization_factor
        # Shape: [out_features, in_features]

        # ========== STEP 7: ADD MERGED UPDATE TO BASE MODEL ==========
        # Final merged parameters: θ_merged = θ_base + average(agreeing τ_i)
        merged_model_parameters = base_model_parameters + normalized_merged_update

        return merged_model_parameters


# EXMPLE CODE FOR TESTING

if __name__ == "__main__":
    # Example usage
    base_params = torch.tensor([0.0, 0.0, 0.0, 0.0])
    ft_params_1 = torch.tensor([1.0, -1.0, 0.5, -0.5])
    ft_params_2 = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    ft_params_3 = torch.tensor([0.5, 0.5, -1.0, -1.0])

    weights = [1.0, 1.0, 1.0]
    densities = [1.0, 1.0, 1.0]

    ties = TIES()
    merged_params = ties.merge(
        weights=weights,
        base_model_parameters=base_params,
        ft_models_parameters=[ft_params_1, ft_params_2, ft_params_3],
        densities=densities,
        device=torch.device("cpu"),
    )

    print("Merged Parameters:", merged_params)
