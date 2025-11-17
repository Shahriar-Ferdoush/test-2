from typing import List, Optional

import torch
from ties_utils import get_task_vector

try:
    from sparsegpt_importance import (
        apply_importance_mask,
        compute_importance_scores,
        drop_and_rescale_with_sparsegpt_importance,
        generate_importance_mask,
    )

    SPARSEGPT_AVAILABLE = True
except ImportError:
    SPARSEGPT_AVAILABLE = False
    import warnings

    warnings.warn(
        "SparseGPT importance module not found. Falling back to random dropout."
    )


def drop_and_rescale(
    task_vector: torch.Tensor,
    density: float,
    rescale: bool = True,
    hessian_inv_diag: Optional[torch.Tensor] = None,
    use_sparsegpt: bool = False,
) -> torch.Tensor:
    """
    DARE (Drop And REscale) method: Drops weights based on importance and rescales remaining weights.

    Two Modes:
    ----------
    1. Random dropout (original DARE):
       - Uses Bernoulli sampling (random coin flip per weight)
       - Fast but ignores weight importance

    2. SparseGPT importance-based dropout:
       - Uses Hessian-based importance: w^2 / (H^-1)^2
       - Keeps top-k most important weights
       - More principled than random dropout

    Mathematical Justification (DARE Paper):
    ----------------------------------------
    Even if we drop >90% of weight updates and rescale by 1/density,
    the merged model performance doesn't degrade significantly.

    Why? Task vectors are often redundant - only a small fraction
    of updates are critical for performance.

    Rescaling Formula:
    -----------------
    masked_weight = weight * mask / density

    This preserves expected magnitude:
    E[masked_weight] = weight * E[mask] / density
                     = weight * density / density
                     = weight

    Args:
        task_vector: The task vector to be trimmed
                    Shape: typically [out_features, in_features]
                    Represents: fine_tuned_weights - base_weights
        density: The density level for trimming (between 0 and 1)
                0.2 = keep 20% of weights, drop 80%
                0.8 = keep 80% of weights, drop 20%
        rescale: Whether to rescale the remaining weights by 1/density
                True (default) = preserve magnitude
                False = just mask without rescaling
        hessian_inv_diag: Diagonal of inverse Hessian [in_features]
                         Required if use_sparsegpt=True
                         One value per INPUT feature (column)
        use_sparsegpt: If True, use SparseGPT importance-based dropping
                      If False, use random dropout (default)

    Returns:
        torch.Tensor: The trimmed (and possibly rescaled) task vector

    Example:
    -------
    task_vec = [1.0, 2.0, 3.0, 4.0]
    density = 0.5

    Random mode:
    - mask = [1, 0, 1, 0] (random)
    - result = [1.0/0.5, 0, 3.0/0.5, 0] = [2.0, 0, 6.0, 0]

    SparseGPT mode:
    - importance = [0.1, 0.5, 0.3, 0.8]
    - mask = [0, 1, 0, 1] (top 2)
    - result = [0, 2.0/0.5, 0, 4.0/0.5] = [0, 4.0, 0, 8.0]
    """

    # Edge case: No dropout needed
    if density >= 1.0:
        return task_vector

    # ========== MODE 1: SparseGPT Importance-Based Dropout ==========
    if use_sparsegpt:
        if not SPARSEGPT_AVAILABLE:
            raise ImportError(
                "SparseGPT importance module not available. Install sparsegpt_importance.py"
            )
        if hessian_inv_diag is None:
            raise ValueError("hessian_inv_diag is required when use_sparsegpt=True")

        # Delegate to SparseGPT implementation
        return drop_and_rescale_with_sparsegpt_importance(
            task_vector, hessian_inv_diag, density, rescale
        )

    # ========== MODE 2: Random Dropout (Original DARE) ==========

    # Choose working dtype for precision
    if (task_vector.device.type != "cpu") or (task_vector.dtype == torch.bfloat16):
        working_dtype = task_vector.dtype
    else:
        working_dtype = torch.float32

    # STEP 1: Generate random mask via Bernoulli sampling
    # Bernoulli(p) = 1 with probability p, 0 with probability (1-p)
    # Here p = density, so each weight kept with probability = density
    mask = torch.bernoulli(torch.full_like(task_vector, density, dtype=working_dtype))
    # Example: density=0.5 → each element has 50% chance to be kept

    # STEP 2: Apply mask (zero out dropped weights)
    result = task_vector.to(working_dtype) * mask

    # STEP 3: Rescale remaining weights to preserve magnitude
    if rescale and density > 0.0:
        result = result / density
        # Dividing by density compensates for dropped weights
        # E.g., if we keep 20% of weights, scale up by 5x

    return result.to(task_vector.dtype)


class DARE:
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
        use_sparsegpt: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Merges multiple task vectors into a single parameter update using the DARE method.

        Args:
            weights (List[float]): Weights for each task vector.
            base_model_parameters (torch.Tensor): The parameters of the base model.
            ft_models_parameters (List[torch.Tensor]): List of parameters from different adapted models.
            densities (List[float]): List of densities for trimming each task vector.
            device (torch.device, optional): Device to perform computations on. If None, uses current device.
            hessian_inv_diags (Optional[List[torch.Tensor]]): List of inverse Hessian diagonals for each task vector.
                                                                Required if use_sparsegpt=True.
            use_sparsegpt (bool): If True, use SparseGPT importance-based dropping instead of random dropout.

        Returns:
            torch.Tensor: The merged model parameters after applying the DARE method.
        """
        if device is None:
            device = base_model_parameters.device

        # Get task vectors
        task_vectors, base_model_parameters = get_task_vector(
            base_model_parameters,
            ft_models_parameters,
            device=device,
        )

        # Validate inputs for SparseGPT mode
        if use_sparsegpt and hessian_inv_diags is None:
            raise ValueError("hessian_inv_diags is required when use_sparsegpt=True")

        if use_sparsegpt and len(hessian_inv_diags) != len(task_vectors):
            raise ValueError(
                f"Number of hessian_inv_diags ({len(hessian_inv_diags)}) must match number of task vectors ({len(task_vectors)})"
            )

        # Trim task vectors based on densities
        if use_sparsegpt:
            trimmed_task_vectors = [
                drop_and_rescale(
                    tv,
                    density,
                    rescale=True,
                    hessian_inv_diag=h_inv,
                    use_sparsegpt=True,
                )
                for tv, density, h_inv in zip(
                    task_vectors, densities, hessian_inv_diags
                )
            ]
        else:
            trimmed_task_vectors = [
                drop_and_rescale(tv, density)
                for tv, density in zip(task_vectors, densities)
            ]

        # Process weights
        weights = torch.tensor(
            weights,
            dtype=trimmed_task_vectors[0].dtype,
            device=device if device is not None else trimmed_task_vectors[0].device,
        )

        while (
            len(weights.shape) < len(trimmed_task_vectors[0].shape) + 1
        ):  # +1 for stacking dim
            weights = weights.unsqueeze(-1)

        # Weighted sum of trimmed task vectors
        weighted_task_vector = (
            torch.stack(trimmed_task_vectors).to(weights.device) * weights
        )
        merged_task_vector = weighted_task_vector.sum(dim=0)

        # Add merged task vector to base model parameters
        merged_model_parameters = base_model_parameters + merged_task_vector

        return merged_model_parameters


# EXAMPLE USAGE
if __name__ == "__main__":
    # Example usage of DARE merging
    base_model_params = torch.tensor([1.0, 2.0, 3.0])
    ft_model_params_1 = torch.tensor([1.5, 2.5, 3.5])
    ft_model_params_2 = torch.tensor([0.5, 1.5, 2.5])

    weights = [0.6, 0.4]
    densities = [0.8, 0.5]

    dare_merger = DARE()
    merged_params = dare_merger.merge(
        weights=weights,
        base_model_parameters=base_model_params,
        ft_models_parameters=[ft_model_params_1, ft_model_params_2],
        densities=densities,
    )

    print("Merged Model Parameters:", merged_params)
