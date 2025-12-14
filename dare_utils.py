from typing import List, Optional

import torch

from ties_utils import get_task_vector

try:
    from sparsegpt_task_vector import (
        compute_importance_scores,
        generate_importance_mask,
        prune_task_vector_with_error_correction,
    )

    SPARSEGPT_AVAILABLE = True
except ImportError:
    SPARSEGPT_AVAILABLE = False
    import warnings

    warnings.warn("SparseGPT module not found. Falling back to random dropout.")


def drop_and_rescale(
    task_vector: torch.Tensor,
    density: float,
    rescale: bool = True,
    hessian_inv: Optional[torch.Tensor] = None,
    use_sparsegpt: bool = False,
    blocksize: int = 128,
) -> torch.Tensor:
    """
    DARE (Drop And REscale): Drops weights with error correction and rescales.

    Two Modes:
    ----------
    1. Random dropout (original DARE):
       - Bernoulli sampling
       - Fast but ignores importance

    2. SparseGPT with error correction:
       - Blockwise OBS with error propagation
       - Maintains quality at high sparsity

    Args:
        task_vector: Task vector [out_features, in_features]
        density: Fraction to keep (0.2 = 20%)
        rescale: If True, scale by 1/density (DARE default)
        hessian_inv: FULL inverse Hessian [in_features, in_features]
                    Required if use_sparsegpt=True
        use_sparsegpt: If True, use error correction
        blocksize: Block size for error correction

    Returns:
        Dropped and rescaled task vector
    """

    # Edge case: No dropout needed
    if density >= 1.0:
        return task_vector

    # ========== MODE 1: SparseGPT with Error Correction ==========
    if use_sparsegpt:
        if not SPARSEGPT_AVAILABLE:
            raise ImportError(
                "SparseGPT module not available. Install sparsegpt_task_vector.py"
            )
        if hessian_inv is None:
            raise ValueError(
                "hessian_inv (full inverse) is required when use_sparsegpt=True"
            )

        # Use error-correcting pruning with rescaling
        return prune_task_vector_with_error_correction(
            task_vector=task_vector,
            hessian_inv=hessian_inv,
            density=density,
            blocksize=blocksize,
            rescale=rescale,  # DARE rescales
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
        hessian_invs: Optional[List[torch.Tensor]] = None,  # Full inverse Hessians
        importance_masks: Optional[List[torch.Tensor]] = None,  # Pre-computed masks
        use_sparsegpt: bool = False,
        blocksize: int = 128,  # For error correction
        **kwargs,
    ) -> torch.Tensor:
        """
        Merges task vectors using DARE with SparseGPT error correction.

        Args:
            weights: Task weights [1.0, 0.5, ...]
            base_model_parameters: Base model parameters
            ft_models_parameters: List of fine-tuned parameters
            densities: Fraction to keep [0.2, 0.2, ...]
            device: Computation device
            hessian_invs: List of FULL inverse Hessians [in_features, in_features]
            importance_masks: Pre-computed masks (alternative)
            use_sparsegpt: If True, use error correction
            blocksize: Block size for error correction

        Returns:
            Merged model parameters
        """
        if device is None:
            device = base_model_parameters.device

        # Get task vectors
        task_vectors, base_model_parameters = get_task_vector(
            base_model_parameters,
            ft_models_parameters,
            device=device,
        )

        # ========== VALIDATE INPUTS - CHECK FOR NaN/Inf IN TASK VECTORS ==========
        for idx, tv in enumerate(task_vectors):
            if torch.isnan(tv).any():
                import warnings

                nan_count = torch.isnan(tv).sum().item()
                warnings.warn(
                    f"Task vector {idx} contains {nan_count} NaN values! "
                    f"This indicates corrupted fine-tuned model weights. "
                    f"Replacing NaN with zeros."
                )
                task_vectors[idx] = torch.nan_to_num(tv, nan=0.0)

            if torch.isinf(tv).any():
                import warnings

                inf_count = torch.isinf(tv).sum().item()
                warnings.warn(
                    f"Task vector {idx} contains {inf_count} Inf values! "
                    f"This indicates numerical instability. "
                    f"Replacing Inf with zeros."
                )
                task_vectors[idx] = torch.nan_to_num(
                    tv, nan=0.0, posinf=0.0, neginf=0.0
                )

        # Validate inputs for SparseGPT mode
        if use_sparsegpt:
            if importance_masks is not None:
                if len(importance_masks) != len(task_vectors):
                    raise ValueError(
                        f"Number of importance_masks ({len(importance_masks)}) must match number of task vectors ({len(task_vectors)})"
                    )
            elif hessian_invs is not None:
                if len(hessian_invs) != len(task_vectors):
                    raise ValueError(
                        f"Number of hessian_invs ({len(hessian_invs)}) must match number of task vectors ({len(task_vectors)})"
                    )
            else:
                raise ValueError(
                    "Either importance_masks or hessian_invs (full inverse) is required when use_sparsegpt=True"
                )

        # Trim task vectors with error correction
        if use_sparsegpt and importance_masks is not None:
            # Apply pre-computed masks with rescaling
            trimmed_task_vectors = []
            for tv, mask in zip(task_vectors, importance_masks):
                if mask.numel() == tv.numel():
                    mask_reshaped = mask.reshape(tv.shape)
                else:
                    raise ValueError(
                        f"Mask size ({mask.numel()}) doesn't match task vector size ({tv.numel()})"
                    )
                masked_tv = tv * mask_reshaped
                # DARE rescaling
                non_zero_count = (mask_reshaped != 0).sum().item()
                actual_density = non_zero_count / mask_reshaped.numel()
                if actual_density > 0:
                    trimmed_task_vectors.append(masked_tv / actual_density)
                else:
                    trimmed_task_vectors.append(masked_tv)
        elif use_sparsegpt and hessian_invs is not None:
            # Error correction with full inverse Hessians
            trimmed_task_vectors = [
                drop_and_rescale(
                    tv,
                    density,
                    rescale=True,
                    hessian_inv=h_inv,
                    use_sparsegpt=True,
                    blocksize=blocksize,
                )
                for tv, density, h_inv in zip(task_vectors, densities, hessian_invs)
            ]
        else:
            # Random dropout (original DARE)
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

        # ========== CRITICAL: CHECK FOR NaN/Inf ==========
        if torch.isnan(merged_task_vector).any():
            import warnings

            nan_count = torch.isnan(merged_task_vector).sum().item()
            warnings.warn(
                f"NaN detected in DARE merge! {nan_count}/{merged_task_vector.numel()} values. "
                f"Replacing NaN with zeros."
            )
            merged_task_vector = torch.nan_to_num(merged_task_vector, nan=0.0)

        if torch.isinf(merged_task_vector).any():
            import warnings

            inf_count = torch.isinf(merged_task_vector).sum().item()
            warnings.warn(
                f"Inf detected in DARE merge! {inf_count}/{merged_task_vector.numel()} values. "
                f"Replacing Inf with zeros."
            )
            merged_task_vector = torch.nan_to_num(
                merged_task_vector, nan=0.0, posinf=0.0, neginf=0.0
            )

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
