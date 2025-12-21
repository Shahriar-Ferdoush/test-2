"""
SparseGPT-based Task Vector Pruning with Error Correction

This implements the SparseGPT algorithm for pruning task vectors in model merging.
Uses Optimal Brain Surgeon (OBS) with blockwise error propagation to maintain
model quality at high sparsity levels.

Key Difference from Original SparseGPT:
---------------------------------------
Original SparseGPT prunes model weights directly:
    W_pruned = prune(W)

Task Vector SparseGPT prunes task vectors (deltas):
    task_vector = W_finetuned - W_base
    task_vector_pruned = prune(task_vector)
    W_merged = W_base + task_vector_pruned

Architecture Similarity:
-----------------------
This code follows the exact structure of sparsegpt.py:
- SparseGPTTaskVector class (analogous to SparseGPT class)
- add_batch() method for Hessian accumulation
- fasterprune() method for pruning with error correction
- Sequential layer processing like llama_sequential()

The key insight: For accurate Hessian computation in sequential layers,
we must add pruned task vectors back to base model weights before computing
inputs to the next layer.

Reference: Frantar & Alistarh (2023) - "SparseGPT: Massive Language Models
           Can Be Accurately Pruned in One-Shot"
"""

import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SparseGPTTaskVector:
    """
    SparseGPT pruning algorithm adapted for task vectors.

    Structure mirrors sparsegpt.py::SparseGPT class exactly.
    Key difference: We prune task vectors (deltas) instead of weights.

    Usage (following SparseGPT pattern):
        1. Create instance: pruner = SparseGPTTaskVector(layer_shape)
        2. Accumulate Hessian: pruner.add_batch(inputs) for each batch
        3. Prune task vector: pruner.fasterprune(task_vector, density, ...)
    """

    def __init__(self, layer_shape: Tuple[int, int], device: torch.device = None):
        """
        Initialize SparseGPT for a layer.

        Args:
            layer_shape: (rows, columns) = (out_features, in_features)
            device: Device for computation (CPU/GPU)
        """
        self.device = device or torch.device("cpu")

        # Store weight matrix dimensions (matching sparsegpt.py naming)
        self.rows, self.columns = layer_shape  # rows=out_features, columns=in_features

        # Initialize Hessian matrix (input covariance accumulator)
        # H[i,j] will store correlation between input features i and j
        self.H = torch.zeros(
            (self.columns, self.columns), dtype=torch.float32, device=self.device
        )

        # Track total number of tokens seen (for running average)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate Hessian statistics from one batch of calibration data.

        Exactly matches sparsegpt.py::SparseGPT.add_batch() logic.

        Args:
            inp: Input activations [batch, seq_len, features] or [batch, features]
        """
        # Ensure input has batch dimension
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)  # [features] -> [1, features]

        # Number of samples in this batch (batch_size or batch*seq_len)
        tmp = inp.shape[0]

        # Reshape to [tokens, in_features] if 3D (matching sparsegpt.py logic)
        if len(inp.shape) == 3:
            # [batch, seq_len, features] -> [batch*seq_len, features]
            inp = inp.reshape((-1, inp.shape[-1]))

        # Transpose: [tokens, features] -> [features, tokens]
        inp = inp.t()

        # === Running average update for Hessian (exact sparsegpt.py formula) ===
        # Scale old Hessian by its weight in the new average
        self.H *= self.nsamples / (self.nsamples + tmp)
        # Update total token count
        self.nsamples += tmp

        # Normalize input (factor of sqrt(2/n) for numerical stability)
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        # Add covariance of this batch: H += X @ X^T
        # H[i,j] accumulates sum of (feature_i * feature_j) over all tokens
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        task_vector: torch.Tensor,
        density: float,
        blocksize: int = 128,
        percdamp: float = 0.01,
        rescale: bool = False,
    ) -> torch.Tensor:
        """
        Prune task vector using SparseGPT algorithm with error correction.

        Follows sparsegpt.py::SparseGPT.fasterprune() structure exactly.
        Key difference: Input is task_vector (delta) not full weights.

        Algorithm (matching sparsegpt.py):
        1. Compute inverse Hessian using Cholesky decomposition
        2. For each block of columns:
            a. Compute importance scores: wÂ²/(H_ii^-1)Â²
            b. For each column i:
                - Prune low-importance weights
                - Compute error: err = w_pruned / H_ii^(-1)
                - Propagate error to remaining columns: W[:,i+1:] -= err @ H^(-1)[i,i+1:]

        Args:
            task_vector: Task vector to prune [rows, columns]
            density: Fraction of weights to keep (0.2 = keep 20%)
            blocksize: Columns per block (trade-off: larger = faster but more memory)
            percdamp: Dampening factor (0.01 = 1% of avg diagonal)
            rescale: If True, scale by 1/density after pruning (DARE-style)

        Returns:
            Pruned task vector with error correction applied
        """
        # Validate inputs
        if len(task_vector.shape) != 2:
            raise ValueError(f"Expected 2D task vector, got shape {task_vector.shape}")

        rows, columns = task_vector.shape
        if rows != self.rows or columns != self.columns:
            raise ValueError(
                f"Task vector shape {task_vector.shape} doesn't match layer shape ({self.rows}, {self.columns})"
            )

        # Store original dtype and device for return
        original_dtype = task_vector.dtype
        original_device = task_vector.device

        # Work in float32 for numerical precision (matching sparsegpt.py)
        W = task_vector.clone().float().to(self.device)

        # Start timing (matching sparsegpt.py)
        tick = time.time()

        # Move Hessian to local variable (matching sparsegpt.py)
        H = self.H

        # === Handle dead neurons (features never activated) ===
        # Matches sparsegpt.py lines 131-133
        dead = torch.diag(H) == 0  # Diagonal = 0 means feature never varied
        H[dead, dead] = 1  # Set to 1 to avoid division by zero
        W[:, dead] = 0  # Zero out task vector for dead features

        # Track reconstruction losses per output neuron (matching sparsegpt.py)
        Losses = torch.zeros(rows, device=self.device)

        # === Add dampening to Hessian diagonal for numerical stability ===
        # Matches sparsegpt.py lines 137-140
        damp = percdamp * torch.mean(torch.diag(H))  # 1% of average diagonal
        diag = torch.arange(columns, device=self.device)
        H[diag, diag] += damp  # H_ii += damp for all i

        # === Compute inverse Hessian using Cholesky decomposition ===
        # Matches sparsegpt.py lines 142-148
        # Step 1: Cholesky decomposition H = L @ L^T
        H = torch.linalg.cholesky(H)
        # Step 2: Compute (L @ L^T)^{-1} = L^{-T} @ L^{-1}
        H = torch.cholesky_inverse(H)
        # Step 3: Cholesky of inverse (for numerical efficiency later)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H  # Hinv is now the Cholesky factor of H^{-1}

        # === Process weights in blocks (columns of W) ===
        # Matches sparsegpt.py lines 154-255
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)  # End of block
            count = i2 - i1  # Number of columns in this block

            # Extract block of task vector [rows, blocksize]
            W1 = W[:, i1:i2].clone()

            # Q1 will store pruned task vector for this block
            Q1 = torch.zeros_like(W1)
            # Err1 stores reconstruction errors for error propagation
            Err1 = torch.zeros_like(W1)
            # Losses1 tracks reconstruction loss per weight
            Losses1 = torch.zeros_like(W1)
            # Extract corresponding block of inverse Hessian
            Hinv1 = Hinv[i1:i2, i1:i2]

            # === Compute importance mask for this block ===
            # Matches sparsegpt.py unstructured pruning logic
            # Importance score: wÂ²/(H_ii^-1)Â²
            tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            # Find threshold for target density
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * (1 - density))]
            # Mask: True = prune, False = keep
            mask1 = tmp <= thresh

            # === Process each column in this block ===
            # Matches sparsegpt.py lines 190-235
            for i in range(count):
                w = W1[:, i]  # Current column of task vector [rows]
                d = Hinv1[i, i]  # Diagonal element of inverse Hessian

                # Prune weights based on mask
                q = w.clone()
                q[mask1[:, i]] = 0  # Zero out pruned weights

                # Store pruned weights
                Q1[:, i] = q
                # Compute reconstruction loss for this column
                Losses1[:, i] = (w - q) ** 2 / d**2

                # === Error propagation to remaining columns ===
                # Matches sparsegpt.py lines 229-235
                err1 = (w - q) / d  # Reconstruction error
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            # Write pruned block back to main task vector
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2  # Accumulate losses

            # Propagate errors to future blocks (matches sparsegpt.py lines 244-248)
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # Print timing and error (matching sparsegpt.py output)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Compute statistics for diagnostics
        total_params = task_vector.numel()
        pruned_params = (W == 0).sum().item()
        actual_sparsity = pruned_params / total_params
        reconstruction_error = torch.sum(Losses).item()

        print(f"  Pruning time: {time.time() - tick:.2f}s")
        print(f"  Reconstruction error: {reconstruction_error:.4f}")

        # Diagnostic info for zero reconstruction error
        if reconstruction_error < 1e-6:
            task_vector_norm = task_vector.norm().item()
            task_vector_sparsity = (task_vector.abs() < 1e-6).float().mean().item()
            print(
                f"    â†’ Task vector norm: {task_vector_norm:.4f} (small = layer barely changed)"
            )
            print(
                f"    â†’ Task vector sparsity: {task_vector_sparsity:.1%} (high = already sparse)"
            )
            if task_vector_norm < 0.01:
                print(
                    f"    â†’ This layer barely changed during fine-tuning (expected)"
                )

        # Optional rescaling (DARE-style)
        if rescale and density > 0:
            W /= density

        # Return in original dtype and device
        return W.to(dtype=original_dtype, device=original_device)

    def free(self):
        """Free memory used by Hessian (matches sparsegpt.py)."""
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# END OF SPARSEGPT TASK VECTOR IMPLEMENTATION
# ============================================================================
#
# Note: Orchestration logic (sequential processing, calibration data handling)
# has been moved to llama_sparse_merge.py following the pattern of sparsegpt/llama.py
#
# This file now contains ONLY the core pruning algorithm (SparseGPTTaskVector class)
# matching the structure of sparsegpt/sparsegpt.py
# ============================================================================
