"""
LLaMA Model Merging Orchestration

This script orchestrates model merging using three independent methods:
1. TIES merging (llama_ties_merge.py)
2. DARE merging (llama_dare_merge.py)
3. SparseGPT merging (llama_sparse_merge.py)

This is the main entry point that runs all methods and generates comparison statistics.

Usage:
-----
python llama_merge.py --config config.yaml

Or programmatically:
    merger = LLaMAMerger(
        base_model_path="meta-llama/Llama-3.2-1B",
        finetuned_model_paths=["model1", "model2"],
        dataset_names=["dataset1", "dataset2"],
        output_dir="./merged_models"
    )
    merger.merge_all_methods()
"""

import argparse
import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_dare_merge import llama_dare_merge

# Import the three independent merge implementations
from llama_sparse_merge import llama_sparse_merge_sequential, load_calibration_data
from llama_ties_merge import llama_ties_merge

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLaMAMerger:
    """
    Orchestrator for LLaMA model merging using three independent methods:
    - TIES (magnitude-based trimming)
    - DARE (random dropout)
    - SparseGPT (Hessian-based importance)
    """

    def __init__(
        self,
        base_model_path: str,
        finetuned_model_paths: List[str],
        dataset_names: List[str],
        output_dir: str = "./merged_models",
        num_calibration_samples: int = 128,
        density: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize LLaMA merger.

        Args:
            base_model_path: Path to base (pretrained) model
            finetuned_model_paths: List of paths to fine-tuned models
            dataset_names: List of dataset names for calibration (for SparseGPT)
            output_dir: Directory to save merged models
            num_calibration_samples: Number of samples for Hessian computation
            density: Fraction of weights to keep (0.2 = 20%)
            device: Device to use ('cuda' or 'cpu')
        """
        self.base_model_path = base_model_path
        self.finetuned_model_paths = finetuned_model_paths
        self.dataset_names = dataset_names
        self.output_dir = Path(output_dir)
        self.num_calibration_samples = num_calibration_samples
        self.density = density
        self.device = torch.device(device)

        # Validate inputs
        if len(finetuned_model_paths) != len(dataset_names):
            raise ValueError(
                f"Number of models ({len(finetuned_model_paths)}) must match "
                f"number of datasets ({len(dataset_names)})"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized LLaMAMerger with {len(finetuned_model_paths)} models")
        logger.info(f"Device: {self.device}")
        logger.info(f"Density: {self.density} (keep {self.density*100}% of weights)")

    def merge_all_methods(self) -> Dict[str, Dict]:
        """
        Run all three merging methods and compare results.

        Returns:
            Dictionary with results for each method
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL MERGING COMPARISON")
        print("=" * 80)
        print(f"Base model: {self.base_model_path}")
        print(f"Fine-tuned models: {len(self.finetuned_model_paths)}")
        print(f"Density: {self.density:.1%}")
        print(f"Device: {self.device}")
        print("=" * 80)

        results = {}

        # Load tokenizer once
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ========================================================================
        # Method 1: TIES Merging
        # ========================================================================
        print("\n" + "=" * 80)
        print("METHOD 1: TIES (Magnitude-Based Trimming)")
        print("=" * 80)

        try:
            start_time = time.time()

            # Load models
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16
            )
            finetuned_models = []
            for ft_path in self.finetuned_model_paths:
                ft_model = AutoModelForCausalLM.from_pretrained(
                    ft_path, torch_dtype=torch.float16
                )
                finetuned_models.append(ft_model)

            # Merge
            merged_ties = llama_ties_merge(
                base_model=base_model,
                finetuned_models=finetuned_models,
                density=self.density,
                device=str(self.device),
            )

            elapsed = time.time() - start_time

            # Save
            output_path = self.output_dir / "ties_magnitude"
            merged_ties.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Evaluate
            perplexity = self.evaluate_model(merged_ties, tokenizer)

            results["TIES-Magnitude"] = {
                "perplexity": perplexity,
                "time": elapsed,
                "path": str(output_path),
            }

            print(f"âœ“ TIES merge complete: PPL={perplexity:.2f}, Time={elapsed:.1f}s")

            # Cleanup
            del base_model, finetuned_models, merged_ties
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"TIES merging failed: {e}")
            results["TIES-Magnitude"] = {"error": str(e)}

        # ========================================================================
        # Method 2: DARE Merging
        # ========================================================================
        print("\n" + "=" * 80)
        print("METHOD 2: DARE (Random Dropout + Rescale)")
        print("=" * 80)

        try:
            start_time = time.time()

            # Load models
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16
            )
            finetuned_models = []
            for ft_path in self.finetuned_model_paths:
                ft_model = AutoModelForCausalLM.from_pretrained(
                    ft_path, torch_dtype=torch.float16
                )
                finetuned_models.append(ft_model)

            # Merge
            merged_dare = llama_dare_merge(
                base_model=base_model,
                finetuned_models=finetuned_models,
                density=self.density,
                device=str(self.device),
            )

            elapsed = time.time() - start_time

            # Save
            output_path = self.output_dir / "dare_random"
            merged_dare.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Evaluate
            perplexity = self.evaluate_model(merged_dare, tokenizer)

            results["DARE-Random"] = {
                "perplexity": perplexity,
                "time": elapsed,
                "path": str(output_path),
            }

            print(f"âœ“ DARE merge complete: PPL={perplexity:.2f}, Time={elapsed:.1f}s")

            # Cleanup
            del base_model, finetuned_models, merged_dare
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"DARE merging failed: {e}")
            results["DARE-Random"] = {"error": str(e)}

        # ========================================================================
        # Method 3: SparseGPT Merging
        # ========================================================================
        print("\n" + "=" * 80)
        print("METHOD 3: SparseGPT (Hessian-Based Importance)")
        print("=" * 80)

        try:
            start_time = time.time()

            # Load models
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16
            )
            finetuned_models = []
            for ft_path in self.finetuned_model_paths:
                ft_model = AutoModelForCausalLM.from_pretrained(
                    ft_path, torch_dtype=torch.float16
                )
                finetuned_models.append(ft_model)

            # Load calibration data (use first dataset)
            calibration_data = load_calibration_data(
                dataset_name=self.dataset_names[0],
                tokenizer=tokenizer,
                num_samples=self.num_calibration_samples,
            )

            # Merge
            merged_sparsegpt = llama_sparse_merge_sequential(
                base_model=base_model,
                finetuned_models=finetuned_models,
                calibration_data=calibration_data,
                density=self.density,
                device=str(self.device),
            )

            elapsed = time.time() - start_time

            # Save
            output_path = self.output_dir / "ties_sparsegpt"
            merged_sparsegpt.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Evaluate
            perplexity = self.evaluate_model(merged_sparsegpt, tokenizer)

            results["TIES-SparseGPT"] = {
                "perplexity": perplexity,
                "time": elapsed,
                "path": str(output_path),
            }

            print(
                f"âœ“ SparseGPT merge complete: PPL={perplexity:.2f}, Time={elapsed:.1f}s"
            )

            # Cleanup
            del base_model, finetuned_models, merged_sparsegpt, calibration_data
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"SparseGPT merging failed: {e}")
            results["TIES-SparseGPT"] = {"error": str(e)}

        # ========================================================================
        # Print Comparison
        # ========================================================================
        self._print_comparison(results)

        return results

    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        eval_dataset: str = "wikitext",
        num_samples: int = 100,
    ) -> float:
        """
        Evaluate perplexity of merged model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            eval_dataset: Dataset name
            num_samples: Number of evaluation samples

        Returns:
            Perplexity score
        """
        logger.info(f"Evaluating model on {eval_dataset}...")

        try:
            # Load evaluation dataset
            if eval_dataset == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                text_field = "text"
            else:
                dataset = load_dataset(eval_dataset, split="test")
                text_field = "text"

            model = model.to(self.device)
            model.eval()

            total_loss = 0.0
            total_tokens = 0

            with torch.no_grad():
                for i, example in enumerate(dataset):
                    if i >= num_samples:
                        break

                    text = example[text_field]
                    if not text or len(text.strip()) < 10:
                        continue

                    # Tokenize
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    ).to(self.device)

                    # Forward pass
                    outputs = model(**inputs, labels=inputs.input_ids)
                    loss = outputs.loss

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item() * inputs.input_ids.numel()
                        total_tokens += inputs.input_ids.numel()

            # Calculate perplexity
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            return perplexity

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return float("inf")

    def _print_comparison(self, results: Dict[str, Dict]):
        """Print formatted comparison of all methods."""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        # Print table header
        print(f"\n{'Method':<20} {'Perplexity':<15} {'Time (s)':<10}")
        print("-" * 60)

        best_ppl = float("inf")
        best_method = None

        for method, result in results.items():
            if "error" in result:
                print(f"{method:<20} {'ERROR':<15} {'-':<10}")
            else:
                ppl = result["perplexity"]
                time_taken = result["time"]
                print(f"{method:<20} {ppl:<15.4f} {time_taken:<10.1f}")

                if ppl < best_ppl:
                    best_ppl = ppl
                    best_method = method

        print("\n" + "=" * 80)
        if best_method:
            print(f"ðŸ† Best method: {best_method}")
            print(f"   Perplexity: {best_ppl:.4f}")
        print("=" * 80 + "\n")


# ============================================================================
# Command-Line Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Merge LLaMA models using TIES, DARE, and SparseGPT methods"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of base model",
    )

    parser.add_argument(
        "--finetuned_models",
        type=str,
        nargs="+",
        required=True,
        help="Paths or HuggingFace IDs of fine-tuned models",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names for calibration (one per model)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_models",
        help="Directory to save merged models",
    )

    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="Fraction of weights to keep (default: 0.2)",
    )

    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.finetuned_models) != len(args.datasets):
        raise ValueError(
            f"Number of models ({len(args.finetuned_models)}) must match "
            f"number of datasets ({len(args.datasets)})"
        )

    # Create merger
    merger = LLaMAMerger(
        base_model_path=args.base_model,
        finetuned_model_paths=args.finetuned_models,
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        num_calibration_samples=args.num_calibration_samples,
        density=args.density,
        device=args.device,
    )

    # Run all methods and compare
    results = merger.merge_all_methods()

    logger.info("\nâœ“ ALL DONE!")


if __name__ == "__main__":
    main()
