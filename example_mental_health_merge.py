"""
Example: Merging Mental Health Counselor Models

This is a simple example showing how to merge your mental health counselor
fine-tuned models using the three merging methods.

Customize the paths and parameters below for your specific setup.
"""

import torch
from llama_merge import LLaMAMerger

# ============================================================================
# CONFIGURATION - Modify these for your setup
# ============================================================================

# Base model configuration
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Fine-tuned models (can be local paths or HuggingFace IDs)
FINETUNED_MODELS = [
    "llama-3.2-1b-mental-health-counselor",  # Your mental health model
    # Add more fine-tuned models here if you have them
    # "llama-3.2-1b-therapist",
    # "llama-3.2-1b-wellness-coach",
]

# Datasets for calibration (one per fine-tuned model)
DATASETS = [
    "Amod/mental_health_counseling_conversations",
    # Add corresponding datasets here
    # "therapist-conversations-dataset",
    # "wellness-coaching-dataset",
]

# Output directory
OUTPUT_DIR = "./merged_mental_health_models"

# Cache directory (stores intermediate files)
CACHE_DIR = "./merge_cache"

# Merging parameters
DENSITY = 0.2  # Keep top 20% of weights
NUM_CALIBRATION_SAMPLES = 128  # Samples for Hessian computation
CALIBRATION_SEQ_LENGTH = 512  # Sequence length

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    print("=" * 60)
    print("Mental Health Counselor Model Merging")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL}")
    print(f"Fine-tuned Models: {len(FINETUNED_MODELS)}")
    for i, model in enumerate(FINETUNED_MODELS):
        print(f"  [{i+1}] {model}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Validate configuration
    if len(FINETUNED_MODELS) != len(DATASETS):
        raise ValueError(
            f"Number of models ({len(FINETUNED_MODELS)}) must match "
            f"number of datasets ({len(DATASETS)})"
        )

    # Create merger
    merger = LLaMAMerger(
        base_model_path=BASE_MODEL,
        finetuned_model_paths=FINETUNED_MODELS,
        dataset_names=DATASETS,
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        calibration_seq_length=CALIBRATION_SEQ_LENGTH,
        density=DENSITY,
        device=DEVICE,
    )

    # Option 1: Run all methods and compare (RECOMMENDED)
    print("\n🚀 Running all three merging methods...")
    results = merger.merge_all_methods()

    # Results are automatically saved to:
    # - {OUTPUT_DIR}/ties_magnitude/
    # - {OUTPUT_DIR}/dare_random/
    # - {OUTPUT_DIR}/ties_sparsegpt/
    # - {OUTPUT_DIR}/comparison_results.json

    print("\n✓ Done! Check the output directory for merged models.")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Cache: {CACHE_DIR}")

    # Option 2: Run individual methods (if you want just one)
    # Uncomment the method you want:

    # # TIES with magnitude trimming (fast, good)
    # print("\n🚀 Running TIES with magnitude trimming...")
    # ties_model = merger.merge_with_ties(use_sparsegpt=False)
    # ties_model.save_pretrained(f"{OUTPUT_DIR}/ties_magnitude")

    # # DARE with random dropout (fastest, good)
    # print("\n🚀 Running DARE with random dropout...")
    # dare_model = merger.merge_with_dare(use_sparsegpt=False)
    # dare_model.save_pretrained(f"{OUTPUT_DIR}/dare_random")

    # # TIES with SparseGPT importance (slower, best)
    # print("\n🚀 Running TIES with SparseGPT importance...")
    # sparsegpt_model = merger.merge_with_ties(use_sparsegpt=True)
    # sparsegpt_model.save_pretrained(f"{OUTPUT_DIR}/ties_sparsegpt")


def test_merged_model():
    """
    Quick test of the best merged model.
    Run this after merging is complete.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the best model (usually TIES-SparseGPT)
    model_path = f"{OUTPUT_DIR}/ties_sparsegpt"

    print(f"\n🧪 Testing merged model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Test prompt
    prompt = "I'm feeling anxious about my job interview tomorrow. What should I do?"

    print(f"\n📝 Prompt: {prompt}")
    print("\n🤖 Response:")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    # Run merging
    main()

    # Optionally test the merged model
    # Uncomment to test:
    # test_merged_model()
