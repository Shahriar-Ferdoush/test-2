"""
Quick Merge Script for Llama Math Model (Low Memory)
=====================================================

This script merges your math fine-tuned model with other LoRA adapters
within the 29GB RAM and 19.5GB storage constraints.

Usage:
------
python merge_math_model.py --adapters adapter1 adapter2 adapter3

Or modify the ADAPTER_PATHS list below and run directly.
"""

import gc
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model (downloads from HuggingFace - will cache ~1.5GB for Llama 3.2 1B)
BASE_MODEL = "meta-llama/Llama-3.2-1B"

# Your LoRA adapters (add more as you train them)
ADAPTER_PATHS = [
    "./llama-3.2-1b-math",  # Your math adapter from the notebook
    # Add more adapters here:
    # "./llama-3.2-1b-code",
    # "./llama-3.2-1b-reasoning",
]

# Weights for each adapter (must sum to 1.0)
ADAPTER_WEIGHTS = [1.0]  # Single adapter gets full weight
# For multiple adapters: [0.4, 0.3, 0.3] etc.

# Output path
OUTPUT_PATH = "./merged_math_model"

# Use 8-bit quantization to save memory?
USE_8BIT = True  # Set to False if you have >16GB RAM

# ============================================================================
# MERGING FUNCTION
# ============================================================================


def merge_lora_adapters_sequential():
    """
    Merge LoRA adapters into base model using sequential merging.

    Memory efficient: Only loads one adapter at a time.
    """
    logger.info("=" * 60)
    logger.info("MERGING LORA ADAPTERS (LOW MEMORY MODE)")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Adapters: {len(ADAPTER_PATHS)}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"8-bit loading: {USE_8BIT}")
    logger.info("=" * 60)

    # Validate weights
    if len(ADAPTER_WEIGHTS) != len(ADAPTER_PATHS):
        raise ValueError(
            f"Number of weights ({len(ADAPTER_WEIGHTS)}) must match adapters ({len(ADAPTER_PATHS)})"
        )

    if abs(sum(ADAPTER_WEIGHTS) - 1.0) > 0.01:
        logger.warning(
            f"Weights sum to {sum(ADAPTER_WEIGHTS):.2f}, not 1.0. Normalizing..."
        )
        total = sum(ADAPTER_WEIGHTS)
        ADAPTER_WEIGHTS[:] = [w / total for w in ADAPTER_WEIGHTS]

    # Load base model
    logger.info("\n[1/3] Loading base model...")
    if USE_8BIT:
        # 8-bit quantization (uses ~1GB for Llama 3.2 1B)
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # Full precision
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    logger.info(f"✓ Base model loaded")
    log_memory()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Merge adapters one by one
    logger.info(f"\n[2/3] Merging {len(ADAPTER_PATHS)} adapters...")

    for idx, (adapter_path, weight) in enumerate(zip(ADAPTER_PATHS, ADAPTER_WEIGHTS)):
        logger.info(f"\n  [{idx+1}/{len(ADAPTER_PATHS)}] Processing: {adapter_path}")
        logger.info(f"    Weight: {weight:.3f}")

        try:
            # Load adapter
            logger.info(f"    Loading adapter...")
            model_with_adapter = PeftModel.from_pretrained(
                model,
                adapter_path,
            )

            log_memory()

            # Scale adapter if weight != 1.0
            if abs(weight - 1.0) > 0.01:
                logger.info(f"    Scaling adapter by {weight:.3f}...")
                # Scale LoRA A and B matrices
                for name, param in model_with_adapter.named_parameters():
                    if "lora" in name.lower() and param.requires_grad:
                        param.data *= weight

            # Merge adapter into base model
            logger.info(f"    Merging adapter into base model...")
            model = model_with_adapter.merge_and_unload()

            logger.info(f"    ✓ Adapter {idx+1} merged")
            log_memory()

            # Cleanup
            del model_with_adapter
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"    ✗ Error merging adapter {adapter_path}: {e}")
            logger.error(f"    Skipping this adapter...")
            continue

    logger.info("\n✓ All adapters merged!")

    # Save merged model
    logger.info(f"\n[3/3] Saving merged model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    logger.info(f"✓ Model saved!")

    # Final memory report
    logger.info("\nFinal memory usage:")
    log_memory()

    return model, tokenizer


def log_memory():
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"    GPU: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved"
        )

    try:
        import psutil

        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1e9
        logger.info(f"    RAM: {ram_gb:.2f}GB")
    except:
        pass


# ============================================================================
# TESTING FUNCTION
# ============================================================================


def test_merged_model(model, tokenizer):
    """Test the merged model with sample math problems"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MERGED MODEL")
    logger.info("=" * 60)

    test_questions = [
        "What is 2 + 2?",
        "Solve for x: 3x + 7 = 22",
        "What is the square root of 144?",
    ]

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\nTest {i}: {question}")
        logger.info("-" * 40)

        # Prepare input
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        logger.info(f"Response: {response[:200]}...")

    logger.info("\n✓ Testing complete!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapters for math model")
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="+",
        help="Paths to LoRA adapters (overrides ADAPTER_PATHS in script)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for each adapter (must sum to 1.0)",
    )
    parser.add_argument("--output", type=str, help="Output path for merged model")
    parser.add_argument(
        "--test", action="store_true", help="Test the merged model after merging"
    )
    parser.add_argument(
        "--no-8bit", action="store_true", help="Disable 8-bit quantization"
    )

    args = parser.parse_args()

    # Override config from command line
    if args.adapters:
        ADAPTER_PATHS[:] = args.adapters

    if args.weights:
        ADAPTER_WEIGHTS[:] = args.weights

    if args.output:
        OUTPUT_PATH = args.output

    if args.no_8bit:
        USE_8BIT = False

    # Merge adapters
    try:
        model, tokenizer = merge_lora_adapters_sequential()

        # Test if requested
        if args.test:
            test_merged_model(model, tokenizer)

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL DONE!")
        logger.info("=" * 60)
        logger.info(f"Merged model saved to: {OUTPUT_PATH}")
        logger.info("You can now load it with:")
        logger.info(f'  model = AutoModelForCausalLM.from_pretrained("{OUTPUT_PATH}")')
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
