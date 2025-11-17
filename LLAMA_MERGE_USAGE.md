# LLaMA Model Merging - Usage Examples

This guide shows how to use `llama_merge.py` to merge fine-tuned LLaMA models using three different methods.

## 📋 Quick Start

### Example 1: Basic Usage

```bash
python llama_merge.py \
    --base_model "meta-llama/Llama-3.2-1B-Instruct" \
    --finetuned_models "model1" "model2" \
    --datasets "dataset1" "dataset2" \
    --output_dir "./merged_models" \
    --density 0.2
```

### Example 2: Your Mental Health Counselor Model

```bash
python llama_merge.py \
    --base_model "meta-llama/Llama-3.2-1B-Instruct" \
    --finetuned_models "llama-3.2-1b-mental-health-counselor" "llama-3.2-1b-doctor" \
    --datasets "Amod/mental_health_counseling_conversations" "medical-dataset" \
    --output_dir "./merged_mental_health" \
    --density 0.2 \
    --num_calibration_samples 128 \
    --device cuda
```

### Example 3: Merging Multiple Tasks

```bash
python llama_merge.py \
    --base_model "meta-llama/Llama-3.2-1B" \
    --finetuned_models \
        "task1-model" \
        "task2-model" \
        "task3-model" \
    --datasets \
        "task1-dataset" \
        "task2-dataset" \
        "task3-dataset" \
    --output_dir "./multi_task_merged" \
    --density 0.3 \
    --cache_dir "./cache"
```

## 🔧 Configuration Options

| Parameter                   | Description                           | Default           |
| --------------------------- | ------------------------------------- | ----------------- |
| `--base_model`              | Path to base/pretrained model         | **Required**      |
| `--finetuned_models`        | List of fine-tuned model paths        | **Required**      |
| `--datasets`                | List of dataset names for calibration | **Required**      |
| `--output_dir`              | Where to save merged models           | `./merged_models` |
| `--cache_dir`               | Where to store intermediate files     | `./merge_cache`   |
| `--density`                 | Fraction of weights to keep (0-1)     | `0.2`             |
| `--num_calibration_samples` | Samples for Hessian computation       | `128`             |
| `--device`                  | Device to use (cuda/cpu)              | Auto-detect       |

## 📊 What Gets Created

After running, you'll have:

```
merged_models/
├── ties_magnitude/          # Simple TIES (magnitude-based)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
│
├── dare_random/             # Simple DARE (random dropout)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
│
├── ties_sparsegpt/          # TIES with SparseGPT importance
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
│
└── comparison_results.json  # Metrics comparison

merge_cache/
├── task_vectors/
│   ├── task_vector_0.pt
│   ├── task_vector_1.pt
│   └── ...
└── hessians/
    ├── hessian_0.pt
    ├── hessian_1.pt
    └── ...
```

## 🎯 Understanding the Methods

### Method 1: TIES with Magnitude Trimming

- **Speed:** Fast ⚡
- **Memory:** Low
- **Accuracy:** Good
- **How:** Keeps top-k weights by absolute value

### Method 2: DARE with Random Dropout

- **Speed:** Fastest ⚡⚡
- **Memory:** Low
- **Accuracy:** Good
- **How:** Randomly drops weights, rescales remaining

### Method 3: TIES with SparseGPT Importance

- **Speed:** Slower 🐌
- **Memory:** Higher (needs Hessians)
- **Accuracy:** Best 🏆
- **How:** Keeps top-k by importance score w²/(H⁻¹)²

## 💾 Memory Management

The script uses a **layer-by-layer** approach to avoid OOM:

1. **Task Vectors:** Computed one model at a time, saved to disk
2. **Hessians:** Computed one model at a time, saved to disk
3. **Merging:** Loads one layer at a time from cache

This allows merging **multiple 7B+ models** on GPUs with limited VRAM!

## 📈 Interpreting Results

The script will print a comparison table:

```
Method               Perplexity     Avg Loss      Time (s)
------------------------------------------------------------
TIES-Magnitude            12.45       2.5210         120.5
DARE-Random               13.21       2.5820          85.3
TIES-SparseGPT           11.87       2.4750         245.8
```

**Lower perplexity = Better!**

## 🚀 Programmatic Usage

```python
from llama_merge import LLaMAMerger

# Create merger
merger = LLaMAMerger(
    base_model_path="meta-llama/Llama-3.2-1B",
    finetuned_model_paths=["model1", "model2"],
    dataset_names=["dataset1", "dataset2"],
    output_dir="./merged",
    density=0.2
)

# Run all methods
results = merger.merge_all_methods()

# Or run individual methods
ties_model = merger.merge_with_ties(use_sparsegpt=False)
dare_model = merger.merge_with_dare(use_sparsegpt=False)
sparsegpt_model = merger.merge_with_ties(use_sparsegpt=True)
```

## 🔍 Advanced: Custom Evaluation

```python
from llama_merge import LLaMAMerger
from transformers import AutoModelForCausalLM, AutoTokenizer

merger = LLaMAMerger(...)

# Merge with TIES-SparseGPT
model = merger.merge_with_ties(use_sparsegpt=True)

# Custom evaluation
tokenizer = AutoTokenizer.from_pretrained(merger.base_model_path)
metrics = merger.evaluate_model(
    model,
    tokenizer,
    eval_dataset="your_custom_dataset",
    num_samples=200
)

print(f"Perplexity: {metrics['perplexity']}")
```

## ⚠️ Troubleshooting

### Out of Memory?

- Reduce `--num_calibration_samples` (e.g., to 64)
- Use CPU for Hessian computation (slower but safer)
- The script already uses layer-by-layer processing

### Dataset Loading Fails?

- The script will create dummy calibration data as fallback
- Check dataset name is correct on HuggingFace
- Ensure dataset has a text field

### Model Loading Fails?

- For LoRA models: Ensure base_model path is correct
- For full models: Check path/HuggingFace ID
- Script supports both formats automatically

### Keys Don't Match?

- The script fuzzy-matches layer names automatically
- Check logs for warnings about missing keys

## 📝 Notes

1. **First run is slow** - Task vectors and Hessians are computed
2. **Subsequent runs are fast** - Cached data is reused
3. **Delete cache** to recompute: `rm -rf merge_cache/`
4. **SparseGPT is worth it** - Usually 5-10% better perplexity

## 🎓 Citation

If you use this in research, cite:

```bibtex
@article{frantar2023sparsegpt,
  title={SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot},
  author={Frantar, Elias and Alistarh, Dan},
  journal={arXiv preprint arXiv:2301.00774},
  year={2023}
}

@article{yadav2023ties,
  title={TIES-Merging: Resolving Interference When Merging Models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Raffel, Colin and Bansal, Mohit},
  journal={arXiv preprint arXiv:2306.01708},
  year={2023}
}
```

## 🤝 Support

Found a bug? Open an issue!
Have questions? Check `ANALYSIS_AND_VERIFICATION.md` for detailed algorithm explanations.
