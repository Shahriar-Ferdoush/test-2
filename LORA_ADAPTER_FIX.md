# 🔧 LoRA Adapter Merging - Complete Fix Guide

## 🎯 The Problem

You fine-tuned LLaMA with LoRA (using PEFT), which creates **adapter files only**, not a full model. When merging, the code needs to:
1. Load the base model
2. Load the LoRA adapter
3. Merge them together
4. THEN compute task vectors

## ✅ What Was Fixed

### Issue 1: Model Type Detection
**Before:** Code tried to load as full model first, then LoRA
**After:** Code detects LoRA adapters by checking for `adapter_config.json`

### Issue 2: Kaggle Base Model Path
**Before:** Used default HuggingFace path
**After:** Use Kaggle-specific path: `/kaggle/input/llama-3.2/transformers/1b-instruct/1`

### Issue 3: File Structure Understanding
**Before:** Expected `config.json` (full model)
**After:** Recognizes `adapter_config.json` (LoRA adapter)

## 📁 LoRA Adapter Files

Your fine-tuned models have this structure:
```
llama-3-1b-medical-chatbot-v1/
├── adapter_config.json      ← LoRA configuration
├── adapter_model.safetensors ← LoRA weights
└── README.md
```

NOT a full model (which would have):
```
full-model/
├── config.json              ← Model architecture
├── pytorch_model.bin        ← Full model weights
├── tokenizer files...
└── etc.
```

## 🔍 How to Verify Your Model Type

Run this in Kaggle:

```python
import os

model_path = "/kaggle/input/your-dataset"  # Your dataset path

files = os.listdir(model_path)
print("Files in model directory:")
for f in files:
    print(f"  - {f}")

print("\nModel type:")
if "adapter_config.json" in files:
    print("  → LoRA Adapter (needs base model)")
elif "config.json" in files:
    print("  → Full Model")
else:
    print("  → Unknown/Invalid")
```

## 🚀 Complete Kaggle Setup

### Cell 1: Setup
```python
!git clone https://github.com/Shahriar-Ferdoush/test-2.git
%cd test-2
!python kaggle_setup.py
```

### Cell 2: Copy LoRA Adapters
```python
import shutil
import os

# Your Kaggle datasets (LoRA adapters)
kaggle_datasets = [
    "/kaggle/input/llama3-2-1b-instruct-ft-doctor-consulting",
    "/kaggle/input/llama3-2-1b-instruct-fine-tuning-therapist-ai",
]

local_models = []

for i, dataset_path in enumerate(kaggle_datasets):
    if os.path.exists(dataset_path):
        local_path = f"./lora_adapter_{i+1}"
        print(f"Copying {dataset_path} → {local_path}")
        shutil.copytree(dataset_path, local_path)
        
        # Verify it's a LoRA adapter
        files = os.listdir(local_path)
        if "adapter_config.json" in files:
            print(f"  ✓ Valid LoRA adapter")
            local_models.append(local_path)
        else:
            print(f"  ⚠️ Not a LoRA adapter!")

print(f"\n✓ Ready: {len(local_models)} LoRA adapters")
```

### Cell 3: Configure Merger
```python
import torch
from llama_merge import LLaMAMerger

# CRITICAL: Use Kaggle's base model path
BASE_MODEL = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"

# Your LoRA adapters (copied above)
FINETUNED_MODELS = local_models

# Calibration datasets
DATASETS = [
    "Amod/mental_health_counseling_conversations",
    "ruslanmv/ai-medical-chatbot",
]

merger = LLaMAMerger(
    base_model_path=BASE_MODEL,  # ← Kaggle base model
    finetuned_model_paths=FINETUNED_MODELS,  # ← LoRA adapters
    dataset_names=DATASETS,
    output_dir="./merged_models",
    cache_dir="./merge_cache",
    density=0.2,
    num_calibration_samples=64,  # Reduced for Kaggle
    device="cuda"
)
```

### Cell 4: Run Merging
```python
results = merger.merge_all_methods()
print(results)
```

## 🔧 Code Changes Made

### In `llama_merge.py`:

**Added LoRA detection:**
```python
# Check if it's a LoRA adapter
if "adapter_config.json" in files_in_path:
    is_lora = True
    logger.info("  Detected LoRA adapter format")
```

**Load LoRA adapters correctly:**
```python
if is_lora:
    # Load base model first
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,  # ← Uses YOUR base_model_path
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, ft_model_path)
    
    # Merge LoRA into base
    model = model.merge_and_unload()
```

## 🎯 Key Points

1. **LoRA ≠ Full Model**
   - LoRA = Small adapter files (~50-100MB)
   - Full Model = Complete model weights (~2-4GB)

2. **Base Model is Required**
   - LoRA adapters MUST be loaded on top of base model
   - Use: `/kaggle/input/llama-3.2/transformers/1b-instruct/1`

3. **Merging Process**
   ```
   Base Model + LoRA Adapter → Merged Model → Task Vector → Merge
   ```

4. **File Locations**
   - Base: `/kaggle/input/llama-3.2/transformers/1b-instruct/1`
   - Adapters: `/kaggle/input/your-dataset/` (copy to working dir)
   - Output: `/kaggle/working/merged_models/`

## 🧪 Testing

### Quick Test (Cell in notebook):
```python
# Test LoRA loading
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/llama-3.2/transformers/1b-instruct/1",
    torch_dtype=torch.float16,
    device_map="cpu"
)
print("✓ Base loaded")

model = PeftModel.from_pretrained(base, "./lora_adapter_1")
print("✓ LoRA loaded")

model = model.merge_and_unload()
print("✓ Merged!")
```

## 📊 Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Copy adapters | 1 min | ~100MB each |
| Verify setup | 2 min | Test loading |
| Task vectors | 10 min | Base + adapter merging |
| Hessians | 15 min | Calibration data |
| Merging | 10 min | All three methods |
| **Total** | **~40 min** | For 2 LoRA adapters |

## ⚠️ Common Issues

### Error: "Can't find 'adapter_config.json'"
- **Cause**: Path is wrong or files didn't copy
- **Fix**: Check with `!ls -la ./lora_adapter_1`

### Error: "Unrecognized model in ./finetuned_model_1"
- **Cause**: Missing `config.json` (trying to load as full model)
- **Fix**: Code now detects LoRA first - update llama_merge.py

### Error: Base model not found
- **Cause**: Wrong base model path
- **Fix**: Use `/kaggle/input/llama-3.2/transformers/1b-instruct/1`

## 📝 Summary

| Component | Path |
|-----------|------|
| Base Model | `/kaggle/input/llama-3.2/transformers/1b-instruct/1` |
| LoRA Adapter 1 | `./lora_adapter_1` (copied from `/kaggle/input/...`) |
| LoRA Adapter 2 | `./lora_adapter_2` (copied from `/kaggle/input/...`) |
| Merged Output | `./merged_models/ties_sparsegpt_merged/` |
| Cache | `./merge_cache/` |

**Workflow:**
1. Copy LoRA adapters from `/kaggle/input/` to working directory
2. Configure merger with correct base model path
3. Code detects LoRA adapters automatically
4. Merges base + LoRA before computing task vectors
5. Proceeds with normal merging

**You're now ready to run! 🚀**
