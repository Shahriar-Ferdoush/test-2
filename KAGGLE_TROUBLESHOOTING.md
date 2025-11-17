# 🔧 Kaggle Troubleshooting Guide

## Error: "Can't find 'adapter_config.json' at '/kaggle/input/...'"

This error means you're trying to load a model from a **Kaggle dataset**, which doesn't work directly.

### ❌ WRONG (Kaggle Dataset Path):

```python
finetuned_model_paths=["/kaggle/input/llama3-2-1b-instruct-ft-doctor-consulting"]
```

### ✅ SOLUTION 1: Use HuggingFace Model ID (Recommended)

Upload your model to HuggingFace Hub first:

```python
# On your local machine or in a separate Kaggle notebook:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/path/to/your/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model")

# Upload to HuggingFace
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

Then use the HuggingFace ID:

```python
finetuned_model_paths=["your-username/model-name"]
```

### ✅ SOLUTION 2: Copy from Kaggle Dataset to Working Directory

```python
# In your Kaggle notebook, BEFORE running merger:

# Copy model from dataset to working directory
!cp -r /kaggle/input/llama3-2-1b-instruct-ft-doctor-consulting ./my_model

# Verify files were copied
!ls -la ./my_model

# Now use the local path
finetuned_model_paths=["./my_model"]
```

### ✅ SOLUTION 3: Mount Kaggle Dataset as Writable (Advanced)

```python
# Add the dataset in Kaggle notebook settings first
# Then copy to working directory:
import shutil
shutil.copytree(
    "/kaggle/input/llama3-2-1b-instruct-ft-doctor-consulting",
    "./my_model"
)
```

---

## Error: "Error no file named pytorch_model.bin, model.safetensors..."

Your model directory is missing the actual model weights.

### Check what files you have:

```python
import os
model_path = "/kaggle/input/your-dataset"  # or "./my_model"
print("Files in model directory:")
print(os.listdir(model_path))
```

### Required files for full model:

- `config.json` (required)
- `pytorch_model.bin` OR `model.safetensors` (model weights)
- `tokenizer_config.json` (for tokenizer)

### Required files for LoRA adapter:

- `adapter_config.json` (required)
- `adapter_model.bin` (LoRA weights)

---

## Error: "HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'"

You're using a file path when HuggingFace expects a model ID.

### ❌ WRONG:

```python
finetuned_model_paths=["/kaggle/input/model"]  # Local path
```

### ✅ CORRECT:

```python
finetuned_model_paths=["username/model-name"]  # HuggingFace ID
```

---

## Quick Diagnostic Script

Run this in a Kaggle cell to diagnose your model:

```python
import os

model_path = "/kaggle/input/your-dataset"  # UPDATE THIS

print("="*80)
print("MODEL DIAGNOSTIC")
print("="*80)

# Check if path exists
if os.path.exists(model_path):
    print(f"✓ Path exists: {model_path}")

    # List all files
    print(f"\nFiles in directory:")
    for f in os.listdir(model_path):
        size = os.path.getsize(os.path.join(model_path, f)) / (1024**2)
        print(f"  - {f} ({size:.2f} MB)")

    # Check for required files
    print(f"\nModel type detection:")
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("  ✓ LoRA adapter detected")
        print("  Required: adapter_config.json, adapter_model.bin")
    elif os.path.exists(os.path.join(model_path, "config.json")):
        has_weights = (
            os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
            os.path.exists(os.path.join(model_path, "model.safetensors"))
        )
        if has_weights:
            print("  ✓ Full model detected")
        else:
            print("  ✗ config.json found but missing weights!")
            print("  Need: pytorch_model.bin OR model.safetensors")
    else:
        print("  ✗ Not a valid model (missing config.json)")
else:
    print(f"✗ Path does not exist: {model_path}")

print("="*80)
```

---

## Working Example for Kaggle

```python
from llama_merge import LLaMAMerger
import torch

# Option 1: HuggingFace model ID (recommended)
merger = LLaMAMerger(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    finetuned_model_paths=[
        "your-username/model1",  # Must be on HuggingFace
        "your-username/model2",
    ],
    dataset_names=[
        "dataset1",
        "dataset2",
    ],
    output_dir="./merged_models",
    density=0.2,
    device="cuda"
)

# Option 2: Copy from Kaggle dataset first
# !cp -r /kaggle/input/your-dataset/model ./my_model
# merger = LLaMAMerger(
#     base_model_path="meta-llama/Llama-3.2-1B-Instruct",
#     finetuned_model_paths=["./my_model"],  # Local copy
#     ...
# )

results = merger.merge_all_methods()
```

---

## Prevention Checklist

Before running on Kaggle:

- [ ] Models are uploaded to HuggingFace Hub
- [ ] Using HuggingFace model IDs (format: "username/model-name")
- [ ] NOT using Kaggle dataset paths directly
- [ ] HF_TOKEN is added to Kaggle secrets
- [ ] Authenticated with HuggingFace (`login(token=...)`)

---

## Still Having Issues?

1. **Check your model format:**

   - Is it a full fine-tuned model or just LoRA adapter?
   - Does it have all required files?

2. **Verify HuggingFace authentication:**

   ```python
   from huggingface_hub import whoami
   print(whoami())  # Should show your username
   ```

3. **Test loading manually:**

   ```python
   from transformers import AutoModelForCausalLM

   # Try loading your model
   model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
   print("✓ Model loads successfully!")
   ```

4. **Check Kaggle logs carefully:**
   - The error message shows the exact file it's looking for
   - Verify that file exists in your model directory

---

## Need More Help?

- Review [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)
- Check [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
- Open an issue on GitHub with:
  - Full error message
  - Output from diagnostic script above
  - Your model structure (list of files)
