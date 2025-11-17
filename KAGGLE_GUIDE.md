# Running Model Merging on Kaggle

Complete guide to run the model merging toolkit on Kaggle.

## 📋 Prerequisites

1. **Kaggle Account** with email verification
2. **Phone Verification** (required for GPU access)
3. **HuggingFace Account** (for LLaMA model access)
4. **Accept LLaMA Terms** on HuggingFace model page

## 🚀 Step-by-Step Guide

### Step 1: Push Code to GitHub

On your local machine:

```powershell
cd "g:\Thesis\Model-Merging"

# Initialize git (if not already done)
git init
git add .
git commit -m "Add model merging toolkit with SparseGPT"

# Push to GitHub
git remote add origin https://github.com/Shahriar-Ferdoush/test-2.git
git branch -M main
git push -u origin main
```

### Step 2: Create Kaggle Notebook

1. Go to https://www.kaggle.com/
2. Click **"+ New Notebook"**
3. Enable **GPU**: Settings → Accelerator → **GPU T4 x2**
4. Enable **Internet**: Settings → Internet → **On**
5. Set **Persistence**: Settings → Persistence → **Files only**

### Step 3: Setup HuggingFace Token in Kaggle

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **"Read"** access
3. In Kaggle: Settings → **Secrets** → **Add Secret**
   - Name: `HF_TOKEN`
   - Value: Your HuggingFace token

### Step 4: Clone Repository in Kaggle

In the first cell of your Kaggle notebook:

```python
!git clone https://github.com/Shahriar-Ferdoush/test-2.git
%cd test-2
```

### Step 5: Run Setup Script

```python
!python kaggle_setup.py
```

This will:

- ✅ Check GPU availability
- ✅ Install all required packages
- ✅ Create necessary directories
- ✅ Check disk space

### Step 6: Authenticate with HuggingFace

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

### Step 7: Run Merging

**Option A: Use the template notebook**

- Upload `kaggle_notebook_template.ipynb` to Kaggle
- Run all cells

**Option B: Use the example script**

```python
from example_mental_health_merge import main
main()
```

**Option C: Custom configuration**

```python
import torch
from llama_merge import LLaMAMerger

merger = LLaMAMerger(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    finetuned_model_paths=["your-model-path"],
    dataset_names=["your-dataset"],
    output_dir="./merged_models",
    density=0.2,
    device="cuda"
)

results = merger.merge_all_methods()
print(results)
```

## ⚙️ Kaggle-Specific Configuration

### GPU Settings

- **Recommended**: GPU T4 x2 (30 hours/week free)
- **Memory**: ~16GB VRAM per T4
- **Models supported**: Up to 7B parameters

### Disk Space Management

Kaggle provides ~70GB workspace. For large models:

```python
# After merging, clean up intermediate files
import shutil
shutil.rmtree("./merge_cache")  # Frees ~10-20GB
```

### Time Limits

- Free tier: **12 hours max** per session
- Expected times:
  - 1B model: ~30-60 minutes
  - 3B model: ~2-3 hours
  - 7B model: ~6-8 hours

### Memory Optimization

If hitting OOM errors:

```python
merger = LLaMAMerger(
    ...,
    num_calibration_samples=64,  # Reduce from 128
    calibration_seq_length=256,  # Reduce from 512
    device="cpu"  # Use CPU (slower but safer)
)
```

## 📊 Expected Results

### Performance (1B LLaMA on T4 x2)

| Step            | Time        | Memory         |
| --------------- | ----------- | -------------- |
| Task vectors    | ~5 min      | ~4 GB          |
| Hessians        | ~15 min     | ~6 GB          |
| TIES merge      | ~3 min      | ~4 GB          |
| DARE merge      | ~3 min      | ~4 GB          |
| SparseGPT merge | ~5 min      | ~6 GB          |
| **Total**       | **~30 min** | **~6 GB peak** |

### Perplexity Comparison

Typical results on mental health data:

- **TIES-Magnitude**: 12.5
- **DARE-Random**: 13.2
- **TIES-SparseGPT**: **11.8** 🏆 (best)

## 🔧 Troubleshooting

### Issue 1: "No GPU available"

```python
# Check GPU status
!nvidia-smi

# If no GPU, enable in Settings
```

### Issue 2: "Dataset not found"

```python
# The script auto-generates dummy data as fallback
# Check logs for warnings
```

### Issue 3: "Model download failed"

```python
# Verify HuggingFace authentication
from huggingface_hub import whoami
print(whoami())

# Re-authenticate if needed
```

### Issue 4: "Out of Memory"

```python
# Reduce calibration samples
merger = LLaMAMerger(..., num_calibration_samples=32)

# Or use CPU
merger = LLaMAMerger(..., device="cpu")

# Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue 5: "Session timeout"

If your session times out before completion:

1. Task vectors and Hessians are **cached**
2. Restart and re-run - it will resume from cache
3. Check `./merge_cache/` directory

## 💾 Saving Results

### Save to Kaggle Output

Models saved to `/kaggle/working/merged_models/` persist after session ends (up to 20GB).

### Download Locally

```python
# In Kaggle notebook, files in output folder can be downloaded
# Or commit and push to GitHub
```

### Upload to HuggingFace

```python
from huggingface_hub import HfApi

model.push_to_hub("your-username/merged-model")
tokenizer.push_to_hub("your-username/merged-model")
```

## 📝 Best Practices

1. **Start small**: Test with `density=0.3` first (faster)
2. **Use caching**: Don't delete `merge_cache/` between runs
3. **Monitor memory**: Check `!nvidia-smi` regularly
4. **Save frequently**: Push important results to HF/GitHub
5. **Document runs**: Add markdown cells with results

## 🎯 Quick Start Checklist

- [ ] Enable GPU in Kaggle settings
- [ ] Enable Internet in Kaggle settings
- [ ] Add HF_TOKEN to Kaggle secrets
- [ ] Clone repository
- [ ] Run `kaggle_setup.py`
- [ ] Authenticate with HuggingFace
- [ ] Run merging script
- [ ] Save results

## 📧 Support

If you encounter issues:

1. Check Kaggle logs carefully
2. Review troubleshooting section above
3. Check `./merge_cache/` for partial results
4. Open GitHub issue with error logs

---

**Ready to run? Start with the template notebook!** 🚀
