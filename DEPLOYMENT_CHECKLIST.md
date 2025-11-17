# 🚀 Complete Deployment Checklist

Follow these steps to push code to GitHub and run on Kaggle.

## ✅ Phase 1: Local Preparation (Do this NOW)

### 1.1 Verify Files

```powershell
cd "g:\Thesis\Model-Merging"
ls
```

Expected files:

- ✅ `llama_merge.py`
- ✅ `sparsegpt_importance.py`
- ✅ `ties_utils.py`
- ✅ `dare_utils.py`
- ✅ `example_mental_health_merge.py`
- ✅ `kaggle_setup.py`
- ✅ `kaggle_notebook_template.ipynb`
- ✅ `requirements.txt`
- ✅ `.gitignore`
- ✅ `README.md`
- ✅ `KAGGLE_GUIDE.md`

### 1.2 Test Locally (Optional)

```powershell
# Quick syntax check
python -m py_compile llama_merge.py
python -m py_compile kaggle_setup.py
```

### 1.3 Initialize Git

```powershell
cd "g:\Thesis\Model-Merging"

# Check if git is initialized
git status

# If not initialized:
git init
git add .
git commit -m "Initial commit: Model merging toolkit with SparseGPT"
```

### 1.4 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `Model-Merging` (or any name)
3. Description: "Model merging toolkit with TIES, DARE, and SparseGPT"
4. Privacy: **Public** or **Private** (your choice)
5. **DO NOT** initialize with README (we already have one)
6. Click **Create Repository**

### 1.5 Push to GitHub

```powershell
# Use the commands GitHub shows you:
git remote add origin https://github.com/Shahriar-Ferdoush/test-2.git
git branch -M main
git push -u origin main
```

**Verify**: Go to your GitHub repo URL and confirm all files are there.

---

## ✅ Phase 2: HuggingFace Setup

### 2.1 Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `kaggle-model-merging`
4. Type: **Read**
5. Click **Generate**
6. **Copy the token** (starts with `hf_...`)

### 2.2 Accept LLaMA Terms

1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Click **"Agree and access repository"**
3. Fill out the form if prompted
4. Wait for approval (usually instant)

---

## ✅ Phase 3: Kaggle Setup

### 3.1 Enable GPU

1. Go to https://www.kaggle.com/
2. Click **Settings** → **Account**
3. **Phone verification** (required for GPU)
4. Verify your phone number

### 3.2 Create New Notebook

1. Click **"+ New Notebook"**
2. Title: `Model Merging with SparseGPT`

### 3.3 Configure Notebook Settings

Click **Settings** (⚙️ icon) and enable:

- **Accelerator**: GPU T4 x2 ✅
- **Internet**: On ✅
- **Persistence**: Files only ✅

### 3.4 Add HuggingFace Token to Secrets

1. In notebook, click **Add-ons** → **Secrets**
2. Click **"+ Add a new secret"**
3. Name: `HF_TOKEN`
4. Value: Paste your HuggingFace token
5. Click **Add**

---

## ✅ Phase 4: Run on Kaggle

### 4.1 Clone Repository (Cell 1)

```python
# Clone your repository
!git clone https://github.com/Shahriar-Ferdoush/test-2.git
%cd test-2

# Verify files
!ls -la
```

### 4.2 Setup Environment (Cell 2)

```python
# Run setup script
!python kaggle_setup.py
```

Expected output:

```
✓ GPU available: Tesla T4
✓ All packages installed
✓ Repository cloned
✓ Working directory: /kaggle/working/test-2
✓ Directories ready
✓ Available disk space: XX.XX GB
SETUP COMPLETE!
```

### 4.3 Authenticate HuggingFace (Cell 3)

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

print("✓ Authenticated with HuggingFace")
```

### 4.4 Configure and Run (Cell 4)

```python
import torch
from llama_merge import LLaMAMerger

# Configure your setup
merger = LLaMAMerger(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    finetuned_model_paths=[
        "your-model-path",  # UPDATE THIS
    ],
    dataset_names=[
        "your-dataset",  # UPDATE THIS
    ],
    output_dir="./merged_models",
    cache_dir="./merge_cache",
    density=0.2,
    num_calibration_samples=128,
    device="cuda"
)

print("✓ Merger configured")
```

### 4.5 Run Merging (Cell 5)

```python
# Run all three methods and compare
results = merger.merge_all_methods()

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
for method, metrics in results.items():
    print(f"\n{method}:")
    print(f"  Perplexity: {metrics['perplexity']:.4f}")
    print(f"  Time: {metrics['time']:.2f}s")

best = min(results, key=lambda k: results[k]['perplexity'])
print(f"\n🏆 Best: {best} (Perplexity: {results[best]['perplexity']:.4f})")
```

### 4.6 Test the Model (Cell 6)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load best model
model_path = "./merged_models/ties_sparsegpt_merged"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Test prompt
prompt = "I've been feeling anxious lately. What should I do?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)
```

---

## ✅ Phase 5: Save Results

### Option A: Download from Kaggle

1. Models are saved in `/kaggle/working/merged_models/`
2. Click **Output** tab in Kaggle
3. Download the merged model folders

### Option B: Upload to HuggingFace

```python
from huggingface_hub import HfApi

repo_name = "your-username/merged-mental-health-counselor"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✓ Uploaded to {repo_name}")
```

### Option C: Push to GitHub

```python
# In Kaggle notebook
!git config --global user.name "Your Name"
!git config --global user.email "your@email.com"
!git add merged_models/
!git commit -m "Add merged models"
!git push
```

---

## ⏱️ Expected Timeline

| Phase              | Time           | Status         |
| ------------------ | -------------- | -------------- |
| Local preparation  | 10 min         | ⬜ Not started |
| GitHub push        | 5 min          | ⬜ Not started |
| HuggingFace setup  | 5 min          | ⬜ Not started |
| Kaggle setup       | 10 min         | ⬜ Not started |
| Environment setup  | 5 min          | ⬜ Not started |
| Model merging (1B) | 30-60 min      | ⬜ Not started |
| Testing & saving   | 10 min         | ⬜ Not started |
| **Total**          | **75-105 min** |                |

---

## 🆘 Quick Troubleshooting

### "git: command not found"

```powershell
# Install Git from https://git-scm.com/download/win
# Or use GitHub Desktop
```

### "Permission denied (GitHub)"

```powershell
# Use personal access token
# GitHub → Settings → Developer settings → Personal access tokens
# Use token as password when pushing
```

### "No GPU in Kaggle"

- Verify phone number in Kaggle account settings
- Check GPU quota (30 hours/week free tier)
- Try different time of day (less congestion)

### "HuggingFace authentication failed"

- Regenerate token with **Read** permission
- Re-add to Kaggle secrets
- Restart kernel

### "Out of memory"

```python
# Reduce calibration samples
merger = LLaMAMerger(..., num_calibration_samples=64)
```

---

## 📋 Pre-Flight Checklist

Before running on Kaggle, verify:

- [ ] All files pushed to GitHub
- [ ] Repository is public or accessible
- [ ] HuggingFace token created (Read permission)
- [ ] LLaMA model terms accepted on HuggingFace
- [ ] Kaggle phone verified
- [ ] GPU enabled in Kaggle notebook
- [ ] Internet enabled in Kaggle notebook
- [ ] HF_TOKEN added to Kaggle secrets
- [ ] Model paths updated in code
- [ ] Dataset names updated in code

---

## 🎯 Next Steps After Successful Run

1. **Compare results**: Which method performed best?
2. **Test thoroughly**: Try different prompts
3. **Document findings**: Record perplexity scores
4. **Share models**: Upload to HuggingFace Hub
5. **Iterate**: Try different density values (0.1, 0.2, 0.3)

---

**Ready? Start with Phase 1!** 🚀

**Questions?** Check [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for detailed troubleshooting.
