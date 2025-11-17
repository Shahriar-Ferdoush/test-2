"""
Kaggle Setup Script
Run this first in your Kaggle notebook to set up the environment.
"""

import os
import subprocess
import sys


def setup_kaggle_environment():
    """
    Sets up the Kaggle environment for model merging.
    Run this in the first cell of your Kaggle notebook.
    """

    print("=" * 80)
    print("KAGGLE ENVIRONMENT SETUP")
    print("=" * 80)

    # 1. Check GPU availability
    print("\n[1/6] Checking GPU availability...")
    import torch

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠ No GPU available. Using CPU (will be slower).")

    # 2. Install required packages
    print("\n[2/6] Installing required packages...")
    packages = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "sentencepiece",
        "protobuf",
    ]

    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    print("✓ All packages installed")

    # 3. Clone repository (if needed)
    print("\n[3/6] Repository setup...")
    repo_url = (
        "https://github.com/Shahriar-Ferdoush/test-2.git"  # Update with your repo
    )
    repo_name = "Model-Merging"

    if not os.path.exists(repo_name):
        print(f"  Cloning repository from {repo_url}...")
        subprocess.check_call(["git", "clone", repo_url, repo_name])
        print("✓ Repository cloned")
    else:
        print("✓ Repository already exists")

    # 4. Change to repository directory
    print("\n[4/6] Setting up paths...")
    os.chdir(repo_name)
    sys.path.insert(0, os.getcwd())
    print(f"✓ Working directory: {os.getcwd()}")

    # 5. Create necessary directories
    print("\n[5/6] Creating directories...")
    directories = [
        "./merge_cache",
        "./merge_cache/task_vectors",
        "./merge_cache/hessians",
        "./merged_models",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    print("✓ Directories ready")

    # 6. Check disk space
    print("\n[6/6] Checking disk space...")
    statvfs = os.statvfs(".")
    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    print(f"✓ Available disk space: {free_space_gb:.2f} GB")

    if free_space_gb < 20:
        print("⚠ Warning: Less than 20GB free. Model merging may fail.")

    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: from example_mental_health_merge import main")
    print("2. Or customize parameters and run manually")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    setup_kaggle_environment()
