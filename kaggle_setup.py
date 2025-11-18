"""
Kaggle Setup Script for Model Merging

IMPORTANT: Run this AFTER cloning the repo in your notebook!

Expected notebook flow:
    !git clone -b v2 https://github.com/Shahriar-Ferdoush/test-2.git
    %cd test-2/Model-Merging
    !python kaggle_setup.py

Or pass branch name:
    !python kaggle_setup.py --branch v2

This script will:
1. Check GPU
2. Install required packages
3. Create directories
4. Verify you're in the right branch
"""

import argparse
import os
import subprocess
import sys


def setup_kaggle_environment(branch_name="v2"):
    """
    Sets up the Kaggle environment for model merging.
    Assumes you've already cloned the repo and cd'd into it.
    """

    print("=" * 80)
    print("KAGGLE ENVIRONMENT SETUP")
    print("=" * 80)

    # 0. Verify we're in a git repo
    print("\n[0/5] Verifying repository...")
    if not os.path.exists(".git"):
        print("❌ ERROR: Not in a git repository!")
        print("\nExpected workflow:")
        print("  !git clone -b v2 https://github.com/Shahriar-Ferdoush/test-2.git")
        print("  %cd test-2/Model-Merging")
        print("  !python kaggle_setup.py")
        return False

    # Check current branch
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()
        print(f"✓ In git repository")
        print(f"  Current branch: {current_branch}")

        if current_branch != branch_name:
            print(
                f"  ⚠️  WARNING: Expected branch '{branch_name}', but on '{current_branch}'"
            )
            print(f"  To switch: !git checkout {branch_name}")
        else:
            print(f"  ✓ On correct branch: {branch_name}")
    except subprocess.CalledProcessError:
        print("  ⚠️  Could not determine current branch")

    # 1. Check GPU availability
    print("\n[1/5] Checking GPU availability...")
    import torch

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠ No GPU available. Using CPU (will be slower).")

    # 2. Install required packages
    print("\n[2/5] Installing required packages...")
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

    # 3. Add current directory to Python path
    print("\n[3/5] Setting up Python path...")
    sys.path.insert(0, os.getcwd())
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"  Added to sys.path")

    # 4. Create necessary directories
    print("\n[4/5] Creating directories...")
    directories = [
        "./merge_cache",
        "./merged_models",
        "./fine_tuned_models",  # For storing fine-tuned models
    ]
    # Note: We don't cache Hessians (saves ~12GB) - computed in-memory only

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    print("✓ Directories ready")

    # 5. Check disk space
    print("\n[5/5] Checking disk space...")
    statvfs = os.statvfs(".")
    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    print(f"✓ Available disk space: {free_space_gb:.2f} GB")

    if free_space_gb < 15:
        print("❌ ERROR: Less than 15GB free! Model merging will fail.")
        print("   Need at least 15GB for base + fine-tuned + merged models")
    elif free_space_gb < 20:
        print("⚠️  WARNING: Less than 20GB free. May run out of space.")
        print("   Recommended: 20GB+ for safe merging")
    else:
        print("✓ Sufficient disk space available")

    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print(f"\n📋 Summary:")
    print(f"   Branch: {current_branch if 'current_branch' in locals() else 'unknown'}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Free space: {free_space_gb:.2f} GB")
    print("\n📌 Next steps:")
    print("   1. Load your fine-tuned models (LoRA adapters or full models)")
    print("   2. Run merging: from llama_merge import LLaMAMerger")
    print("   3. Check KAGGLE_SETUP.md for detailed instructions")
    print("\n💡 Notes:")
    print("   - Hessians computed in-memory (not cached) to save ~12GB")
    print("   - Expected storage: ~15-20GB total for 2 models + merge")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Kaggle environment setup for model merging"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="v2",
        help="Expected git branch name (default: v2)",
    )
    args = parser.parse_args()

    success = setup_kaggle_environment(branch_name=args.branch)
    sys.exit(0 if success else 1)
