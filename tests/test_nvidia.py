"""
Test NVIDIA GPU availability
"""

import sys
import subprocess
from pathlib import Path

def check_nvidia_smi():
    """Check if nvidia-smi is available and get GPU info"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("NVIDIA-SMI Output:")
        print("-" * 50)
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("nvidia-smi not found. This could mean:")
        print("1. NVIDIA drivers are not installed")
        print("2. NVIDIA drivers are not in system PATH")
        print("\nPlease ensure you have:")
        print("1. Installed NVIDIA drivers from https://www.nvidia.com/download/index.aspx")
        print("2. Installed CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA configuration"""
    print("\nPyTorch CUDA Configuration:")
    print("-" * 50)
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("\nPyTorch is not detecting CUDA. This could mean:")
        print("1. PyTorch was installed without CUDA support")
        print("2. CUDA toolkit is not properly installed")
        print("3. Your GPU is not CUDA-capable")
        print("\nTry reinstalling PyTorch with:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def main():
    print("\nChecking NVIDIA GPU Setup...")
    print("=" * 50)
    
    nvidia_available = check_nvidia_smi()
    
    if nvidia_available:
        check_torch_cuda()
    
if __name__ == "__main__":
    main()