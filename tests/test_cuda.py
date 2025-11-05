"""
Test CUDA availability and device configuration
"""

import torch
import sys
from pathlib import Path
import subprocess

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def setup_cuda_optimizations():
    """Configure CUDA for optimal performance"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = False
        # Set default tensor type to CUDA
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return True
    return False

def get_nvidia_smi_output():
    """Get GPU information from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi command not found"

def main():
    print("\nChecking CUDA availability:")
    print("-" * 50)
    
    # Setup CUDA optimizations
    cuda_optimized = setup_cuda_optimizations()
    print(f"CUDA optimizations enabled: {cuda_optimized}")
    
    # Check CUDA availability
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        print(f"\nGPU Properties:")
        print(f"  Name: {device_props.name}")
        print(f"  Compute capability: {device_props.major}.{device_props.minor}")
        print(f"  Total memory: {device_props.total_memory / 1024**2:.2f} MB")
        print(f"  Multi-processor count: {device_props.multi_processor_count}")
        
        # Test CUDA memory allocation
        print("\nTesting CUDA memory allocation:")
        try:
            x = torch.rand(1000, 1000).cuda()
            print("  Successfully allocated tensor on GPU")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error allocating tensor on GPU: {e}")
    
    # Check cuDNN
    print("\nChecking cuDNN:")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN is enabled: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
    
    # Show GPU utilization
    print("\nGPU Utilization:")
    print("-" * 50)
    print(get_nvidia_smi_output())

if __name__ == "__main__":
    main()