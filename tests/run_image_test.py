"""
Utility script to run image processing test with a sample image
"""

import os
import sys
from pathlib import Path
import shutil
import argparse

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def setup_test_image(image_path: str = None):
    """
    Set up test image in the correct location.
    
    Args:
        image_path: Optional path to an image to use for testing
    """
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    target_path = test_data_dir / "test_image.jpg"
    
    # If no image is provided but test_image.jpg exists, proceed
    if not image_path and target_path.exists():
        print(f"Using existing test image at {target_path}")
        return True
        
    if image_path:
        # Copy provided image to test directory
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False
            
        shutil.copy2(image_path, target_path)
        print(f"Copied test image to {target_path}")
        return True
    
    print(f"Please add a test image at: {target_path}")
    print("You can use any image file for testing.")
    return False

def main():
    parser = argparse.ArgumentParser(description='Run image processing test')
    parser.add_argument('--image', type=str, help='Path to image file to test')
    args = parser.parse_args()
    
    if setup_test_image(args.image):
        # Import and run tests
        try:
            print("\nStarting image processing test...")
            # Use relative import from the same directory
            from . import test_image_processing
            test_image_processing.main()
        except ImportError as e:
            print(f"\nError importing test module: {e}")
            print("Make sure all required packages are installed:")
            print("pip install -r requirements.txt")
            print("\nMake sure you are running from the project root directory")
        except Exception as e:
            print(f"\nError running tests: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()