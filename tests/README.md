# Tests

Comprehensive testing suite for the Blind Assistant application.

## Overview

The test suite includes:
- GPU/CUDA verification
- Model integration tests
- Image processing tests
- Driver compatibility tests

## Test Structure

```
tests/
├── test_data/              # Sample test images
├── test_output/            # Test results and logs
├── utils/
│   └── logging/            # Test logging utilities
├── test_cuda.py            # GPU/CUDA verification
├── test_image_processing.py # Integration tests
├── test_nvidia.py          # NVIDIA driver tests
└── run_image_test.py       # Image processing demo
```

## Running Tests

### Quick Tests

```bash
# Verify GPU setup
python -m tests.test_cuda

# Test object detection
python -m tests.run_image_test

# Check NVIDIA drivers
python -m tests.test_nvidia

# Run all tests
python -m pytest tests/ -v
```

### Individual Tests

#### GPU/CUDA Test

```bash
python -m tests.test_cuda
```

**What it tests**:
- PyTorch installation
- CUDA availability
- GPU device detection
- CUDA version
- GPU memory

**Expected output**:
```
=== CUDA/GPU Test ===
✓ PyTorch version: 2.0.0+cu118
✓ CUDA available: True
✓ CUDA version: 11.8
✓ GPU device: NVIDIA GeForce RTX 3050
✓ GPU memory: 4096 MB
All tests passed!
```

#### Object Detection Test

```bash
python -m tests.run_image_test
```

**What it tests**:
- Object detection model loading
- Image processing pipeline
- Detection accuracy
- Visualization

**Expected output**:
- Detected objects list
- Annotated image saved to `test_output/`

#### NVIDIA Driver Test

```bash
python -m tests.test_nvidia
```

**What it tests**:
- NVIDIA driver installation
- Driver version
- GPU compatibility

## Test Data

### Sample Images

Place test images in `tests/test_data/`:
- `test_image.jpg` - Indoor scene
- `test_outdoor.jpg` - Outdoor scene
- `test_text.jpg` - Image with text

### Test Output

Results are saved to `tests/test_output/`:
- `annotated_test_image.jpg` - Object detection results
- Test logs and reports

## Code Snippets

### Quick GPU Verification

```python
"""Quick test to verify GPU setup"""
import torch

print("=" * 50)
print("GPU Configuration Test")
print("=" * 50)

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.current_device()}")
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ Total GPU memory: {total_memory:.2f} GB")
    
    # Test tensor operation
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"✓ GPU tensor operations working")
else:
    print("✗ CUDA is not available")
    print("Please check your PyTorch installation and GPU drivers")

print("=" * 50)
```

### Complete Object Detection Test

```python
"""Comprehensive object detection test"""
from src.services.detection.object_detector import ObjectDetector
import cv2
import time
from pathlib import Path

def test_object_detection():
    print("\n" + "=" * 60)
    print("Object Detection Test Suite")
    print("=" * 60)
    
    # Initialize detector
    print("\n[1/5] Initializing detector...")
    try:
        detector = ObjectDetector()
        print("✓ Detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False
    
    # Load test image
    print("\n[2/5] Loading test image...")
    test_image_path = Path('tests/test_data/test_image.jpg')
    if not test_image_path.exists():
        print(f"✗ Test image not found: {test_image_path}")
        return False
    
    frame = cv2.imread(str(test_image_path))
    print(f"✓ Image loaded: {frame.shape}")
    
    # Run detection
    print("\n[3/5] Running object detection...")
    start_time = time.time()
    try:
        objects, annotated_frame = detector.detect(frame)
        elapsed = time.time() - start_time
        print(f"✓ Detection completed in {elapsed*1000:.2f}ms")
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False
    
    # Analyze results
    print("\n[4/5] Analyzing results...")
    print(f"Total objects detected: {len(objects)}")
    
    if objects:
        print("\nDetected objects:")
        for i, obj in enumerate(objects, 1):
            print(f"  {i}. {obj['class']}")
            print(f"     Confidence: {obj['confidence']:.2%}")
            print(f"     Depth score: {obj['depth_score']:.4f}")
            print(f"     Position: x={obj['position']['x']:.2f}, y={obj['position']['y']:.2f}")
    else:
        print("No objects detected")
    
    # Save results
    print("\n[5/5] Saving results...")
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'test_detection_result.jpg'
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"✓ Annotated image saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_object_detection()
    exit(0 if success else 1)
```

### Camera Service Test

```python
"""Test IP Webcam camera connection"""
from src.services.camera_service import CameraService
import cv2
import time

def test_camera(ip_address="192.168.1.100"):
    print("\n" + "=" * 50)
    print(f"Camera Service Test: {ip_address}")
    print("=" * 50)
    
    # Initialize camera
    print("\n[1/3] Connecting to camera...")
    camera = CameraService(source="phone", ip_address=ip_address)
    
    try:
        camera.start()
        print("✓ Camera connected")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    
    # Capture frames
    print("\n[2/3] Capturing test frames...")
    frame_count = 0
    start_time = time.time()
    
    for i in range(30):  # Capture 30 frames
        frame = camera.capture_frame()
        if frame is not None:
            frame_count += 1
            
            # Display first frame info
            if i == 0:
                print(f"✓ Frame shape: {frame.shape}")
                print(f"✓ Frame dtype: {frame.dtype}")
        
        time.sleep(0.033)  # ~30 FPS
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    
    print(f"✓ Captured {frame_count}/30 frames")
    print(f"✓ Average FPS: {fps:.2f}")
    
    # Save sample frame
    print("\n[3/3] Saving sample frame...")
    frame = camera.capture_frame()
    if frame is not None:
        cv2.imwrite('test_output/camera_test_frame.jpg', frame)
        print("✓ Sample frame saved to: test_output/camera_test_frame.jpg")
    
    # Cleanup
    camera.stop()
    print("\n" + "=" * 50)
    print("Camera test completed!")
    print("=" * 50)
    return True

if __name__ == '__main__':
    # Get IP from user
    ip = input("Enter IP Webcam address (default: 192.168.1.100): ").strip()
    if not ip:
        ip = "192.168.1.100"
    
    success = test_camera(ip)
    exit(0 if success else 1)
```

### Text-to-Speech Test

```python
"""Test TTS functionality"""
from src.services.audio.text_to_speech import TextToSpeech
import time

def test_tts():
    print("\n" + "=" * 50)
    print("Text-to-Speech Test Suite")
    print("=" * 50)
    
    # Initialize TTS
    print("\n[1/4] Initializing TTS...")
    try:
        tts = TextToSpeech()
        print("✓ TTS initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test basic speech
    print("\n[2/4] Testing basic speech...")
    try:
        tts.speak("This is a test message")
        print("✓ Basic speech working")
        time.sleep(3)
    except Exception as e:
        print(f"✗ Speech failed: {e}")
        tts.cleanup()
        return False
    
    # Test queue handling
    print("\n[3/4] Testing queue management...")
    try:
        print("Sending rapid messages (only last should play)...")
        for i in range(5):
            tts.speak(f"Message number {i+1}")
            time.sleep(0.1)
        print("✓ Queue handling working")
        time.sleep(3)
    except Exception as e:
        print(f"✗ Queue test failed: {e}")
        tts.cleanup()
        return False
    
    # Test long message
    print("\n[4/4] Testing long message...")
    try:
        long_msg = "I can see a keyboard on the left side, a mouse on the right, and a monitor in the center."
        tts.speak(long_msg)
        print("✓ Long message handling working")
        time.sleep(5)
    except Exception as e:
        print(f"✗ Long message failed: {e}")
        tts.cleanup()
        return False
    
    # Cleanup
    tts.cleanup()
    print("\n" + "=" * 50)
    print("All TTS tests passed!")
    print("=" * 50)
    return True

if __name__ == '__main__':
    success = test_tts()
    exit(0 if success else 1)
```

### Full Pipeline Integration Test

```python
"""Test complete detection and narration pipeline"""
from src.services.camera_service import CameraService
from src.services.detection.object_detector import ObjectDetector
from src.services.narration_service import NarrationService
from src.services.audio.text_to_speech import TextToSpeech
import cv2
import time

def test_full_pipeline(ip_address="192.168.1.100", duration=30):
    """Test complete pipeline for specified duration"""
    
    print("\n" + "=" * 60)
    print("Full Pipeline Integration Test")
    print("=" * 60)
    
    # Initialize all services
    print("\nInitializing services...")
    try:
        camera = CameraService(source="phone", ip_address=ip_address)
        detector = ObjectDetector()
        narration_service = NarrationService()
        tts = TextToSpeech()
        print("✓ All services initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Start camera
    print("\nStarting camera...")
    camera.start()
    time.sleep(1)
    
    # Test pipeline
    print(f"\nRunning pipeline for {duration} seconds...")
    print("(Press Ctrl+C to stop early)\n")
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    narration_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Detect objects (every 3rd frame)
            if frame_count % 3 == 0:
                objects, annotated = detector.detect(frame)
                detection_count += 1
                
                # Generate narration
                if objects:
                    narration = narration_service.generate(objects, [])
                    print(f"[{time.time() - start_time:.1f}s] {narration}")
                    
                    # Speak narration
                    tts.speak(narration)
                    narration_count += 1
                
                # Display
                cv2.imshow('Pipeline Test', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        # Cleanup
        camera.stop()
        tts.cleanup()
        cv2.destroyAllWindows()
    
    # Results
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Pipeline Test Results")
    print("=" * 60)
    print(f"Duration: {elapsed:.2f} seconds")
    print(f"Frames captured: {frame_count}")
    print(f"Detections run: {detection_count}")
    print(f"Narrations generated: {narration_count}")
    print(f"Average FPS: {frame_count/elapsed:.2f}")
    print(f"Detection rate: {detection_count/elapsed:.2f} per second")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    # Get parameters
    ip = input("Enter IP Webcam address (default: 192.168.1.100): ").strip()
    if not ip:
        ip = "192.168.1.100"
    
    duration = input("Test duration in seconds (default: 30): ").strip()
    duration = int(duration) if duration else 30
    
    success = test_full_pipeline(ip, duration)
    exit(0 if success else 1)
```

### Performance Benchmark

```python
"""Benchmark detection performance"""
from src.services.detection.object_detector import ObjectDetector
import cv2
import time
import numpy as np

def benchmark_detector(num_iterations=100):
    print("\n" + "=" * 60)
    print("Object Detection Performance Benchmark")
    print("=" * 60)
    
    # Initialize
    print("\nInitializing detector...")
    detector = ObjectDetector()
    frame = cv2.imread('tests/test_data/test_image.jpg')
    
    # Warmup
    print("Running warmup (10 iterations)...")
    for _ in range(10):
        detector.detect(frame)
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    object_counts = []
    
    for i in range(num_iterations):
        start = time.time()
        objects, _ = detector.detect(frame)
        elapsed = time.time() - start
        
        times.append(elapsed * 1000)  # Convert to ms
        object_counts.append(len(objects))
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Statistics
    times = np.array(times)
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Iterations: {num_iterations}")
    print(f"\nDetection Time (ms):")
    print(f"  Mean:    {np.mean(times):.2f} ms")
    print(f"  Median:  {np.median(times):.2f} ms")
    print(f"  Std Dev: {np.std(times):.2f} ms")
    print(f"  Min:     {np.min(times):.2f} ms")
    print(f"  Max:     {np.max(times):.2f} ms")
    print(f"\nObjects Detected:")
    print(f"  Mean:    {np.mean(object_counts):.1f}")
    print(f"  Min:     {np.min(object_counts)}")
    print(f"  Max:     {np.max(object_counts)}")
    print(f"\nTheoretical FPS: {1000/np.mean(times):.2f}")
    print("=" * 60)

if __name__ == '__main__':
    iterations = input("Number of iterations (default: 100): ").strip()
    iterations = int(iterations) if iterations else 100
    
    benchmark_detector(iterations)
```

## Writing Tests

### Basic Test Structure

```python
import unittest
from src.services.detection.object_detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        """Initialize detector before each test"""
        self.detector = ObjectDetector()
    
    def test_detection(self):
        """Test object detection"""
        import cv2
        frame = cv2.imread('tests/test_data/test_image.jpg')
        objects, annotated_frame = self.detector.detect(frame)
        
        self.assertIsNotNone(objects)
        self.assertIsInstance(objects, list)
        self.assertGreater(len(objects), 0)
    
    def tearDown(self):
        """Cleanup after each test"""
        pass

if __name__ == '__main__':
    unittest.main()
```

### Using Pytest

```python
import pytest
from src.services.detection.object_detector import ObjectDetector
import cv2

@pytest.fixture
def detector():
    """Fixture for object detector"""
    return ObjectDetector()

@pytest.fixture
def test_frame():
    """Fixture for test image"""
    return cv2.imread('tests/test_data/test_image.jpg')

def test_object_detection(detector, test_frame):
    """Test object detection"""
    objects, _ = detector.detect(test_frame)
    assert len(objects) > 0
    assert all('class' in obj for obj in objects)
    assert all('confidence' in obj for obj in objects)

def test_detection_threshold(detector, test_frame):
    """Test confidence threshold"""
    detector.confidence_threshold = 0.50
    objects, _ = detector.detect(test_frame)
    assert all(obj['confidence'] >= 0.50 for obj in objects)
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# Test object detector
python -m pytest tests/test_object_detector.py

# Test narration service
python -m pytest tests/test_narration_service.py

# Test TTS
python -m pytest tests/test_tts.py
```

### Integration Tests

Test component interactions:

```python
# Test full pipeline
python -m pytest tests/test_image_processing.py

# Test assistant
python -m pytest tests/test_assistant.py
```

### Performance Tests

Measure performance metrics:

```python
import time
from src.services.detection.object_detector import ObjectDetector

def test_detection_speed():
    detector = ObjectDetector()
    frame = cv2.imread('tests/test_data/test_image.jpg')
    
    start = time.time()
    for _ in range(10):
        objects, _ = detector.detect(frame)
    elapsed = time.time() - start
    
    avg_time = elapsed / 10
    print(f"Average detection time: {avg_time*1000:.2f}ms")
    assert avg_time < 0.2  # Should be under 200ms
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Coverage

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Benchmarking

### Performance Benchmarks

```python
from src.services.detection.object_detector import ObjectDetector
import cv2
import time

def benchmark_detection():
    detector = ObjectDetector()
    frame = cv2.imread('tests/test_data/test_image.jpg')
    
    # Warmup
    for _ in range(5):
        detector.detect(frame)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        detector.detect(frame)
        times.append(time.time() - start)
    
    print(f"Mean: {np.mean(times)*1000:.2f}ms")
    print(f"Std: {np.std(times)*1000:.2f}ms")
    print(f"Min: {np.min(times)*1000:.2f}ms")
    print(f"Max: {np.max(times)*1000:.2f}ms")

if __name__ == '__main__':
    benchmark_detection()
```

## Troubleshooting Tests

### GPU Tests Failing

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Test Data Missing

```bash
# Download test images
# Place sample images in tests/test_data/

# Or use your own images
cp /path/to/image.jpg tests/test_data/test_image.jpg
```

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/BlindAssitant

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%     # Windows
```

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Reuse common setup code
3. **Test Edge Cases**: Test boundary conditions
4. **Mock External Dependencies**: Don't rely on network/hardware
5. **Keep Tests Fast**: Optimize test execution
6. **Document Tests**: Add clear docstrings
7. **Run Tests Often**: Test before commits

## Test Utilities

### Logging

```python
from tests.utils.logging import setup_test_logger

logger = setup_test_logger('test_name')
logger.info('Test started')
logger.error('Test failed')
```

### Assertions

```python
# NumPy assertions
import numpy.testing as npt
npt.assert_array_equal(actual, expected)
npt.assert_almost_equal(actual, expected, decimal=2)

# PyTest assertions
assert value == expected
assert value > 0
assert 'key' in dictionary
```

## Related Documentation

- [PyTest Documentation](https://docs.pytest.org/)
- [Unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
