# 👁️ Blind Assistant - Beta v0.1.1

> An AI-powered real-time assistance system for visually impaired individuals using computer vision and natural language processing.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### ✅ Currently Working
- **🚀 High-Performance Processing** - 8-9 FPS real-time processing (10x faster than v0.0.1)
- **🎯 Smart Narration** - Intelligent throttling prevents spam (5-second intervals, requires meaningful changes)
- **🖼️ Selective Scene Captioning** - AI descriptions only when needed (80% overhead reduction)
- **📊 Real-time Performance Monitoring** - FPS, latency, GPU/CPU usage tracking
- **👁️ Real-time Object Detection** - YOLOv8-medium detecting 80+ object classes in 35-40ms
- **📍 Spatial Awareness** - Distance estimation and positional tracking (left/right/center, above/below)
- **💬 Natural Language Narration** - Contextual scene descriptions with spatial relationships
- **🔊 Text-to-Speech** - Audio feedback using Google Text-to-Speech (gTTS)
- **⚡ GPU Acceleration** - CUDA-optimized with mixed precision (FP16) for maximum speed
- **📱 Phone Camera Integration** - Connect via IP Webcam app (WiFi)
- **🎛️ Configurable Modes** - Toggle captioning and display for different performance profiles

### 🚧 In Development
- Text Recognition (OCR) - Disabled due to compatibility issues (planned fix)
- Voice Commands - Model ready but not integrated yet
- TensorRT Optimization - Target 10+ FPS on laptop
- Mobile App Development - Android/iOS deployment

## 📋 Table of Contents
- [Demo](#-demo)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🎬 Demo

[Demo video coming soon]

**What it does:**
1. Connects to your phone camera via WiFi
2. Detects objects in real-time (8-9 FPS)
3. Generates AI scene descriptions when needed
4. Estimates distances and positions
5. Speaks intelligent, non-repetitive descriptions

**Example Output:**
> "I can see a laptop in the center. a close up of a laptop with a mouse on it"  
> [5 seconds later, after objects change]  
> "I can see a mouse on the right. a computer mouse sitting on top of a wooden table"

## 💻 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (Tested on RTX 3050, 4GB+ VRAM recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and dependencies

### Software Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.11 or higher
- **CUDA**: 11.8 (for GPU acceleration)
- **cuDNN**: Compatible with CUDA 11.8

### Mobile Requirements
- Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app
- Same WiFi network as computer

## 🚀 Installation

Choose your preferred installation method:

### Method 1: Docker (Easiest - Recommended for Linux)

**Prerequisites**: Docker with NVIDIA Container Toolkit

```bash
# Clone the repository
git clone https://github.com/taseebali/blind-assistant.git
cd blind-assistant

# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Or use the convenience script
# Linux/Mac
chmod +x docker/docker-build.sh
./docker/docker-build.sh build
./docker/docker-build.sh run 192.168.1.100

# Windows
docker\docker-build.bat build
docker\docker-build.bat run 192.168.1.100
```

👉 **See [Docker README](docker/README.md) for detailed instructions**

### Method 2: Quick Setup Script

```bash
# 1. Clone the repository
git clone https://github.com/taseebali/blind-assistant.git
cd blind-assistant

# 2. Run the setup script
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Method 3: Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/taseebali/blind-assistant.git
cd blind-assistant

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 4. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -m tests.test_cuda
```

## ⚡ Quick Start

### Step 1: Setup Phone Camera

1. Install **IP Webcam** app on your Android phone
2. Open the app and start the server
3. Note the IP address shown (e.g., `192.168.1.100`)
4. Make sure your phone and computer are on the same WiFi network

### Step 2: Run the Assistant

```bash
# Activate virtual environment (if not already active)
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run with your phone's IP address
python run.py
# When prompted, enter your phone's IP address (e.g., 192.168.1.100)
```

### Step 3: Use the Application

- Point your phone camera at objects/scenes
- Listen to the audio descriptions
- Press `Q` to quit the display window
- Press `Ctrl+C` to stop the application

## 📖 Usage

### Basic Usage

```python
from src.core.optimized_assistant import OptimizedAssistant

# Create assistant with display and scene captioning
assistant = OptimizedAssistant(
    show_display=True,
    camera_ip="192.168.1.100",  # Your phone's IP
    enable_captioning=True      # Enable AI scene descriptions
)

# Start the assistant
assistant.start()
```

### Advanced Usage

```python
# Disable visual display (audio only) or disable scene captioning
assistant = OptimizedAssistant(
    show_display=False, 
    camera_ip="192.168.1.100",
    enable_captioning=False  # Disable captioning for faster processing
)

# Use with custom settings
from src.services.detection.object_detector import ObjectDetector

detector = ObjectDetector(
    model_name='yolov8m.pt',
    confidence_threshold=0.30
)
```

### Command Line Options

```bash
# Run with specific camera IP
python run.py 192.168.1.100

# Run tests
python -m pytest tests/

# Check GPU status
python -m tests.test_cuda

# Test with sample image
python -m tests.run_image_test
```

## Running Tests

1. Run all tests:
```bash
python -m pytest tests/
```

2. Run specific test modules:
```bash
# GPU/CUDA tests
python -m tests.test_cuda

# Image processing tests
python -m tests.run_image_test

# NVIDIA driver tests
python -m tests.test_nvidia
```

## Features

- Real-time object detection using YOLOv8
- Text recognition using EasyOCR
- Scene description using BLIP
- Natural language narration
- Text-to-speech output
- GPU acceleration for all AI models
- Comprehensive testing and logging

## 📁 Project Structure

```
blind-assistant/
├── src/
│   ├── core/
│   │   ├── README.md                    # Core module documentation
│   │   └── assistant.py                 # Main BlindAssistant orchestrator
│   ├── services/
│   │   ├── detection/
│   │   │   ├── README.md                # Detection services docs
│   │   │   ├── object_detector.py       # YOLOv8 object detection
│   │   │   ├── text_detector.py         # EasyOCR text recognition (disabled)
│   │   │   └── caption_generator.py     # BLIP image captioning (disabled)
│   │   ├── audio/
│   │   │   ├── README.md                # Audio services docs
│   │   │   ├── text_to_speech.py        # gTTS text-to-speech
│   │   │   └── speech_recognizer.py     # Whisper (not implemented)
│   │   ├── README.md                    # Services overview
│   │   ├── camera_service.py            # IP Webcam integration
│   │   └── narration_service.py         # Natural language generation
│   └── main.py                          # Application entry point
├── tests/
│   ├── test_data/                       # Sample test images
│   ├── test_output/                     # Test results
│   ├── utils/logging/                   # Test logging utilities
│   ├── README.md                        # Testing documentation
│   ├── test_cuda.py                     # GPU/CUDA verification
│   ├── test_image_processing.py         # Integration tests
│   └── test_nvidia.py                   # NVIDIA driver tests
├── run.py                               # Main entry point
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
├── LICENSE                              # MIT License
└── README.md                            # This file
```

### Module Documentation

Each module has its own README with detailed information:
- **[Core Module](src/core/README.md)** - Assistant orchestration and coordination
- **[Services](src/services/README.md)** - Individual AI services and utilities
- **[Detection Services](src/services/detection/README.md)** - Computer vision models
- **[Audio Services](src/services/audio/README.md)** - Speech synthesis and recognition
- **[Tests](tests/README.md)** - Testing framework and guidelines

## ⚡ Performance

### Current Metrics (Optimized - v0.1.0)

**RTX 3050 Laptop GPU (4GB VRAM)**

#### Processing Speed
- **Overall FPS**: 8-9 FPS (10x faster than v0.0.1)
- **Object Detection**: 35-40ms per frame
- **Scene Captioning**: 350-450ms (only when triggered)
- **Narration Generation**: <1ms
- **Total Response Time**: ~100ms from scene change to narration

#### Resource Usage
- **GPU Memory**: 0.7GB allocated, 1.1GB reserved (down from 3-4GB)
- **CPU Usage**: 8-12% average
- **RAM Usage**: System dependent

#### Optimization Results
- **Frame Skip Ratio**: ~240% (processes 1 in 3 frames efficiently)
- **Captioning Overhead**: Reduced from 100% to ~20% via selective triggering
- **Narration Rate**: ~7-8 meaningful narrations per 38 seconds (vs 96 spam before)

### Performance Modes

#### Maximum Speed (Headless + No Captioning)
```python
assistant = OptimizedAssistant(
    show_display=False,
    enable_captioning=False
)
# Expected: 10+ FPS
```

#### Balanced (Default)
```python
assistant = OptimizedAssistant(
    show_display=True,
    enable_captioning=True
)
# Expected: 8-9 FPS
```

#### Full Features (Display + Captioning)
```python
assistant = OptimizedAssistant(
    show_display=True,
    enable_captioning=True
)
# Expected: 8-9 FPS (optimized captioning keeps speed high)
```

### Optimization Tips
1. **Disable captioning** for maximum speed: `enable_captioning=False`
2. **Adjust frame skip**: `assistant.frame_skip = 5` (process every 5th frame)
3. **Increase narration interval**: `assistant.narration_interval = 10` (narrate every 10s)
4. **Disable visual display**: `show_display=False` for headless operation
5. **Use smaller model** (future): Switch to `yolov8n.pt` for even faster processing

### Real-Time Monitoring

The system provides live performance metrics:
```
Processing: 8.79 FPS | Avg: 37.3ms

============================================================
PERFORMANCE REPORT (Every 10 seconds)
============================================================

Timing (ms):
  object_detection         :   35.1 (min:  29.1, max:  39.8)
  scene_captioning         :  415.1 (min: 345.1, max: 571.4)
  narration_generation     :    0.3 (min:   0.0, max:   1.5)
  
Counters:
  frames_processed         : 269
  captions_generated       : 6
  narrations_generated     : 81
  
Rates:
  Processing FPS:           7.88
```

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `CUDA not available`
```bash
# Solution: Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

### Runtime Issues

**Problem**: Cannot connect to phone camera
```bash
# Solution:
# 1. Check both devices are on same WiFi
# 2. Verify IP address in IP Webcam app
# 3. Try pinging the phone: ping 192.168.1.100
# 4. Check firewall settings
```

**Problem**: Slow performance / Low FPS
```bash
# Solution:
# 1. Check GPU usage: nvidia-smi
# 2. Reduce camera quality in IP Webcam settings
# 3. Use smaller model in object_detector.py
# 4. Close other GPU-intensive applications
```

**Problem**: Audio not working
```bash
# Solution:
# 1. Check internet connection (gTTS requires internet)
# 2. Verify speakers/headphones are connected
# 3. Check Windows audio settings
```

### Testing

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

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **EasyOCR** by JaidedAI
- **BLIP** by Salesforce Research
- **gTTS** by Pierre Nicolas Durette
- **IP Webcam** by Pavel Khlebovich

## 📧 Contact

**Author**: Taseeb Ali
**GitHub**: [@taseebali](https://github.com/taseebali)
**Repository**: [blind-assistant](https://github.com/taseebali/blind-assistant)

## 🗺️ Roadmap

### v0.2.0 (Next Release)
- [ ] Fix text detection (OCR)
- [ ] Implement voice commands
- [ ] Add offline mode
- [ ] Improve response time (<1 second)
- [ ] Mobile app version

### v0.3.0 (Future)
- [ ] Navigation assistance
- [ ] Obstacle detection
- [ ] Face recognition
- [ ] Custom wake words
- [ ] Cloud deployment

---

**Note**: This is a beta version. Please report issues on the [GitHub Issues](https://github.com/taseebali/blind-assistant/issues) page.
