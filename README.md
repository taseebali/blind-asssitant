# Blind Assistant

An AI-powered assistant for the visually impaired that provides real-time object detection, text recognition, and scene description.

## Prerequisites

1. Python 3.11 or higher
2. NVIDIA GPU (tested with RTX 3050)
3. CUDA Toolkit 11.8
4. cuDNN compatible with CUDA 11.8

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd BlindAssitant
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install PyTorch with CUDA support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:
```bash
pip install ultralytics  # YOLOv8
pip install easyocr     # Text detection
pip install transformers # BLIP image captioning
pip install pygame      # Audio playback
pip install opencv-python
pip install pillow
pip install pytest
```

## Running the Application

1. Run the main application (with visual display):
```bash
python -m src.main
```

2. Run without visual display:
```python
from src.core.assistant import BlindAssistant
assistant = BlindAssistant(show_display=False)
assistant.start()
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

## Project Structure

```
BlindAssitant/
├── src/
│   ├── core/
│   │   └── assistant.py        # Main BlindAssistant class
│   ├── services/
│   │   ├── detection/
│   │   │   ├── object_detector.py    # YOLOv8 object detection
│   │   │   ├── text_detector.py      # EasyOCR text recognition
│   │   │   └── caption_generator.py   # BLIP image captioning
│   │   ├── audio/
│   │   │   └── text_to_speech.py     # Text-to-speech service
│   │   └── narration_service.py      # Natural language generation
│   └── main.py                 # Application entry point
├── tests/
│   ├── test_data/             # Test images
│   ├── test_output/           # Test results
│   ├── utils/
│   │   └── logging/           # Test logging
│   ├── test_cuda.py          # GPU testing
│   ├── test_image_processing.py  # Integration tests
│   └── test_nvidia.py        # Driver testing
└── requirements.txt           # Dependencies
```

## Performance

The application is optimized for GPU acceleration:
- YOLOv8: ~6.6ms inference time
- EasyOCR: ~100-200ms per image
- BLIP: ~200-300ms per caption

## Troubleshooting

1. CUDA Issues:
- Run `python -m tests.test_cuda` to verify GPU setup
- Check NVIDIA driver installation
- Verify PyTorch CUDA installation

2. Model Downloads:
- Models will be downloaded automatically on first run
- Ensure internet connection for initial setup

3. Common Issues:
- "CUDA not available": Install correct PyTorch version
- "Model not found": Check internet connection
- "GPU out of memory": Reduce batch size or use smaller models

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request