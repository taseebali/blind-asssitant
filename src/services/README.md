# Services Module

This module contains all AI services and utility components for the Blind Assistant.

## Overview

The services module provides modular, independent AI services that can be used standalone or integrated into the main assistant:

- **Detection Services**: Computer vision models for object detection, text recognition, and image captioning
- **Audio Services**: Text-to-speech and speech recognition
- **Camera Service**: Video capture from IP Webcam
- **Narration Service**: Natural language generation

## Module Structure

```
services/
├── detection/
│   ├── README.md                 # Detection services documentation
│   ├── object_detector.py        # YOLOv8 object detection
│   ├── text_detector.py          # EasyOCR text recognition
│   └── caption_generator.py      # BLIP image captioning
├── audio/
│   ├── README.md                 # Audio services documentation
│   ├── text_to_speech.py         # gTTS text-to-speech
│   └── speech_recognizer.py      # Whisper speech recognition
├── camera_service.py             # IP Webcam integration
└── narration_service.py          # Natural language generation
```

## Services Overview

### Detection Services

#### Object Detector (YOLOv8)
**Status**: ✅ Working  
**Model**: YOLOv8-medium  
**Purpose**: Detect and localize objects in images

**Features**:
- 80+ object classes
- Real-time detection
- GPU accelerated
- Depth estimation based on object size
- Spatial position tracking

**Usage**:
```python
from src.services.detection.object_detector import ObjectDetector

detector = ObjectDetector()
objects, annotated_frame = detector.detect(frame)
```

#### Text Detector (EasyOCR)
**Status**: ⚠️ Disabled (compatibility issues)  
**Purpose**: Extract text from images

**Features**:
- Multi-language support
- GPU accelerated
- Bounding box detection

#### Caption Generator (BLIP)
**Status**: ⚠️ Disabled (performance)  
**Purpose**: Generate natural language descriptions of images

**Features**:
- Context-aware captions
- GPU accelerated
- High accuracy

### Audio Services

#### Text-to-Speech (gTTS)
**Status**: ✅ Working  
**Purpose**: Convert text to natural speech

**Features**:
- Natural-sounding voice
- Queue-based playback
- Automatic audio file management

**Usage**:
```python
from src.services.audio.text_to_speech import TextToSpeech

tts = TextToSpeech()
tts.speak("Hello, world!")
```

#### Speech Recognizer (Whisper)
**Status**: 🚧 Not implemented  
**Purpose**: Convert speech to text for voice commands

### Camera Service

**Status**: ✅ Working  
**Purpose**: Capture video frames from IP Webcam app

**Features**:
- WiFi streaming from Android phone
- Configurable quality and FPS
- Auto-reconnection
- Frame preprocessing

**Usage**:
```python
from src.services.camera_service import CameraService

camera = CameraService(source="phone", ip_address="192.168.1.100")
camera.start()
frame = camera.capture_frame()
```

### Narration Service

**Status**: ✅ Working  
**Purpose**: Generate natural language scene descriptions

**Features**:
- Spatial relationship descriptions
- Object counting and grouping
- Position-aware narration
- Distance-based filtering

**Usage**:
```python
from src.services.narration_service import NarrationService

narration_service = NarrationService()
narration = narration_service.generate(objects, texts)
# Output: "I can see a keyboard on the left, with a mouse to the right of it"
```

## Code Snippets

### Complete Service Integration Example

```python
from src.services.camera_service import CameraService
from src.services.detection.object_detector import ObjectDetector
from src.services.narration_service import NarrationService
from src.services.audio.text_to_speech import TextToSpeech
import cv2
import time

def process_scene():
    """Complete example of all services working together"""
    
    # Initialize all services
    camera = CameraService(source="phone", ip_address="192.168.1.100")
    detector = ObjectDetector()
    narration_service = NarrationService()
    tts = TextToSpeech()
    
    print("Starting camera...")
    camera.start()
    
    try:
        while True:
            # 1. Capture frame
            frame = camera.capture_frame()
            if frame is None:
                print("No frame received")
                time.sleep(0.1)
                continue
            
            # 2. Detect objects
            objects, annotated_frame = detector.detect(frame)
            print(f"Detected {len(objects)} objects")
            
            # 3. Generate narration
            if objects:
                narration = narration_service.generate(objects, [])
                print(f"Narration: {narration}")
                
                # 4. Speak narration
                tts.speak(narration)
            
            # 5. Display result (optional)
            cv2.imshow('Scene', annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Wait 2 seconds before next narration
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        camera.stop()
        tts.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_scene()
```

### Using Individual Services

#### Camera Service Only

```python
from src.services.camera_service import CameraService
import cv2

# Connect to IP Webcam
camera = CameraService(source="phone", ip_address="192.168.1.100")
camera.start()

# Capture and display frames
while True:
    frame = camera.capture_frame()
    if frame is not None:
        cv2.imshow('Camera Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
```

#### Object Detection Only

```python
from src.services.detection.object_detector import ObjectDetector
import cv2

# Load an image
frame = cv2.imread('test_image.jpg')

# Detect objects
detector = ObjectDetector()
objects, annotated_frame = detector.detect(frame)

# Print detections
for obj in objects:
    print(f"{obj['class']}: {obj['confidence']:.2f}")
    print(f"  Position: {obj['position']}")
    print(f"  Depth: {obj['depth_score']:.4f}")

# Save result
cv2.imwrite('result.jpg', annotated_frame)
```

#### Text-to-Speech Only

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

tts = TextToSpeech()

# Speak multiple messages
messages = [
    "Welcome to Blind Assistant",
    "I can help you understand your surroundings",
    "Point your camera at objects to detect them"
]

for msg in messages:
    tts.speak(msg)
    time.sleep(3)  # Wait for speech to complete

tts.cleanup()
```

#### Narration Service Only

```python
from src.services.narration_service import NarrationService

# Sample detection data
objects = [
    {
        'class': 'keyboard',
        'confidence': 0.95,
        'depth_score': 0.08,
        'position': {'x': 0.3, 'y': 0.5}  # Left side
    },
    {
        'class': 'mouse',
        'confidence': 0.90,
        'depth_score': 0.06,
        'position': {'x': 0.7, 'y': 0.5}  # Right side
    },
    {
        'class': 'monitor',
        'confidence': 0.85,
        'depth_score': 0.15,
        'position': {'x': 0.5, 'y': 0.2}  # Center top
    }
]

# Generate natural language description
narration_service = NarrationService()
narration = narration_service.generate(objects, [])
print(narration)
# Output: "I can see a monitor in the center, with a keyboard on the left and a mouse on the right"
```

### Error Handling Pattern

```python
from src.services.camera_service import CameraService
from src.services.detection.object_detector import ObjectDetector
import cv2

camera = CameraService(source="phone", ip_address="192.168.1.100")
detector = ObjectDetector()

try:
    camera.start()
    
    while True:
        # Handle camera errors
        frame = camera.capture_frame()
        if frame is None:
            print("Camera error, retrying...")
            time.sleep(1)
            continue
        
        # Handle detection errors
        try:
            objects, annotated = detector.detect(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            objects = []
            annotated = frame
        
        # Continue processing even if errors occur
        if objects:
            print(f"Found {len(objects)} objects")
        
        cv2.imshow('Frame', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    camera.stop()
    cv2.destroyAllWindows()
```

## Performance Considerations

### GPU Memory Usage
- **Object Detector**: ~1.5GB VRAM
- **Text Detector**: ~2GB VRAM (when enabled)
- **Caption Generator**: ~1GB VRAM (when enabled)
- **Total**: ~3-4GB VRAM for all services

### Processing Times (RTX 3050)
- **Object Detection**: 100-150ms per frame
- **Text Detection**: 200-300ms per frame (when enabled)
- **Image Captioning**: 200-300ms per frame (when enabled)
- **Narration Generation**: <10ms
- **Text-to-Speech**: 500-1000ms (depends on internet)

## Configuration

Each service can be configured independently:

```python
# Object Detector
detector = ObjectDetector(
    model_name='yolov8m.pt',      # Model size (n/s/m/l/x)
    confidence_threshold=0.30      # Detection threshold
)

# Camera Service
camera = CameraService(
    source="phone",
    ip_address="192.168.1.100",
    port="8080"
)

# Narration Service
narration_service = NarrationService()
narration_service.depth_threshold = 0.05  # Distance filter
```

## Error Handling

All services implement robust error handling:

- **Automatic retries**: Camera reconnection, model reloading
- **Graceful degradation**: Continue operation if one service fails
- **Detailed logging**: Error messages with context

## Testing

Each service has standalone tests:

```bash
# Test object detection
python -c "from src.services.detection.object_detector import ObjectDetector; print('OK')"

# Test camera service
python -c "from src.services.camera_service import CameraService; print('OK')"

# Test TTS
python -c "from src.services.audio.text_to_speech import TextToSpeech; print('OK')"
```

## Dependencies

### Core Dependencies
- `torch`, `torchvision`: Deep learning framework
- `opencv-python`: Computer vision
- `numpy`: Numerical computing

### Service-Specific Dependencies
- `ultralytics`: YOLOv8 object detection
- `easyocr`: Text recognition
- `transformers`: BLIP image captioning
- `gtts`: Text-to-speech
- `pygame`: Audio playback

## Future Enhancements

- [ ] Enable text detection with fixed compatibility
- [ ] Implement voice commands with Whisper
- [ ] Add face recognition service
- [ ] Add obstacle detection service
- [ ] Offline TTS option
- [ ] Custom model training support
- [ ] Service health monitoring

## Related Documentation

- [Detection Services](detection/README.md)
- [Audio Services](audio/README.md)
- [Core Module](../core/README.md)
