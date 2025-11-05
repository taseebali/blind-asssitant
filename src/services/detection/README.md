# Detection Services

Computer vision models for object detection, text recognition, and image captioning.

## Overview

The detection services module provides three main AI models for visual understanding:

1. **Object Detector** (YOLOv8) - Real-time object detection ✅
2. **Text Detector** (EasyOCR) - Optical character recognition ⚠️ Disabled
3. **Caption Generator** (BLIP) - Image captioning ⚠️ Disabled

## Object Detector

### Description

Uses YOLOv8-medium model for real-time object detection with spatial awareness.

### Features

- **80+ Object Classes**: person, car, chair, keyboard, mouse, etc.
- **GPU Accelerated**: CUDA-optimized inference
- **Spatial Tracking**: Position (left/center/right) and depth estimation
- **Custom Mappings**: Bottle/vase/cup → "drink" for better narration
- **Visual Feedback**: Bounding boxes with labels and confidence scores

### Usage

```python
from src.services.detection.object_detector import ObjectDetector
import cv2

# Initialize detector
detector = ObjectDetector(
    model_name='yolov8m.pt',        # Model size
    confidence_threshold=0.30        # Detection threshold
)

# Load and process image
frame = cv2.imread('image.jpg')
objects, annotated_frame = detector.detect(frame)

# Access detection results
for obj in objects:
    print(f"Detected: {obj['class']}")
    print(f"Confidence: {obj['confidence']:.2f}")
    print(f"Position: {obj['position']}")
    print(f"Depth Score: {obj['depth_score']}")
```

### Detection Output Format

```python
[
    {
        'class': 'keyboard',
        'confidence': 0.95,
        'box': (100, 200, 300, 400),  # x1, y1, x2, y2
        'depth_score': 0.08,           # Larger = closer
        'position': {
            'x': 0.5,                  # 0=left, 0.5=center, 1=right
            'y': 0.6,                  # 0=top, 0.5=middle, 1=bottom
            'center': (200, 300)        # Pixel coordinates
        }
    },
    ...
]
```

### Configuration

```python
# Model Selection
# yolov8n.pt - Nano (fastest, least accurate)
# yolov8s.pt - Small
# yolov8m.pt - Medium (default, balanced)
# yolov8l.pt - Large
# yolov8x.pt - Extra Large (slowest, most accurate)

detector = ObjectDetector(model_name='yolov8m.pt')

# Confidence Threshold
detector.confidence_threshold = 0.30  # Lower = more detections

# Custom Class Mappings
detector.class_mappings = {
    'bottle': 'drink',
    'wine glass': 'drink',
    'cup': 'drink',
    'vase': 'drink'
}

# Class-Specific Thresholds
detector.class_conf_thresholds = {
    'person': 0.40,    # Higher threshold for people
    'bottle': 0.25     # Lower threshold for bottles
}
```

### Performance

**RTX 3050 (4GB VRAM)**:
- Inference Time: 100-150ms per frame
- GPU Memory: ~1.5GB
- Detection Range: 1-50 objects per frame
- Accuracy: ~85% mAP on COCO dataset

### Depth Estimation

The detector estimates relative depth using object size:

```python
# depth_score calculation
box_area = (x2 - x1) * (y2 - y1)
frame_area = width * height
depth_score = box_area / frame_area

# Interpretation:
# > 0.10 = Very close
# 0.05-0.10 = Close
# 0.02-0.05 = Medium distance
# < 0.02 = Far away
```

### Visualization

Bounding boxes are drawn with:
- **Color intensity** based on depth (brighter = closer)
- **Class label** and confidence score
- **Thin green boxes** for clean appearance

---

## Text Detector

### Status: ⚠️ Currently Disabled

**Reason**: OpenCV threshold error with image format compatibility

### Description

Uses EasyOCR for optical character recognition (when enabled).

### Features

- Multi-language support
- GPU accelerated
- Bounding box detection
- Confidence scores

### Usage (when enabled)

```python
from src.services.detection.text_detector import TextDetector

detector = TextDetector(languages=['en'], confidence_threshold=0.5)
texts, annotated_frame = detector.detect(frame)

for text in texts:
    print(f"Text: {text['text']}")
    print(f"Confidence: {text['confidence']}")
```

### Issue

Current error:
```
OpenCV(4.12.0) error: (-210:Unsupported format or combination of formats)
in function 'cv::threshold'
```

**Workaround**: Text detection is disabled in `assistant.py` until fix is implemented.

---

## Caption Generator

### Status: ⚠️ Disabled for Performance

### Description

Uses BLIP (Bootstrapping Language-Image Pre-training) for image captioning.

### Features

- Context-aware captions
- GPU accelerated
- High accuracy

### Usage (when enabled)

```python
from src.services.detection.caption_generator import CaptionGenerator

generator = CaptionGenerator()
caption, confidence = generator.generate(frame)
print(f"Caption: {caption}")
```

### Performance Impact

- Adds 200-300ms per frame
- Uses ~1GB additional VRAM
- Currently disabled to maintain real-time performance

---

## Code Snippets

### Basic Object Detection

```python
from src.services.detection.object_detector import ObjectDetector
import cv2

# Initialize detector
detector = ObjectDetector()

# Load image
image_path = 'path/to/image.jpg'
frame = cv2.imread(image_path)

# Detect objects
objects, annotated_frame = detector.detect(frame)

# Print all detections
print(f"Found {len(objects)} objects:")
for obj in objects:
    print(f"- {obj['class']}: {obj['confidence']:.2%}")

# Display result
cv2.imshow('Detections', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Real-time Video Detection

```python
from src.services.detection.object_detector import ObjectDetector
from src.services.camera_service import CameraService
import cv2

# Initialize services
detector = ObjectDetector(confidence_threshold=0.30)
camera = CameraService(source="phone", ip_address="192.168.1.100")

camera.start()

try:
    while True:
        # Capture frame
        frame = camera.capture_frame()
        if frame is None:
            continue
        
        # Detect objects
        objects, annotated = detector.detect(frame)
        
        # Display
        cv2.imshow('Real-time Detection', annotated)
        
        # Print detections
        if objects:
            classes = [obj['class'] for obj in objects]
            print(f"Detected: {', '.join(set(classes))}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
```

### Filtering Detections

```python
from src.services.detection.object_detector import ObjectDetector
import cv2

detector = ObjectDetector()
frame = cv2.imread('image.jpg')
objects, annotated = detector.detect(frame)

# Filter by confidence
high_confidence = [obj for obj in objects if obj['confidence'] > 0.7]
print(f"High confidence detections: {len(high_confidence)}")

# Filter by class
people = [obj for obj in objects if obj['class'] == 'person']
print(f"Found {len(people)} people")

# Filter by depth (close objects)
close_objects = [obj for obj in objects if obj['depth_score'] > 0.05]
print(f"Close objects: {[obj['class'] for obj in close_objects]}")

# Filter by position (left side)
left_objects = [obj for obj in objects if obj['position']['x'] < 0.4]
print(f"Objects on left: {[obj['class'] for obj in left_objects]}")
```

### Custom Configuration

```python
from src.services.detection.object_detector import ObjectDetector

# Different model sizes
detector_fast = ObjectDetector(model_name='yolov8n.pt')  # Fastest
detector_balanced = ObjectDetector(model_name='yolov8m.pt')  # Default
detector_accurate = ObjectDetector(model_name='yolov8x.pt')  # Most accurate

# Custom confidence thresholds
detector = ObjectDetector(confidence_threshold=0.40)  # More strict

# Modify class mappings
detector.class_mappings = {
    'bottle': 'container',
    'cup': 'container',
    'bowl': 'container'
}

# Process image
objects, annotated = detector.detect(frame)
```

### Spatial Analysis

```python
from src.services.detection.object_detector import ObjectDetector
import cv2

detector = ObjectDetector()
frame = cv2.imread('scene.jpg')
objects, annotated = detector.detect(frame)

# Analyze spatial relationships
for i, obj1 in enumerate(objects):
    for obj2 in objects[i+1:]:
        # Check if objects are aligned horizontally
        y_diff = abs(obj1['position']['y'] - obj2['position']['y'])
        if y_diff < 0.1:  # Similar y-position
            x_diff = obj1['position']['x'] - obj2['position']['x']
            if x_diff < 0:
                print(f"{obj1['class']} is left of {obj2['class']}")
            else:
                print(f"{obj1['class']} is right of {obj2['class']}")
        
        # Check if objects are aligned vertically
        x_diff = abs(obj1['position']['x'] - obj2['position']['x'])
        if x_diff < 0.1:  # Similar x-position
            y_diff = obj1['position']['y'] - obj2['position']['y']
            if y_diff < 0:
                print(f"{obj1['class']} is above {obj2['class']}")
            else:
                print(f"{obj1['class']} is below {obj2['class']}")
```

### Complete Detection Pipeline

```python
from src.services.detection.object_detector import ObjectDetector
from src.services.narration_service import NarrationService
from src.services.audio.text_to_speech import TextToSpeech
import cv2

def analyze_scene(image_path):
    """Complete scene analysis with narration"""
    
    # Initialize services
    detector = ObjectDetector()
    narration_service = NarrationService()
    tts = TextToSpeech()
    
    # Load and detect
    frame = cv2.imread(image_path)
    objects, annotated_frame = detector.detect(frame)
    
    # Print detection details
    print(f"\n{'='*50}")
    print(f"Scene Analysis: {image_path}")
    print(f"{'='*50}")
    print(f"Total objects detected: {len(objects)}\n")
    
    for obj in objects:
        print(f"Class: {obj['class']}")
        print(f"  Confidence: {obj['confidence']:.2%}")
        print(f"  Depth Score: {obj['depth_score']:.4f}")
        print(f"  Position: x={obj['position']['x']:.2f}, y={obj['position']['y']:.2f}")
        print()
    
    # Generate and speak narration
    if objects:
        narration = narration_service.generate(objects, [])
        print(f"Narration: {narration}")
        tts.speak(narration)
    
    # Save result
    output_path = image_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated image to: {output_path}")
    
    # Display
    cv2.imshow('Scene Analysis', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Cleanup
    tts.cleanup()

# Usage
analyze_scene('test_images/desk.jpg')
```

### Batch Processing

```python
from src.services.detection.object_detector import ObjectDetector
import cv2
import os
from pathlib import Path

detector = ObjectDetector()

# Process all images in a directory
input_dir = Path('input_images')
output_dir = Path('output_images')
output_dir.mkdir(exist_ok=True)

for image_path in input_dir.glob('*.jpg'):
    print(f"Processing: {image_path.name}")
    
    # Detect
    frame = cv2.imread(str(image_path))
    objects, annotated = detector.detect(frame)
    
    # Save result
    output_path = output_dir / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(output_path), annotated)
    
    # Save detection log
    log_path = output_dir / f"{image_path.stem}_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Image: {image_path.name}\n")
        f.write(f"Detections: {len(objects)}\n\n")
        for obj in objects:
            f.write(f"{obj['class']}: {obj['confidence']:.2%}\n")
    
    print(f"  Found {len(objects)} objects")

print("Batch processing complete!")
```

## Error Handling

All detectors handle errors gracefully:

```python
try:
    objects, frame = detector.detect(frame)
except Exception as e:
    print(f"Detection error: {e}")
    # Returns empty list and original frame
    objects = []
```

## Testing

```bash
# Test object detection
python -m tests.run_image_test

# Test with sample image
python -c "
from src.services.detection.object_detector import ObjectDetector
import cv2
detector = ObjectDetector()
frame = cv2.imread('tests/test_data/test_image.jpg')
objects, _ = detector.detect(frame)
print(f'Detected {len(objects)} objects')
"
```

## Dependencies

- `torch` - Deep learning framework
- `ultralytics` - YOLOv8 implementation
- `easyocr` - Text recognition
- `transformers` - BLIP model
- `opencv-python` - Image processing
- `PIL` - Image handling

## Future Improvements

- [ ] Fix text detector compatibility issue
- [ ] Enable caption generator with optimization
- [ ] Add custom object classes
- [ ] Implement object tracking
- [ ] Add distance measurement
- [ ] Support for video input
- [ ] Batch processing optimization

## Related Documentation

- [Services Overview](../README.md)
- [Core Module](../../core/README.md)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
