# Core Module

This module contains the main orchestration logic for the Blind Assistant application.

## Overview

The core module is responsible for:
- Coordinating all AI services
- Managing multi-threaded processing pipeline
- Handling inter-service communication
- Managing application lifecycle

## Components

### `assistant.py`

The main `BlindAssistant` class that orchestrates the entire application.

#### Key Features

- **Multi-threaded Architecture**: Separate threads for camera capture, frame processing, audio output, and command handling
- **Queue-based Communication**: Uses Python queues for thread-safe data exchange
- **Real-time Processing**: Optimized for low-latency object detection and narration
- **GPU Acceleration**: Leverages CUDA for all AI models

#### Architecture

```python
BlindAssistant
├── Camera Thread (_capture_frames)
│   └── Continuously captures frames from IP Webcam
├── Processing Thread (_process_frames)
│   ├── Runs object detection (YOLOv8)
│   ├── Runs text detection (EasyOCR - disabled)
│   ├── Runs image captioning (BLIP - disabled)
│   └── Generates narrations
├── Audio Thread (_handle_audio)
│   └── Converts narrations to speech
└── Command Thread (_handle_commands)
    └── Listens for voice commands (not implemented)
```

## Usage

### Basic Usage

```python
from src.core.assistant import BlindAssistant

# Initialize with display and camera IP
assistant = BlindAssistant(
    show_display=True,
    camera_ip="192.168.1.100"
)

# Start the assistant
assistant.start()
```

### Advanced Configuration

```python
from src.core.assistant import BlindAssistant

# Headless mode (no visual display)
assistant = BlindAssistant(
    show_display=False,
    camera_ip="192.168.1.100"
)

# Custom frame queue size
assistant.frame_queue = Queue(maxsize=5)

# Start
assistant.start()
```

## Class Methods

### `__init__(show_display=True, camera_ip=None)`

Initialize the Blind Assistant.

**Parameters:**
- `show_display` (bool): Whether to show the visual display window
- `camera_ip` (str): IP address of the phone running IP Webcam app

**Example:**
```python
assistant = BlindAssistant(show_display=True, camera_ip="192.168.1.100")
```

### `start()`

Start the assistant and begin processing frames.

**Behavior:**
- Starts all worker threads
- Begins camera capture
- Opens display window (if enabled)
- Blocks until Ctrl+C or 'Q' key pressed

**Example:**
```python
try:
    assistant.start()
except KeyboardInterrupt:
    print("Stopping...")
```

### `cleanup()`

Clean up resources and stop all threads.

**Behavior:**
- Stops all worker threads
- Releases camera resources
- Closes display windows
- Cleans up audio files

**Example:**
```python
try:
    assistant.start()
finally:
    assistant.cleanup()
```

## Threading Model

### Frame Queue
- **Purpose**: Buffer frames between camera and processing threads
- **Size**: 2 frames (configurable)
- **Behavior**: Drops old frames when full to maintain real-time processing

### Narration Queue
- **Purpose**: Buffer narrations for text-to-speech
- **Behavior**: Clears old narrations to always speak the latest

### Command Queue
- **Purpose**: Buffer voice commands (not currently used)

## Performance Considerations

### Frame Skipping
- Processes every 3rd frame by default
- Configurable via `frame_skip` attribute
- Trade-off between responsiveness and accuracy

### Scene Change Detection
- Tracks object positions to detect scene changes
- Only generates narrations when scene changes
- Minimum 2-second interval between narrations

### GPU Optimization
- Uses CUDA mixed precision (`torch.amp.autocast`)
- All models loaded on GPU
- Optimized memory management

## Configuration

### Timing Parameters

```python
# In _process_frames method
narration_interval = 2  # Seconds between narrations
frame_skip = 3          # Process every Nth frame
```

### Queue Sizes

```python
self.frame_queue = Queue(maxsize=2)      # Frame buffer
self.narration_queue = Queue()           # Text to speak
self.command_queue = Queue()             # Voice commands
```

## Error Handling

The assistant handles various error conditions:

- **Camera disconnection**: Automatic reconnection
- **Model inference errors**: Logs and continues
- **Audio playback errors**: Skips failed narrations
- **Thread exceptions**: Graceful degradation

## Dependencies

- `cv2` (OpenCV): Video capture and display
- `torch`: GPU acceleration
- `threading`: Multi-threaded processing
- `queue`: Thread-safe communication
- All service modules from `src.services`

## Code Snippets

### Complete Working Example

```python
from src.core.assistant import BlindAssistant

def main():
    # Get camera IP from user
    camera_ip = input("Enter phone IP address (e.g., 192.168.1.100): ")
    
    # Initialize assistant
    assistant = BlindAssistant(
        show_display=True,  # Show video feed with detections
        camera_ip=camera_ip
    )
    
    print("Starting Blind Assistant...")
    print("Press 'Q' or Ctrl+C to stop")
    
    try:
        assistant.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        assistant.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
```

### Headless Mode (No Display)

```python
from src.core.assistant import BlindAssistant

# Perfect for deployment on devices without screens
assistant = BlindAssistant(
    show_display=False,  # No visual display
    camera_ip="192.168.1.100"
)

try:
    print("Running in headless mode...")
    assistant.start()
except KeyboardInterrupt:
    assistant.cleanup()
```

### Custom Configuration

```python
from src.core.assistant import BlindAssistant
from queue import Queue

# Initialize with custom settings
assistant = BlindAssistant(
    show_display=True,
    camera_ip="192.168.1.100"
)

# Customize frame processing
assistant.frame_skip = 5  # Process every 5th frame (faster but less responsive)

# Customize queue sizes
assistant.frame_queue = Queue(maxsize=5)  # Larger buffer for smoother processing

# Start
assistant.start()
```

### Accessing Internal State

```python
from src.core.assistant import BlindAssistant
import threading
import time

assistant = BlindAssistant(show_display=True, camera_ip="192.168.1.100")

# Custom monitoring thread
def monitor_performance():
    while assistant.running:
        queue_size = assistant.frame_queue.qsize()
        print(f"Frame queue size: {queue_size}")
        time.sleep(1)

# Start monitoring
monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
monitor_thread.start()

# Start assistant
try:
    assistant.start()
finally:
    assistant.cleanup()
```

### Processing Pipeline Overview

```python
# This is what happens internally:

# 1. Camera Thread captures frames
frame = camera_service.capture_frame()
frame_queue.put(frame)  # Non-blocking

# 2. Processing Thread processes frames
frame = frame_queue.get(timeout=1)
objects, annotated = detector.detect(frame)
narration = narration_service.generate(objects, [])
narration_queue.put(narration)

# 3. Audio Thread speaks narrations
text = narration_queue.get(timeout=1)
tts.speak(text)

# 4. Display Thread shows annotated frames
cv2.imshow('Blind Assistant', annotated_frame)
```

## Future Improvements

- [ ] Voice command integration
- [ ] Configuration file support
- [ ] Dynamic model switching
- [ ] Performance profiling
- [ ] Better error recovery
- [ ] Recording/playback mode
