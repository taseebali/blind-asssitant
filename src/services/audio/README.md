# Audio Services

Text-to-speech and speech recognition services for audio interaction.

## Overview

The audio services module provides:

1. **Text-to-Speech** (gTTS) - Convert text to natural speech ✅
2. **Speech Recognizer** (Whisper) - Convert speech to text 🚧 Not implemented

## Text-to-Speech (gTTS)

### Description

Converts text to natural-sounding speech using Google Text-to-Speech API.

### Features

- **Natural Voice**: Human-like speech synthesis
- **Queue-based Playback**: Non-blocking audio playback
- **Automatic Cleanup**: Manages temporary audio files
- **Fast Synthesis**: ~500-1000ms latency
- **Multiple Languages**: Supports 100+ languages

### Usage

#### Basic Usage

```python
from src.services.audio.text_to_speech import TextToSpeech

# Initialize TTS
tts = TextToSpeech()

# Speak text
tts.speak("Hello, I can see a keyboard on your desk")

# Clean up when done
tts.cleanup()
```

#### Queue-based Usage

```python
# Multiple narrations are queued automatically
tts.speak("First message")
tts.speak("Second message")
tts.speak("Third message")

# Only the latest message will be spoken (old ones are cleared)
```

#### Context Manager

```python
from src.services.audio.text_to_speech import TextToSpeech

with TextToSpeech() as tts:
    tts.speak("Hello, world!")
    # Automatically cleaned up
```

### Implementation Details

#### Architecture

```python
TextToSpeech
├── __init__()           # Initialize pygame mixer and audio thread
├── speak(text)          # Add text to speech queue
├── _process_audio_queue() # Background thread for processing
├── _play_audio(text)    # Convert and play audio
└── cleanup()            # Stop threads and clean up
```

#### Threading Model

- **Main Thread**: Application logic
- **Audio Thread**: Background thread for processing audio queue
- **Queue**: Thread-safe communication

```python
# Audio processing flow
text → audio_queue → _process_audio_queue() → _play_audio() → speakers
```

### Configuration

```python
from src.services.audio.text_to_speech import TextToSpeech
from gtts import gTTS

# Initialize with custom settings
tts = TextToSpeech()

# Language settings (modify in _play_audio method)
# English: 'en'
# Spanish: 'es'
# French: 'fr'
# German: 'de'
# ... 100+ languages supported

# Speed settings
# slow=False (default) - Normal speed
# slow=True - Slower speed for better clarity
```

### Audio File Management

Temporary audio files are automatically managed:

```python
# Unique filename for each narration
temp_file = f"temp_audio_{int(time.time()*1000)}.mp3"

# Automatic cleanup after playback
pygame.mixer.music.unload()
os.remove(temp_file)
```

### Performance

- **Synthesis Time**: 500-1000ms (requires internet)
- **File Size**: ~50-200KB per narration
- **Memory**: ~10MB overhead
- **Network**: Requires internet connection

### Error Handling

```python
try:
    tts.speak("Test message")
except Exception as e:
    print(f"TTS error: {e}")
    # Continue without audio output
```

Common errors:
- **No internet**: Cannot synthesize speech
- **Audio device unavailable**: Check speakers/headphones
- **File access error**: Temporary file conflicts

### Integration Example

```python
from src.services.detection.object_detector import ObjectDetector
from src.services.narration_service import NarrationService
from src.services.audio.text_to_speech import TextToSpeech
import cv2

# Initialize services
detector = ObjectDetector()
narration_service = NarrationService()
tts = TextToSpeech()

# Process frame
frame = cv2.imread('scene.jpg')
objects, _ = detector.detect(frame)

# Generate and speak narration
narration = narration_service.generate(objects, [])
if narration:
    tts.speak(narration)

# Cleanup
tts.cleanup()
```

---

## Speech Recognizer (Whisper)

### Status: 🚧 Not Implemented

### Description

Will use OpenAI's Whisper model for speech-to-text conversion.

### Planned Features

- Voice command recognition
- Multiple language support
- Offline operation
- GPU acceleration
- Real-time transcription

### Planned Usage

```python
from src.services.audio.speech_recognizer import SpeechRecognizer

# Initialize recognizer
recognizer = SpeechRecognizer()

# Listen for commands
command = recognizer.listen()
print(f"Command: {command}")

# Process commands
if "what do you see" in command:
    # Trigger object detection
    pass
elif "read text" in command:
    # Trigger text detection
    pass
```

### Implementation Roadmap

1. [ ] Add Whisper model integration
2. [ ] Implement microphone input
3. [ ] Add command parsing
4. [ ] Add wake word detection
5. [ ] Add continuous listening mode
6. [ ] Add noise cancellation

---

## Dependencies

### Text-to-Speech
- `gtts` - Google Text-to-Speech
- `pygame` - Audio playback
- `os` - File management
- `time` - Timing
- `threading` - Background processing
- `queue` - Thread-safe communication

### Speech Recognizer (planned)
- `openai-whisper` - Speech recognition
- `pyaudio` - Microphone input
- `torch` - GPU acceleration

## Installation

```bash
# Install TTS dependencies
pip install gtts pygame

# Install speech recognition dependencies (future)
pip install openai-whisper pyaudio
```

## Testing

## Code Snippets

### Basic Text-to-Speech

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

# Initialize TTS
tts = TextToSpeech()

# Speak a simple message
tts.speak("Hello, welcome to Blind Assistant")

# Wait for speech to complete
time.sleep(3)

# Cleanup
tts.cleanup()
```

### Real-time Scene Narration

```python
from src.services.audio.text_to_speech import TextToSpeech
from src.services.detection.object_detector import ObjectDetector
from src.services.narration_service import NarrationService
from src.services.camera_service import CameraService
import time

# Initialize services
tts = TextToSpeech()
detector = ObjectDetector()
narration_service = NarrationService()
camera = CameraService(source="phone", ip_address="192.168.1.100")

camera.start()

try:
    while True:
        # Capture and detect
        frame = camera.capture_frame()
        if frame is None:
            continue
        
        objects, _ = detector.detect(frame)
        
        # Generate and speak narration
        if objects:
            narration = narration_service.generate(objects, [])
            tts.speak(narration)  # Old narrations automatically cleared
        
        # Wait 2 seconds before next narration
        time.sleep(2)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    camera.stop()
    tts.cleanup()
```

### Queue Management Demo

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

tts = TextToSpeech()

# Rapidly queue multiple messages
tts.speak("First message")
time.sleep(0.1)
tts.speak("Second message")
time.sleep(0.1)
tts.speak("Third message")
time.sleep(0.1)
tts.speak("Final message")

# Only "Final message" will be spoken (others are cleared)
# This ensures you always hear the latest information

time.sleep(3)
tts.cleanup()
```

### Context Manager Usage

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

# Automatic cleanup when done
with TextToSpeech() as tts:
    tts.speak("This is the first message")
    time.sleep(2)
    tts.speak("This is the second message")
    time.sleep(2)
# Cleanup happens automatically here
```

### Error Handling

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

tts = TextToSpeech()

def speak_safely(text):
    """Speak with error handling"""
    try:
        tts.speak(text)
        return True
    except Exception as e:
        print(f"TTS error: {e}")
        return False

# Try to speak
if speak_safely("Testing text to speech"):
    print("Speech successful")
    time.sleep(3)
else:
    print("Speech failed (check internet connection)")

tts.cleanup()
```

### Interactive Narration Demo

```python
from src.services.audio.text_to_speech import TextToSpeech
import time

tts = TextToSpeech()

# Simulate interactive assistant
narrations = [
    "Welcome to Blind Assistant",
    "I will help you understand your surroundings",
    "Point your camera at objects",
    "I will describe what I see",
    "Press Q to quit"
]

print("Starting narration demo...")
for narration in narrations:
    print(f"Speaking: {narration}")
    tts.speak(narration)
    time.sleep(3)  # Wait for speech to complete

print("Demo complete")
tts.cleanup()
```

### Test TTS with Detailed Output

```python
from src.services.audio.text_to_speech import TextToSpeech
import time
import threading

def test_tts():
    """Comprehensive TTS test"""
    
    tts = TextToSpeech()
    
    print("=" * 50)
    print("Text-to-Speech Test")
    print("=" * 50)
    
    # Test 1: Basic speech
    print("\nTest 1: Basic speech")
    tts.speak("This is a basic test message")
    time.sleep(3)
    print("✓ Basic speech completed")
    
    # Test 2: Queue handling
    print("\nTest 2: Queue handling (rapid messages)")
    for i in range(5):
        tts.speak(f"Message number {i+1}")
        time.sleep(0.2)
    print("✓ Only the last message should be spoken")
    time.sleep(3)
    
    # Test 3: Long message
    print("\nTest 3: Long message")
    long_message = "I can see a keyboard on the left side of the desk, with a mouse to the right of it, and a monitor in the center."
    tts.speak(long_message)
    time.sleep(5)
    print("✓ Long message completed")
    
    # Test 4: Special characters
    print("\nTest 4: Special characters")
    tts.speak("Testing numbers: 1, 2, 3. And symbols: @ # $")
    time.sleep(3)
    print("✓ Special characters handled")
    
    # Cleanup
    tts.cleanup()
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    test_tts()
```

### Integration with Object Detection

```python
from src.services.audio.text_to_speech import TextToSpeech
from src.services.detection.object_detector import ObjectDetector
import cv2
import time

def describe_image(image_path):
    """Detect objects and speak description"""
    
    tts = TextToSpeech()
    detector = ObjectDetector()
    
    # Load and detect
    frame = cv2.imread(image_path)
    objects, annotated = detector.detect(frame)
    
    if not objects:
        tts.speak("I don't see any objects")
    else:
        # Count objects
        object_counts = {}
        for obj in objects:
            class_name = obj['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Generate speech
        if len(object_counts) == 1:
            class_name, count = list(object_counts.items())[0]
            if count == 1:
                message = f"I can see a {class_name}"
            else:
                message = f"I can see {count} {class_name}s"
        else:
            items = [f"{count} {name}" if count > 1 else f"a {name}" 
                    for name, count in object_counts.items()]
            message = f"I can see {', '.join(items)}"
        
        # Speak
        tts.speak(message)
        print(f"Narration: {message}")
    
    # Display
    cv2.imshow('Detection', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    time.sleep(3)
    tts.cleanup()

# Usage
describe_image('test_image.jpg')
```

### Test Audio Output

```bash
# Quick test
python -c "from src.services.audio.text_to_speech import TextToSpeech; tts = TextToSpeech(); tts.speak('Hello world'); import time; time.sleep(3)"
```

## Troubleshooting

### No Audio Output

**Problem**: TTS runs but no sound
```bash
# Check audio device
# Windows: Check Volume Mixer
# Linux: alsamixer
# Mac: System Preferences → Sound

# Check pygame mixer
python -c "import pygame; pygame.mixer.init(); print('Audio initialized')"
```

### Internet Connection Required

**Problem**: gTTS requires internet
```bash
# Solution: Check internet connection
ping google.com

# Future: Will add offline TTS option
```

### Slow Synthesis

**Problem**: TTS takes too long
```bash
# Solution: Check internet speed
# gTTS requires API call to Google servers
# Typical latency: 500-1000ms

# Future: Will add local TTS for offline use
```

## Future Enhancements

### Text-to-Speech
- [ ] Add offline TTS option (pyttsx3)
- [ ] Add voice customization
- [ ] Add volume control
- [ ] Add playback speed control
- [ ] Add audio effects
- [ ] Support multiple voices

### Speech Recognizer
- [ ] Implement Whisper integration
- [ ] Add microphone input
- [ ] Add command parsing
- [ ] Add wake word ("Hey Assistant")
- [ ] Add continuous listening
- [ ] Add noise cancellation
- [ ] Support multiple languages

## Related Documentation

- [Services Overview](../README.md)
- [Core Module](../../core/README.md)
- [gTTS Documentation](https://gtts.readthedocs.io/)
- [Pygame Documentation](https://www.pygame.org/docs/)
