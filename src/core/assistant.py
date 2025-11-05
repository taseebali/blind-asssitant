"""
Main assistant class that coordinates all services
"""

import cv2
import time
import threading
import torch
from queue import Queue, Empty
from typing import Optional, Dict, List

from src.services.camera_service import CameraService
from src.services.detection.object_detector import ObjectDetector
from src.services.detection.text_detector import TextDetector
from src.services.detection.caption_generator import CaptionGenerator
from src.services.audio.text_to_speech import TextToSpeech
from src.services.audio.speech_recognizer import SpeechRecognizer
from src.services.narration_service import NarrationService

class BlindAssistant:
    def __init__(self, show_display=True, camera_ip=None):
        """Initialize the Blind Assistant.
        
        Args:
            show_display (bool): Whether to show the visual display
            camera_ip (str): IP address of the phone running IP Webcam
        """
        self.show_display = show_display
        
        # Initialize queues for inter-thread communication
        self.frame_queue = Queue(maxsize=2)  # Frame buffer
        self.narration_queue = Queue()       # Text to be spoken
        self.command_queue = Queue()         # Voice commands
        
        # Initialize services
        print("Initializing services...")
        self.camera = CameraService(source="phone", ip_address=camera_ip)
        self.object_detector = ObjectDetector()
        self.text_detector = TextDetector()
        self.caption_generator = CaptionGenerator()
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        self.narration_service = NarrationService()
        
        # Thread control
        self.running = False
        self.threads = []
        
        # Frame processing control
        self.frame_skip = 2  # Process every 2nd frame
        self.max_fps = 30
        self.last_process_time = 0
        self.process_interval = 1.0 / self.max_fps

    def start(self):
        """Start the assistant and begin processing frames."""
        print("Starting Blind Assistant...")
        self.running = True
        
        # Start worker threads
        self.threads = [
            threading.Thread(target=self._capture_frames),
            threading.Thread(target=self._process_frames),
            threading.Thread(target=self._handle_audio),
            threading.Thread(target=self._handle_commands)
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        # Main display loop if show_display is enabled
        try:
            if self.show_display:
                self._display_loop()
            else:
                # Wait for Ctrl+C
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping Blind Assistant...")
        finally:
            self.cleanup()

    def _capture_frames(self):
        """Continuously capture frames from camera."""
        self.camera.start()
        while self.running:
            frame = self.camera.capture_frame()
            if frame is not None:
                # Skip frames if queue is full
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put((frame, time.time()))

    def _process_frames(self):
        """Process frames from the camera in real-time."""
        frame_count = 0
        start_time = time.time()
        last_narration_time = 0
        narration_interval = 2  # Generate narration every 2 seconds max (faster updates)
        previous_objects = set()
        
        while self.running:
            try:
                # Skip frames if we're falling behind
                if not self.frame_queue.empty():
                    while self.frame_queue.qsize() > 1:
                        _ = self.frame_queue.get_nowait()

                frame, timestamp = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    continue

                # Process every 3rd frame for balance between speed and accuracy
                frame_count += 1
                if frame_count % 3 != 0:
                    continue

                current_time = time.time()

                # Process frame with AI models
                with torch.amp.autocast('cuda'):
                    try:
                        objects, frame = self.object_detector.detect(frame.copy())
                        
                        # Temporarily disable text detection due to threshold errors
                        # texts, frame = self.text_detector.detect(frame)
                        texts = []  # Empty for now
                    except Exception as e:
                        print(f"Error in model inference: {str(e)}")
                        continue

                # Check if scene has changed significantly
                current_objects = set((obj['class'], 
                                      int(obj['position']['x'] * 10), 
                                      int(obj['position']['y'] * 10)) 
                                     for obj in objects)
                
                # More sensitive scene change detection
                scene_changed = len(current_objects.symmetric_difference(previous_objects)) > 0
                time_for_narration = (current_time - last_narration_time) >= narration_interval
                
                # Generate narration if scene changed OR enough time passed with objects present
                if objects and (scene_changed or time_for_narration):
                    narration = self.narration_service.generate(objects, texts)
                    if narration:
                        self.narration_queue.put(narration)
                        last_narration_time = current_time
                        previous_objects = current_objects
                        print(f"[{current_time - start_time:.1f}s] {narration}")

                # Update FPS calculation
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    print(f"Processing rate: {fps:.1f} FPS")
                    start_time = time.time()

                # Update display frame
                if self.show_display:
                    # Clear old frames from display queue
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            break
                    self.frame_queue.put((frame, timestamp))

                self.last_process_time = current_time

            except Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    def _handle_audio(self):
        """Handle text-to-speech output."""
        last_narration = None
        while self.running:
            try:
                # Clear ALL old narrations except the most recent one
                while self.narration_queue.qsize() > 1:
                    _ = self.narration_queue.get_nowait()
                
                narration = self.narration_queue.get(timeout=1.0)
                # Skip if this is a duplicate of the last narration
                if narration != last_narration:
                    print(f"Speaking: {narration}")
                    self.tts.speak(narration)
                    last_narration = narration
            except Empty:
                continue

    def _handle_commands(self):
        """Listen for and process voice commands."""
        while self.running:
            command = self.speech_recognizer.listen()
            if command:
                self.command_queue.put(command)

    def _display_loop(self):
        """Display processed frames if show_display is enabled."""
        while self.running:
            try:
                frame, _ = self.frame_queue.get(timeout=1.0)
                cv2.imshow('Blind Assistant', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            except Empty:
                continue

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Release resources
        self.camera.release()
        cv2.destroyAllWindows()