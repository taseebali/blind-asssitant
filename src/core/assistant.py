"""
Main assistant class that coordinates all services
"""

import cv2
import time
import threading
import torch
import numpy as np
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
    def __init__(self, show_display=False, camera_ip=None):
        """Initialize the Blind Assistant.
        
        Args:
            show_display (bool): Whether to show the visual display
            camera_ip (str): IP address of the phone running IP Webcam
        """
        self.show_display = show_display
        
        # Initialize queues with minimal buffering
        self.frame_queue = Queue(maxsize=1)      # Reduce buffer size
        self.narration_queue = Queue(maxsize=2)  # Limit narration buffer
        self.command_queue = Queue()
        
        # Initialize services
        print("Initializing services...")
        self.camera = CameraService(source="phone", ip_address=camera_ip)
        self.object_detector = ObjectDetector()
        self.text_detector = TextDetector()
        self.caption_generator = CaptionGenerator()
        self.narration_service = NarrationService()
        self.tts = TextToSpeech()
        
        # Processing control
        self.running = False
        self.process_interval = 0.2  # 5 FPS target for processing
        self.last_process_time = 0

    def start(self):
        """Start the assistant with minimal threads."""
        print("Starting Blind Assistant...")
        self.running = True
        self.camera.start()
        
        # Start only essential threads
        self.threads = [
            threading.Thread(target=self._process_frames),
            threading.Thread(target=self._handle_audio)
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping Blind Assistant...")
        finally:
            self._process_remaining_narrations()
            self.cleanup()

    def _process_frames(self):
        """Process frames with minimal overhead."""
        frame_count = 0
        start_time = time.time()
        
        # Initialize display if needed
        if self.show_display:
            cv2.namedWindow('Blind Assistant', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Blind Assistant', 640, 480)  # Smaller window for performance
            
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    time.sleep(0.01)
                    continue
                
                # Get frame from camera
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Process frame
                objects, annotated_frame = self.object_detector.detect(frame)
                
                # Show frame if display is enabled
                if self.show_display and annotated_frame is not None:
                    # Add FPS counter
                    fps = frame_count / (current_time - start_time) if current_time > start_time else 0
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cv2.imshow('Blind Assistant', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                # Generate narration if objects detected
                if objects:
                    narration = self.narration_service.generate(
                        objects=objects,
                        texts=[],
                        caption=None  # Disable caption for speed
                    )
                    
                    if narration:
                        if not self.narration_queue.full():
                            self.narration_queue.put(narration)
                
                self.last_process_time = current_time
                frame_count += 1
                
                # Log FPS every 5 seconds
                if frame_count % 25 == 0:
                    fps = frame_count / (current_time - start_time)
                    print(f"Processing rate: {fps:.1f} FPS")
                    frame_count = 0
                    start_time = current_time
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

    def _handle_audio(self):
        """Handle text-to-speech output with improved deduplication."""
        last_narrations = []  # Keep track of last N narrations
        max_history = 3  # Number of previous narrations to remember
        min_narration_interval = 2.0  # Minimum seconds between narrations
        last_narration_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Wait for minimum interval between narrations
                if current_time - last_narration_time < min_narration_interval:
                    time.sleep(0.1)
                    continue
                
                # Clear old narrations if queue is backing up
                while self.narration_queue.qsize() > 2:
                    _ = self.narration_queue.get_nowait()
                
                narration = self.narration_queue.get(timeout=1.0)
                
                # Skip if this narration is too similar to recent ones
                if any(self._similar_narration(narration, prev) for prev in last_narrations):
                    continue
                
                # Update narration history
                last_narrations.append(narration)
                if len(last_narrations) > max_history:
                    last_narrations.pop(0)
                
                print(f"Speaking narration: {narration}")
                self.tts.speak(narration)
                last_narration_time = time.time()
                
            except Empty:
                continue
                
    def _similar_narration(self, narr1: str, narr2: str) -> bool:
        """Check if two narrations are similar enough to be considered duplicates."""
        # Convert to sets of words for comparison
        words1 = set(narr1.lower().split())
        words2 = set(narr2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > 0.7  # 70% similarity threshold

    def _handle_commands(self):
        """Listen for and process voice commands."""
        while self.running:
            command = self.speech_recognizer.listen()
            if command:
                self.command_queue.put(command)

    def _process_remaining_narrations(self):
        """Process any remaining narrations in the queue before shutdown."""
        print("Processing remaining narrations...")
        try:
            while not self.narration_queue.empty():
                narration = self.narration_queue.get_nowait()
                print(f"Final narration: {narration}")
                self.tts.speak(narration)
        except Empty:
            pass
            
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        
        # Close display windows
        if self.show_display:
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Give time for windows to close
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Release resources
        if self.camera:
            self.camera.release()
            
        # Close all windows
        if self.show_display:
            cv2.destroyAllWindows()
            # Wait a bit to ensure windows are closed
            time.sleep(0.5)