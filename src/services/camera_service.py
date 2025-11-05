"""
Camera service with GPU acceleration
"""

import cv2
import time
import requests
import numpy as np
import torch
from typing import Optional, Tuple
from urllib.parse import urljoin

class CameraService:
    def __init__(self, source: str = "phone", ip_address: str = None, port: str = "8080"):
        """Initialize camera service.
        
        Args:
            source: "phone" for IP Webcam, or camera index (0, 1, etc.)
            ip_address: IP address of the phone running IP Webcam
            port: Port number (default: 8080 for IP Webcam)
        """
        self.source = source
        self.ip_address = ip_address
        self.port = port
        self.cap = None
        self.base_url = None
        
        # Frame management
        self.frame_count = 0
        self.last_processed_frame = 0
        self.last_capture_time = 0
        self.target_fps = 15  # Lower target FPS
        self.process_interval = 1.0 / 10  # Process 10 FPS
        self.frame_interval = 1.0 / self.target_fps
        
        # CUDA stream for async operations
        if torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
        
        # Frame dropping configuration
        self.drop_threshold = 3  # Drop frames if we're this many frames behind
        self.max_queue_size = 2  # Maximum frames to keep in queue
        
        # Performance optimization
        self.target_resolution = (480, 360)  # Smaller resolution for faster processing
        self.last_frame = None
        self.change_threshold = 0.15  # Increased threshold for less sensitivity
        self.min_change_interval = 0.3  # 300ms between significant changes
        self.blur_size = (5, 5)  # Larger blur for more stable detection
        self.pixel_threshold = 35  # Less sensitive change detection
        
        # Frame skipping for pygame display
        self.display_interval = 0.1  # 10 FPS display refresh
        self.last_display_time = 0
        
        # GPU setup for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True
        else:
            print("GPU not available, using CPU")

    def start(self):
        """Start the camera capture with improved stream handling."""
        if self.source == "phone":
            # Use snapshot URL instead of video stream for more reliable operation
            self.base_url = f"http://{self.ip_address}:{self.port}/shot.jpg"
            
            # Test connection before proceeding
            try:
                response = requests.get(self.base_url, timeout=5)
                response.raise_for_status()
                print("Successfully connected to IP camera")
                
                # Get initial frame to verify dimensions
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    height, width = frame.shape[:2]
                    print(f"IP Camera initialized at {width}x{height}")
                self.cap = None  # No VideoCapture needed for IP camera
                
            except Exception as e:
                print(f"Failed to connect to IP camera: {e}")
                print("Falling back to default camera")
                self.source = 0  # Fall back to default camera
                self.cap = cv2.VideoCapture(self.source)
                
        if self.source != "phone":
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
                
            # Set optimal resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better processing
            
            # Print actual camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps}fps")
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame with optimized timing and frame dropping."""
        current_time = time.time()
        
        # Check if we should capture a new frame based on target FPS
        if current_time - self.last_capture_time < self.frame_interval:
            return None
            
        # Check display timing for pygame
        is_display_frame = current_time - self.last_display_time >= self.display_interval
        
        # Update timestamps
        self.last_capture_time = current_time
        if is_display_frame:
            self.last_display_time = current_time
            
        self.frame_count += 1
        
        # Aggressive frame dropping when falling behind
        frames_behind = self.frame_count - self.last_processed_frame
        if frames_behind > self.drop_threshold:
            self.last_processed_frame = self.frame_count - (self.drop_threshold // 2)
            if not is_display_frame:  # Still allow display frames through
                return None
            
        frame = None
        
        if self.source == "phone":
            try:
                # Fast capture for IP camera
                response = requests.get(self.base_url, timeout=0.5)
                if response.status_code == 200:
                    # Efficient image decoding
                    image_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except requests.exceptions.RequestException:
                return None
                
        elif self.source == "webcam":
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    return None
                    
        if frame is None:
            return None
            
        # Resize for better performance
        try:
            frame = cv2.resize(frame, self.target_resolution)
            return frame
        except Exception:
            return None
            
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process frame with GPU acceleration when available."""
        if frame is None:
            return None

        try:
            if self.use_gpu:
                # Upload frame to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)

                # Resize on GPU
                gpu_frame = cv2.cuda.resize(gpu_frame, self.target_resolution)

                # Convert to grayscale on GPU for change detection
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

                if self._has_significant_change_gpu(gpu_gray):
                    # Enhancement if needed
                    if self._needs_enhancement_gpu(gpu_gray):
                        gpu_frame = self._enhance_frame_gpu(gpu_frame)

                    # Download result back to CPU
                    result = gpu_frame.download()
                    return result
                return None
            else:
                # Fallback to CPU processing
                return super().process_frame(frame)

        except Exception as e:
            print(f"GPU processing error: {e}, falling back to CPU")
            return super().process_frame(frame)

    def _needs_enhancement(self, frame: np.ndarray) -> bool:
        """Check if frame needs contrast enhancement."""
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        return mean_brightness < 100  # Only enhance dark frames
    
    def _has_significant_change(self, frame: np.ndarray) -> bool:
        """
        Detect if there's a significant change in the frame compared to the last frame.
        Optimized for speed and stability.
        """
        current_time = time.time()
        
        # Always process if this is the first frame
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.last_significant_change = current_time
            return True
            
        # Check if minimum time has elapsed since last change
        if current_time - self.last_significant_change < self.min_change_interval:
            return False
            
        # Convert numpy arrays to PyTorch tensors on GPU
        with torch.no_grad():
            current_tensor = torch.from_numpy(frame).to(self.device)
            last_tensor = torch.from_numpy(self.last_frame).to(self.device)
            
            # Convert to grayscale using GPU
            current_gray = 0.299 * current_tensor[:,:,0] + 0.587 * current_tensor[:,:,1] + 0.114 * current_tensor[:,:,2]
            last_gray = 0.299 * last_tensor[:,:,0] + 0.587 * last_tensor[:,:,1] + 0.114 * last_tensor[:,:,2]
            
            # Compute difference on GPU
            diff = torch.abs(current_gray - last_gray)
            mad = torch.mean(diff).item()
            
            if mad > self.change_threshold:
                self.last_frame = frame.copy()
                self.last_significant_change = current_time
                return True
                
        return False
    
    def _has_significant_change_gpu(self, gpu_gray: cv2.cuda_GpuMat) -> bool:
        """GPU-accelerated change detection."""
        if not hasattr(self, 'last_gpu_frame'):
            self.last_gpu_frame = gpu_gray.clone()
            return True

        current_time = time.time()
        if current_time - getattr(self, 'last_significant_change', 0) < self.min_change_interval:
            return False

        # Compute difference on GPU
        gpu_diff = cv2.cuda.absdiff(gpu_gray, self.last_gpu_frame)
        
        # Use GPU stream for async operations
        with self.cuda_stream:
            # Blur on GPU
            gpu_diff = cv2.cuda.GaussianBlur(gpu_diff, self.blur_size, 0)
            
            # Get mean difference using GPU reduction
            mean_diff = cv2.cuda.reduce(gpu_diff, 0, cv2.REDUCE_AVG).download()[0]

        if mean_diff > self.change_threshold:
            self.last_gpu_frame = gpu_gray.clone()
            self.last_significant_change = current_time
            return True
        return False

    def _enhance_frame_gpu(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        """GPU-accelerated frame enhancement."""
        # Convert to LAB color space on GPU
        gpu_lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2Lab)
        
        # Extract L channel
        gpu_channels = cv2.cuda.split(gpu_lab)
        
        # Apply CLAHE on GPU
        gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0)
        gpu_channels[0] = gpu_clahe.apply(gpu_channels[0], self.cuda_stream)
        
        # Merge channels back
        gpu_lab = cv2.cuda.merge(gpu_channels)
        
        # Convert back to BGR
        return cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_Lab2BGR)

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()