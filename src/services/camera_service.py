"""
Camera service for handling video capture operations
"""

import cv2
import time
import requests
import numpy as np
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
        
        self.frame_buffer_size = 4
        self.target_resolution = (640, 480)  # Reduced resolution for speed
        
    def start(self):
        """Start the camera capture."""
        if self.source == "phone":
            # Add performance-related parameters to URL for low latency
            params = {
                "quality": "40",  # Lower quality for faster streaming
                "fps": "30",     # Target 30 FPS
                "size": f"{self.target_resolution[0]}x{self.target_resolution[1]}"
            }
            url_params = "&".join([f"{k}={v}" for k, v in params.items()])
            self.base_url = f"http://{self.ip_address}:{self.port}/video?{url_params}"
            
            # Configure OpenCV capture with optimized parameters for minimal latency
            self.cap = cv2.VideoCapture(self.base_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream")
            
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
        """Capture a single frame."""
        if self.cap is None or not self.cap.isOpened():
            return None
            
        try:
            # Read frame directly without clearing buffer for faster response
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Try to reconnect if using IP Webcam
                if self.source == "phone":
                    try:
                        self.cap.release()
                        time.sleep(0.1)  # Reduced sleep time
                        self.cap = cv2.VideoCapture(urljoin(self.base_url, "/video"))
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = self.cap.read()
                    except Exception as e:
                        print(f"Error reconnecting to camera: {e}")
                        return None
                        
            if ret and frame is not None:
                try:
                    # Basic image preprocessing (optimized)
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                    
                    # Ensure consistent BGR format for OpenCV operations
                    if len(frame.shape) == 2:  # If grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # If RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        
                    # Skip contrast enhancement for better performance
                    if self._needs_enhancement(frame):
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        l = clahe.apply(l)
                        lab = cv2.merge((l,a,b))
                        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        
                    return frame
                except Exception as e:
                    print(f"Error preprocessing frame: {e}")
                    return None
            return None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
            try:
                # Basic image preprocessing (optimized)
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Convert to RGB for consistency
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:  # Ensure BGR
                    frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
                    
                # Ensure uint8 data type
                frame = frame.astype(np.uint8)
                
                # Verify frame format
                if not (frame.shape[2] == 3 and frame.dtype == np.uint8):
                    print(f"Warning: Invalid frame format after preprocessing: shape={frame.shape}, dtype={frame.dtype}")
                    return None
                    
            except Exception as e:
                print(f"Error preprocessing frame: {e}")
                return None
            
            # Skip contrast enhancement for better performance
            # Only apply CLAHE if needed for low-light conditions
            if self._needs_enhancement(frame):
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        return frame if ret else None
        
    def _needs_enhancement(self, frame: np.ndarray) -> bool:
        """Check if frame needs contrast enhancement."""
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        return mean_brightness < 100  # Only enhance dark frames
        
    def release(self):
        """Release camera resources."""
        try:
            if self.cap is not None:
                # Release the capture object directly without grabbing frames
                self.cap.release()
                self.cap = None
        except Exception as e:
            # Ignore errors during cleanup
            pass
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass