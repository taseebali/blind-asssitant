"""
Object detection service using YOLOv8
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8s.pt', confidence_threshold: float = 0.45):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_name: Path to the YOLO model (using small model for better speed)
            confidence_threshold: Minimum confidence score for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Configure YOLO parameters
        self.model_params = {
            'conf': confidence_threshold,  # Confidence threshold
            'iou': 0.45,                  # NMS IoU threshold
            'device': 0 if torch.cuda.is_available() else 'cpu',  # Use GPU device 0 if available
            'max_det': 10,                # Maximum detections per image
            'agnostic_nms': True,         # Class-agnostic NMS
            'half': True                  # Use FP16 half-precision inference
        }
        
        # Set model device and optimize
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        if torch.cuda.is_available():
            self.model.fuse()
        
        # Track previous detections for stability
        self.previous_detections = []
        self.detection_history = []
        self.history_length = 5  # Number of frames to track
        
        # Class mappings for office environment
        self.class_mappings = {
            'dining table': 'desk',  # Office context
            'laptop': 'computer',
            'wine glass': 'container',
            'cup': 'container',
            'bottle': 'container',
            'vase': 'container',
            'light': 'lamp',
            'chair': 'chair',
            'tv': 'monitor',  # Fix TV misclassification
            'tvmonitor': 'monitor',
            'display': 'monitor'
        }
        
        # Class-specific confidence thresholds
        self.class_conf_thresholds = {
            'dining table': 0.60,  # Higher threshold for large objects
            'desk': 0.60,
            'chair': 0.55,
            'laptop': 0.50,
            'light': 0.45,  # Lower for small objects
            'lamp': 0.45,
            'mouse': 0.55,  # Higher to avoid false positives
            'cell phone': 0.55,
            'bottle': 0.55,
            'cup': 0.55,
            'default': 0.45
        }
        
        # Context-specific confidence thresholds
        self.class_conf_thresholds = {
            'dining table': 0.45,  # Higher threshold to avoid false positives
            'desk': 0.35,
            'laptop': 0.35,
            'light': 0.25,  # Lower threshold for small objects like lamps
            'lamp': 0.25,
            'chair': 0.40,
            'monitor': 0.35,
            'keyboard': 0.35,
            'mouse': 0.35
        }
        
        # Custom size thresholds for different object types
        self.size_thresholds = {
            'lamp': 0.01,  # Small objects need lower size threshold
            'mouse': 0.01,
            'keyboard': 0.02,
            'default': 0.03
        }
        
        # Class-specific confidence thresholds
        self.class_conf_thresholds = {
            'bottle': 0.25,
            'wine glass': 0.25,
            'cup': 0.25,
            'vase': 0.25,
            'bowl': 0.25
        }
        
        # Configure model for GPU and performance
        self.model.to('cuda')  # Move model to GPU
        self.model.fuse()  # Fuse layers for better performance
        
        # Set model parameters for better performance
        self.model.overrides['conf'] = confidence_threshold  # Detection confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 50  # Maximum number of detections per image

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in the frame with improved spatial awareness and confidence handling.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
            - List of detections (dict with class, confidence, and coordinates)
            - Annotated frame with detection boxes
        """
        def debug_params(params: dict) -> None:
            """Debug helper to print current parameters"""
            print("\nYOLO Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")

        if frame is None or frame.size == 0:
            return [], frame

        # Calculate frame dimensions and area
        height, width = frame.shape[:2]
        frame_area = height * width
        
        try:
            # Debug parameters on first run
            if not hasattr(self, '_params_printed'):
                debug_params(self.model_params)
                self._params_printed = True
                
            # Process frame with optimized settings
            results = self.model(
                source=frame,
                verbose=False,
                show=False,
                **self.model_params
            )
        
            detections = []
            annotated_frame = frame.copy()

            # Process the first (and only) result
            if len(results) > 0:
                result = results[0]  # YOLOv8 returns a list of Results objects
                
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        try:
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Get confidence and class
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            
                            # Apply custom confidence threshold if available
                            class_threshold = self.class_conf_thresholds.get(class_name, self.confidence_threshold)
                            if conf < class_threshold:
                                continue
                                
                            # Get class-specific confidence threshold
                            conf_threshold = self.class_conf_thresholds.get(class_name, self.confidence_threshold)
                            if conf < conf_threshold:
                                continue
                            
                            # Calculate object size and check against threshold
                            box_area = (x2 - x1) * (y2 - y1)
                            relative_size = box_area / frame_area
                            size_threshold = self.size_thresholds.get(class_name, self.size_thresholds['default'])
                            
                            # Don't filter small objects that are clearly detected
                            if relative_size < size_threshold and conf < conf_threshold + 0.1:
                                continue
                            
                            # Apply class mapping if available
                            display_name = self.class_mappings.get(class_name, class_name)
                            
                            # Calculate box center
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            # Calculate relative position (0-1 scale)
                            rel_x = center_x / frame.shape[1]
                            rel_y = center_y / frame.shape[0]
                            
                            detections.append({
                                'class': display_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'depth_score': relative_size,  # Larger objects are typically closer
                                'position': {
                                    'x': rel_x,  # 0 = left, 1 = right
                                    'y': rel_y,  # 0 = top, 1 = bottom
                                    'center': (center_x, center_y)
                                }
                            })
                            
                            # Draw thin bounding box with color based on depth (closer = brighter green)
                            color_intensity = int(min(255, 100 + (relative_size * 1000)))
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, color_intensity, 0), 1)
                            
                            # Format text with class and confidence
                            text = f'{display_name} {conf:.2f}'
                            
                            # Calculate text size for background
                            font_scale = 0.5
                            font_thickness = 1
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                font_scale, font_thickness)
                            
                            # Draw text background (small green bar)
                            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 4), 
                                        (x1 + text_w + 4, y1), (0, 255, 0), -1)
                            
                            # Draw white text
                            cv2.putText(annotated_frame, text, (x1 + 2, y1 - 4), 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                      (255, 255, 255), font_thickness)
                            
                        except Exception as box_error:
                            print(f"Error processing box: {box_error}")
                            continue
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return [], frame