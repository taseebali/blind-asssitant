"""
Object detection service using YOLOv8
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8m.pt', confidence_threshold: float = 0.30):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_name: Path to the YOLO model (using medium model for better accuracy)
            confidence_threshold: Minimum confidence score for detections (increased for higher quality detections)
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Custom class mappings to improve detection accuracy
        self.class_mappings = {
            'bottle': 'drink',
            'wine glass': 'drink',
            'cup': 'drink',
            'vase': 'drink',  # Often misclassifies drinks as vases
            'bowl': 'container'
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
        Detect objects in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
            - List of detections (dict with class, confidence, and coordinates)
            - Annotated frame with detection boxes
        """
        # Resize frame for faster processing while maintaining aspect ratio
        height, width = frame.shape[:2]
        target_width = 640
        target_height = int(target_width * height / width)
        
        # Process frame
        with torch.amp.autocast('cuda'):  # Use mixed precision
            results = self.model(frame, verbose=False)  # Disable verbose output
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls)
                class_name = result.names[class_id]
                
                # Apply custom confidence threshold if available
                class_threshold = self.class_conf_thresholds.get(class_name, self.confidence_threshold)
                if conf < class_threshold:
                    continue
                    
                # Apply class mapping if available
                display_name = self.class_mappings.get(class_name, class_name)
                
                # Calculate relative depth based on box size and position
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                relative_size = box_area / frame_area
                
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, color_intensity, 0), 1)  # Thin box
                
                # Format text with class and confidence
                text = f'{class_name} {conf:.2f}'
                
                # Calculate text size for background
                font_scale = 0.5
                font_thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Draw text background (small green bar)
                cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), (0, 255, 0), -1)
                
                # Draw white text
                cv2.putText(frame, text,
                          (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, (255, 255, 255), font_thickness)  # White text on green background
        
        return detections, frame