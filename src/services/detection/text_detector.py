"""
Text detection service using EasyOCR
"""

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple

class TextDetector:
    def __init__(self, languages: List[str] = ['en'], confidence_threshold: float = 0.5):
        """
        Initialize the EasyOCR text detector.
        
        Args:
            languages: List of language codes for OCR
            confidence_threshold: Minimum confidence score for text detection
            gpu: Whether to use GPU acceleration
        """
        self.reader = easyocr.Reader(languages, gpu=True)  # Enable GPU
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect text in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
            - List of text detections (dict with text, confidence, and coordinates)
            - Annotated frame with text boxes
        """
        # Ensure frame is in RGB format for EasyOCR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print(f"Unexpected frame format: shape={frame.shape}")
            return [], frame
            
        try:
            results = self.reader.readtext(frame_rgb)
        except Exception as e:
            print(f"Error in text detection: {e}")
            return [], frame
            
        detected_text = []
        annotated_frame = frame.copy()
        
        for (bbox, text, prob) in results:
            if prob > self.confidence_threshold:
                detected_text.append({
                    'text': text,
                    'confidence': prob,
                    'box': bbox
                })
                
                # Draw text bounding box
                try:
                    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], True, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, text,
                              (int(bbox[0][0]), int(bbox[0][1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    continue
        
        return detected_text, frame