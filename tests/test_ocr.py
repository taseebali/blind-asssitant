"""
Test OCR functionality
"""

import cv2
import numpy as np
import pytest
from src.services.detection.text_detector import TextDetector

@pytest.fixture
def sample_image():
    """Create a test image with text."""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, "Test OCR 123", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img

def test_text_detector_initialization():
    """Test TextDetector initialization."""
    detector = TextDetector()
    assert detector is not None
    assert detector.confidence_threshold == 0.4

def test_text_detection(sample_image):
    """Test text detection on a simple image."""
    detector = TextDetector()
    detections, annotated = detector.detect(sample_image)
    
    # Save test output
    cv2.imwrite("tests/test_output/ocr_test_output.jpg", annotated)
    
    # Basic assertions
    assert len(detections) > 0, "No text detected"
    assert isinstance(detections, list), "Detections should be a list"
    assert isinstance(annotated, np.ndarray), "Annotated frame should be an image"
    
    # Print detections for debugging
    print("\nDetected text:")
    for d in detections:
        print(f"Text: {d['text']}, Confidence: {d['confidence']:.2f}")

def test_text_detection_with_real_image():
    """Test text detection on a real image."""
    # Load test image
    img_path = "tests/test_data/test_image.jpg"
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load test image from {img_path}"
    
    detector = TextDetector()
    detections, annotated = detector.detect(img)
    
    # Save annotated output
    cv2.imwrite("tests/test_output/ocr_real_test_output.jpg", annotated)
    
    # Print results
    print("\nDetected text in real image:")
    for d in detections:
        print(f"Text: {d['text']}, Confidence: {d['confidence']:.2f}")
    
    assert isinstance(detections, list), "Detections should be a list"
    assert isinstance(annotated, np.ndarray), "Annotated frame should be an image"

def test_text_detection_empty_image():
    """Test text detection with empty/invalid input."""
    detector = TextDetector()
    
    # Test with None
    detections, annotated = detector.detect(None)
    assert len(detections) == 0, "Should handle None input"
    
    # Test with empty image
    empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections, annotated = detector.detect(empty_img)
    assert len(detections) == 0, "Should handle empty image"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])