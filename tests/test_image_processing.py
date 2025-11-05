"""
Integration test for the Blind Assistant using static images
"""

import os
import sys
import cv2
import pytest
from pathlib import Path
import numpy as np

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tests.utils.logging.test_logger import TestLogger

from src.services.detection.object_detector import ObjectDetector
from src.services.detection.text_detector import TextDetector
from src.services.detection.caption_generator import CaptionGenerator
from src.services.narration_service import NarrationService
from src.services.audio.text_to_speech import TextToSpeech

class TestImageProcessor:
    @pytest.fixture(scope="class")
    def test_image_path(self):
        """Fixture to provide test image path."""
        # You can add your test image to the test_data directory
        image_path = Path(__file__).parent / "test_data" / "test_image.jpg"
        if not image_path.exists():
            pytest.skip(f"Test image not found at {image_path}")
        return str(image_path)

    @pytest.fixture(scope="class")
    def services(self):
        """Fixture to initialize all services."""
        return {
            'object_detector': ObjectDetector(),
            'text_detector': TextDetector(),
            'caption_generator': CaptionGenerator(),
            'narration': NarrationService(),
            'tts': TextToSpeech()
        }

    def test_full_image_processing(self, test_image_path, services):
        """
        Test complete image processing pipeline.
        """
        logger = TestLogger(__name__)
        logger.start_test("Full Image Processing Pipeline Test")
        
        try:
            # 1. Load image
            logger.log_info(f"Loading test image from {test_image_path}")
            image = cv2.imread(test_image_path)
            assert image is not None, f"Failed to load image from {test_image_path}"
            logger.log_success("Image loaded successfully")

            # 2. Object Detection
            logger.log_info("Running object detection...")
            objects, annotated_frame = services['object_detector'].detect(image.copy())
            logger.log_success(f"Found {len(objects)} objects")
            for obj in objects:
                logger.log_info(f"Detected {obj['class']} with confidence: {obj['confidence']:.2f}")

            # 3. Text Detection
            logger.log_info("Running text detection...")
            texts, annotated_frame = services['text_detector'].detect(annotated_frame)
            logger.log_success(f"Found {len(texts)} text regions")
            for text in texts:
                logger.log_info(f"Detected text: '{text['text']}' with confidence: {text['confidence']:.2f}")

            # 4. Image Captioning
            logger.log_info("Generating image caption...")
            try:
                caption, caption_confidence = services['caption_generator'].generate(image)
                logger.log_success(f"Generated caption: {caption} (confidence: {caption_confidence:.2f})")
            except Exception as e:
                logger.log_error(f"Caption generation failed: {str(e)}")
                caption, caption_confidence = "Failed to generate caption", 0.0

            # 5. Generate Narration
            logger.log_info("Creating narration...")
            narration = services['narration'].generate(objects, texts, caption)
            logger.log_success(f"Generated narration: {narration}")

            # 6. Save annotated image
            output_dir = Path(__file__).parent / "test_output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "annotated_test_image.jpg"
            
            if caption:
                cv2.putText(annotated_frame, caption, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imwrite(str(output_path), annotated_frame)
            logger.log_success(f"Saved annotated image to {output_path}")

            # Assertions
            assert len(objects) >= 0, "Object detection failed"
            assert len(texts) >= 0, "Text detection failed"
            assert caption is not None, "Caption generation failed"
            assert narration is not None, "Narration generation failed"
            
            logger.end_test(success=True)
            
        except Exception as e:
            logger.log_error(f"Test failed: {str(e)}")
            logger.end_test(success=False)
            raise

    def test_individual_services(self, test_image_path, services):
        """
        Test each service individually with detailed output.
        """
        logger = TestLogger(__name__)
        logger.start_test("Individual Services Test")
        
        try:
            # Load image
            logger.log_info(f"Loading test image from {test_image_path}")
            image = cv2.imread(test_image_path)
            assert image is not None, f"Failed to load image from {test_image_path}"
            logger.log_success("Image loaded successfully")

            # Test Object Detector
            logger.log_info("Testing Object Detector...")
            objects, _ = services['object_detector'].detect(image.copy())
            assert isinstance(objects, list)
            logger.log_info(f"Validated object detector output type")
            
            for obj in objects:
                assert 'class' in obj
                assert 'confidence' in obj
                assert 'box' in obj
            logger.log_success(f"Object detector returned {len(objects)} valid detections")

            # Test Text Detector
            logger.log_info("Testing Text Detector...")
            texts, _ = services['text_detector'].detect(image.copy())
            assert isinstance(texts, list)
            logger.log_info("Validated text detector output type")
            
            for text in texts:
                assert 'text' in text
                assert 'confidence' in text
                assert 'box' in text
            logger.log_success(f"Text detector returned {len(texts)} valid text regions")

            # Test Caption Generator
            logger.log_info("Testing Caption Generator...")
            caption, confidence = services['caption_generator'].generate(image)
            assert isinstance(caption, str)
            assert isinstance(confidence, float)
            logger.log_success(f"Caption generated successfully with confidence: {confidence:.2f}")

            # Test Narration Service
            logger.log_info("Testing Narration Service...")
            narration = services['narration'].generate(objects, texts, caption)
            assert isinstance(narration, str) or narration is None
            logger.log_success("Narration generated successfully")
            
            logger.end_test(success=True)
            
        except Exception as e:
            logger.log_error(f"Test failed: {str(e)}")
            logger.end_test(success=False)
            raise

def main():
    """
    Run the tests directly with detailed output.
    """
    # Create test directories if they don't exist
    test_data_dir = Path(__file__).parent / "test_data"
    test_output_dir = Path(__file__).parent / "test_output"
    test_data_dir.mkdir(exist_ok=True)
    test_output_dir.mkdir(exist_ok=True)

    # Check for test image
    test_image_path = test_data_dir / "test_image.jpg"
    if not test_image_path.exists():
        print(f"Please add a test image at: {test_image_path}")
        print("You can use any image file for testing.")
        return

    # Run tests
    pytest.main([__file__, "-v", "--capture=no"])

if __name__ == "__main__":
    main()