"""
Image caption generation using BLIP model
"""

import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from typing import Tuple

class CaptionGenerator:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the BLIP caption generator.
        
        Args:
            model_name: HuggingFace model name/path
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        if self.device == "cuda":
            # Enable CUDA optimizations
            self.model.half()  # Convert to FP16 for faster inference
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

    def generate(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Generate a caption for the image frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
            - Generated caption string
            - Confidence score
        """
        with torch.amp.autocast('cuda'):  # Enable automatic mixed precision
            # Convert numpy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                # Convert inputs to half precision
                inputs = {k: v.half() if torch.is_floating_point(v) else v 
                         for k, v in inputs.items()}
            
            # Generate caption with optimized settings
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=4,  # Beam search for better quality
                    min_length=10,
                    length_penalty=1.0,
                    do_sample=False  # Use beam search without temperature
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence based on output logits
            confidence = float(torch.sigmoid(outputs.scores[0].mean()).cpu()) if hasattr(outputs, 'scores') else 1.0
            
            return caption, confidence