# """
# Speech recognition service using Whisper
# """

import whisper
import numpy as np
import torch
from typing import Optional

class SpeechRecognizer:
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper speech recognition.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)

    def listen(self) -> Optional[str]:
        """
        Listen for and transcribe speech command.
        
        Returns:
            Transcribed text if speech detected, None otherwise
        """
        # TODO: Implement real-time audio capture
        # For now, this is a placeholder that would be replaced with actual
        # microphone input processing
        return None

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: numpy array of audio samples
            
        Returns:
            Transcribed text
        """
        result = self.model.transcribe(audio_data)
        return result["text"]