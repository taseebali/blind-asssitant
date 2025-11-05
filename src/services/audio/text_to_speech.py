# """
# Text-to-speech service using gTTS
# """

import os
import pygame
from gtts import gTTS
from queue import Queue
from threading import Thread
import time
from typing import Optional

class TextToSpeech:
    def __init__(self):
        """Initialize the TTS service with audio playback queue."""
        pygame.mixer.init()
        self.audio_queue = Queue()
        self.running = True
        self.audio_thread = Thread(target=self._process_audio_queue, daemon=True)
        self.audio_thread.start()

    def speak(self, text: str):
        """
        Add text to the speech queue.
        
        Args:
            text: Text to be converted to speech
        """
        if text:
            # Clear old pending audio and add new one for immediate response
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
            self.audio_queue.put(text)

    def _process_audio_queue(self):
        """Process and play text from the audio queue."""
        while self.running:
            if not self.audio_queue.empty():
                text = self.audio_queue.get()
                self._play_audio(text)
            time.sleep(0.1)

    def _play_audio(self, text: str):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to be converted and played
        """
        try:
            # Create unique temp file for each audio
            temp_file = f"temp_audio_{int(time.time()*1000)}.mp3"
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Make sure mixer is done before removing
            pygame.mixer.music.unload()
            try:
                os.remove(temp_file)
            except:
                pass  # Ignore deletion errors
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            # Cleanup any remaining temp files
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()
        pygame.mixer.quit()