# /app/ml_models/audio_analysis.py

import pyaudio
import numpy as np
import threading
import time
from collections import deque

class AudioAnalyzer:
    """
    A class to handle real-time audio input and analyze its volume
    to detect potential conversations or whispers.
    """
    # --- Constants for audio stream ---
    FORMAT = pyaudio.paInt16  # Data format for the audio stream (16-bit integers)
    CHANNELS = 1              # Mono audio
    RATE = 44100              # Standard sample rate in Hz
    CHUNK = 1024              # Number of audio frames per buffer

    def __init__(self, volume_threshold=500, silence_duration=2):
        """
        Initializes the AudioAnalyzer.

        Args:
            volume_threshold (int): The RMS amplitude threshold to consider as "sound".
                                    This value is empirical and may need tuning for your microphone.
            silence_duration (int): The number of seconds of continuous sound to trigger an alert.
        """
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.volume_threshold = volume_threshold
        self.silence_duration = silence_duration
        
        # A deque is a special list that has a maximum size. Old items are automatically
        # discarded when new items are added, making it perfect for tracking recent audio levels.
        # We'll store roughly `silence_duration` seconds of audio level history.
        history_size = int(self.RATE / self.CHUNK * self.silence_duration)
        self.volume_history = deque(maxlen=history_size)
        
        self.sound_detected = False
        print("[INFO] AudioAnalyzer initialized.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """This is the function that gets called every time a new chunk of audio is ready."""
        # Convert the raw byte data into an array of numbers
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Calculate the Root Mean Square (RMS) - a good measure of volume
        rms = np.sqrt(np.mean(np.square(audio_data, dtype=np.float64)))
        
        # Add the current volume to our history
        self.volume_history.append(rms)
        
        # Check if the recent average volume is above the threshold
        if len(self.volume_history) == self.volume_history.maxlen:
            # Check if all recent readings are above the threshold
            if all(v > self.volume_threshold for v in self.volume_history):
                self.sound_detected = True
            else:
                self.sound_detected = False
        else:
            self.sound_detected = False

        return (in_data, pyaudio.paContinue)

    def start(self):
        """Starts the audio monitoring in a separate thread."""
        if self.is_running:
            print("[WARNING] AudioAnalyzer is already running.")
            return
            
        print("[INFO] Starting audio stream...")
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.is_running = True
            print("[INFO] Audio stream started successfully.")
        except Exception as e:
            print(f"[ERROR] Could not open audio stream: {e}")
            print("[INFO] Audio analysis will be disabled.")
            self.is_running = False

    def stop(self):
        """Stops the audio monitoring thread."""
        if not self.is_running:
            return
            
        print("[INFO] Stopping audio stream...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.is_running = False
        print("[INFO] Audio stream stopped.")

    def is_sound_detected(self):
        """Public method to check if a continuous sound was detected."""
        return self.sound_detected

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    # You will likely need to adjust this threshold for your specific microphone and environment.
    # Start with a low value (like 100) and increase it until background noise is ignored.
    analyzer = AudioAnalyzer(volume_threshold=300)
    analyzer.start()
    
    print("\n[INFO] Monitoring audio. Speak continuously for 2 seconds to trigger an alert.")
    print("Press Ctrl+C to quit.")
    
    try:
        while True:
            if analyzer.is_sound_detected():
                print("ALERT: Continuous sound detected! (Potential Conversation)")
                # In a real app, we'd want a cooldown period after an alert.
                time.sleep(2) # Simple cooldown
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted. Shutting down.")
    finally:
        analyzer.stop()
