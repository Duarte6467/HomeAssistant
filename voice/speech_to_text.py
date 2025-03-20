import wave
from time import time
import pyaudio
import numpy as np
from io import BytesIO
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
# --- Configuration ---
# Moved configuration variables here for clarity.
# Ensure these are set in your environment or directly in the script (less secure for API keys).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable not set. Please set it.")
    exit()


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Standard sample rate for speech recognition
CHUNK = 1024
SILENCE_THRESHOLD = 300  # Adjust as needed for your environment
SILENCE_DURATION = 1.0  # Seconds of silence to consider end of speech
PRE_SPEECH_BUFFER_DURATION = 0.5  # Seconds of pre-speech to include

class Voices: # Keeping this, even if not used, as it was in original code for potential future use
    ADAM = "adam" # Example, could be removed if Voices enum is not used


class VoiceAssistant:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.g_client = Groq(api_key=GROQ_API_KEY) # Initialize Groq client - CORRECTED: g_client instead of oai_client

    @staticmethod
    def is_silence(data):
        """Detect if the provided audio data is silence using RMS."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < SILENCE_THRESHOLD

    def listen_for_speech(self):
        """Continuously detect silence and start recording when speech is detected."""
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Listening for speech...")
        pre_speech_buffer = []
        pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

        while True:
            data = stream.read(CHUNK)
            pre_speech_buffer.append(data)
            if len(pre_speech_buffer) > pre_speech_chunks:
                pre_speech_buffer.pop(0)

            if not self.is_silence(data):
                print("Speech detected, start recording...")
                stream.stop_stream()
                stream.close()
                return self.record_audio(pre_speech_buffer)

    def record_audio(self, pre_speech_buffer):
        """Record audio until silence is detected."""
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = pre_speech_buffer.copy()

        silent_chunks = 0
        print("Recording...") # Indicate recording has started
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if self.is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                print("Silence detected, recording stopped.") # Indicate recording stopped by silence
                break

        stream.stop_stream()
        stream.close()

        audio_bytes = BytesIO()
        with wave.open(audio_bytes, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        audio_bytes.seek(0)
        return audio_bytes

    def speech_to_text_g(self, audio_bytes):
        """
        Transcribe speech to text using Groq's Whisper API.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text, or None if transcription fails.
        """
        start_time = time()  # More descriptive variable name
        audio_bytes.seek(0)
        try:
            print("Transcribing audio with Groq...") # Indicate transcription started
            transcription = self.g_client.audio.transcriptions.create(
                file=("recording.wav", audio_bytes.read()), # More descriptive filename
                model="whisper-large-v3-turbo",
            )
            end_time = time() # More descriptive variable name
            print(f"Groq Transcription object: {transcription}") # For detailed debugging
            print(f"Transcription completed in: {end_time - start_time:.2f} seconds")
            return transcription.text
        except Exception as e:
            print(f"Error during Groq transcription: {e}")
            return None

def main():
    """Main function to demonstrate the voice assistant."""
    assistant = VoiceAssistant()

    while True:
        try:
            audio_bytes = assistant.listen_for_speech()
            if audio_bytes:
                text = assistant.speech_to_text_g(audio_bytes)
                if text:
                    print(f"Transcribed Text: {text}")
                else:
                    print("Transcription failed.") # Indicate transcription failure clearly

        except KeyboardInterrupt:
            print("Exiting voice assistant...") # More user-friendly exit message
            break
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}") # More specific error message in main loop

if __name__ == "__main__":
    main()