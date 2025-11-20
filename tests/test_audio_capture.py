import time
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from NativaGPT.lib.speech_to_text.audio_capture import AudioCapture
from NativaGPT.lib.config_manager import ConfigManager
import logging
import wave
import io

def save_audio_to_wav_in_memory(audio_data: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Save audio data to an in-memory WAV file."""
    audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM format
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes (16-bit)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    wav_buffer.seek(0)  # Rewind the buffer to the beginning
    return wav_buffer

def main():
    """Main function to test audio capture and playback."""
    config_manager = ConfigManager("/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json")
    config = config_manager.get()
    audio_capture = AudioCapture(config)
    audio_capture.start_recording()

    captured_audio = []  # List to store captured audio chunks

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
            # Collect audio data from the queue
            while not audio_capture.audio_queue.empty():
                audio_chunk = audio_capture.audio_queue.get()
                captured_audio.append(audio_chunk)
    except KeyboardInterrupt:
        audio_capture.stop_recording()
        logging.info("Exiting...")

        # Combine all audio chunks into a single numpy array
        if captured_audio:
            full_audio = np.concatenate(captured_audio)
            sample_rate = audio_capture.sample_rate

            # Save the captured audio to an in-memory WAV file
            wav_buffer = save_audio_to_wav_in_memory(full_audio, sample_rate)
            logging.info("Audio saved to memory.")

            # Play the captured audio using pydub
            logging.info("Playing captured audio...")
            wav_buffer.seek(0)  # Ensure the buffer is at the beginning
            audio = AudioSegment.from_wav(wav_buffer)
            play(audio)

if __name__ == "__main__":
    main()