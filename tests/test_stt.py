import time
import numpy as np
import sounddevice as sd
import io
import threading
import logging
import queue
from pydub import AudioSegment
from pydub.playback import play
from NativaGPT.lib.speech_to_text.audio_capture import AudioCapture
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.speech_to_text.stt_prompt_handler import STTPromptHandler

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_and_playback_audio(audio_data, sample_rate):
    virtual_wav = io.BytesIO()
    import wave
    with wave.open(virtual_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    virtual_wav.seek(0)
    audio = AudioSegment.from_wav(virtual_wav)
    play(audio)

def collect_audio_chunks(audio_queue):
    chunks = []
    while True:
        try:
            chunk = audio_queue.get_nowait()
            chunks.append(chunk)
        except queue.Empty:
            break
    return chunks

if __name__ == "__main__":
    config_manager = ConfigManager("/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json")
    config = config_manager.get()

    print("Audio Configuration:")
    print(f"Volume Threshold: {config['stt_config']['audio_capture_config']['volume_threshold']}")
    print(f"Silence Timeout: {config['stt_config']['audio_capture_config']['silence_timeout']}")
    print(f"Sample Rate: {config['stt_config']['audio_capture_config']['target_sample_rate']}")

    # Create audio capture instance directly
    audio_capture = AudioCapture(config)
    stt_handler = STTPromptHandler(config)

    print("\nStarting audio capture system...")
    audio_capture.start()

    try:
        print("Listening... (Press Ctrl+C to stop)")
        last_utterance_time = 0

        while True:
            # Process any pending audio
            try:
                audio_capture.pre_process_audio()
            except Exception as e:
                print(f"Error processing audio: {e}")

            current_time = time.time()

            # Wait for utterance with a short timeout
            if audio_capture.wait_for_utterance_complete(timeout=0.1):
                # Ensure we don't process the same utterance multiple times
                if current_time - last_utterance_time > 1.0:  # Minimum 1 second between utterances
                    print("Utterance detected, processing...")

                    # Collect audio chunks
                    chunks = collect_audio_chunks(audio_capture.get_audio_queue())

                    if chunks:
                        # Combine all chunks
                        full_audio = np.concatenate(chunks)

                        # Play back the audio
                        print("Playing back recording...")
                        #process_and_playback_audio(full_audio, audio_capture.target_sample_rate)

                        # Send to STT
                        response = stt_handler.send_stt_prompt(full_audio)
                        if response and "transcription" in response:
                            print("Transcription:", response["transcription"])

                        # Clear utterance complete flag
                        audio_capture.utterance_complete.clear()
                        audio_capture.clear_queue(audio_capture.audio_queue)
                        audio_capture.clear_queue(audio_capture.processed_audio_queue)
                        audio_capture.current_utterance_processed = False

                        print("\nReady for next recording...")
                        last_utterance_time = current_time
                    else:
                        print("No audio chunks available, continuing to listen...")
                        audio_capture.utterance_complete.clear()
                        time.sleep(0.1)

            # Print status every 5 seconds
            if current_time % 5 < 0.1:
                print("System is listening...")

            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        audio_capture.stop_recording()
        print("Audio capture system stopped.")