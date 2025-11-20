import threading
import time
import numpy as np
import sounddevice as sd
import io
from pydub import AudioSegment
from pydub.playback import play
from NativaGPT.lib.speech_to_text.audio_capture import AudioCapture
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.speech_to_text.stt_prompt_handler import STTPromptHandler

if __name__ == "__main__":
    # Initialize Nativa components
    config_manager = ConfigManager("/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json")
    config = config_manager.get()
    audio_capture = AudioCapture(config)
    speech_to_text = STTPromptHandler(config)

    print("Starting Nativa audio capture...")

    while True:
        try:
            # Ensure audio capture is started
            if not audio_capture.is_recording:
                audio_capture.start()
                print("Recording... Speak now!")

            # Record and process for 5 seconds
            recording_duration = 6
            end_time = time.time() + recording_duration

            while time.time() < end_time:
                # Process any pending audio
                audio_capture.pre_process_audio()
                time.sleep(0.01)  # Small sleep to prevent CPU overuse

            # Pause recording for processing
            audio_capture.start_processing()
            print("Processing audio...")

            # Get the processed audio from Nativa
            audio_queue = audio_capture.get_audio_queue()
            audio_chunks = []

            while not audio_queue.empty():
                chunk = audio_queue.get()
                audio_chunks.append(chunk)

            # If we got any audio data, process it
            if audio_chunks:
                # Combine all chunks into one array
                audio_data = np.concatenate(audio_chunks)

                # Create virtual WAV file in memory
                virtual_wav = io.BytesIO()

                # Write the audio data to the virtual file
                import wave
                with wave.open(virtual_wav, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes per sample
                    wav_file.setframerate(audio_capture.target_sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                # Reset the virtual file pointer to the beginning
                virtual_wav.seek(0)

                # Play the audio directly from memory
                print("Playing back the recording...")
                audio = AudioSegment.from_wav(virtual_wav)
                play(audio)
                print("Playback finished!")

                # Send it to the Speech-to-Text API
                speech_to_text.processing_lock = threading.Lock()
                response = speech_to_text.send_stt_prompt(audio_data)
                speech_to_text.response_queue.put(response)
                stt_response = speech_to_text.get_stt_response()

                if stt_response is not None and "transcription" in stt_response:
                    print("Transcription:", stt_response["transcription"])
            else:
                print("No audio was captured!")

            # Resume audio capture
            audio_capture.stop_processing()
            print("\nReady for next recording...")
            time.sleep(0.5)  # Small pause between recordings

        except KeyboardInterrupt:
            print("\nStopping the program...")
            audio_capture.stop_recording()
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            audio_capture.stop_recording()
            audio_capture.stop_processing()
            time.sleep(1)  # Wait before retrying
            continue
