import json
import requests
import queue
import numpy as np
import io
import soundfile as sf
import time
import threading
import sounddevice as sd
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play
from NativaGPT.lib.speech_to_text.audio_capture import AudioCapture
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger

class STTPromptHandler:
    def __init__(self, config):
        self.config = config
        self.audio_capture = AudioCapture(config)
        self.response_queue = queue.Queue()
        self.is_running = False

        # State management
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.last_api_call = 0
        self.min_api_interval = 1.0  # Minimum time between API calls in seconds

        # Utterance management
        self.current_audio_chunks = []
        self.processing_utterance = False
        self.last_utterance_time = 0
        self.min_utterance_interval = 0.5  # Minimum time between utterances
        self.current_utterance_processed = False
        self.utterance_complete = threading.Event()
        self.audio_process_thread = None
        self.utterance_lock = threading.Lock()

        # Initialize audio related locks
        self.audio_lock = threading.Lock()
        self.api_lock = threading.Lock()

        logger.info("STT Prompt Handler initialized")

    def convert_audio_to_wav(self, audio_input):
        audio_buffer = io.BytesIO()
        if isinstance(audio_input, np.ndarray):
            # Normalize audio to [-1, 1] range
            audio_input = audio_input / np.max(np.abs(audio_input))

            # Convert to int16
            audio_input = np.int16(audio_input * 32767)

            # Write with explicit parameters
            sf.write(audio_buffer,
                     audio_input,
                     int(self.audio_capture.sample_rate),
                     format='WAV',
                     subtype='PCM_16')

            audio_buffer.seek(0)
            print(f"Created WAV with size: {audio_buffer.getbuffer().nbytes} bytes")

        return audio_buffer

    def convert_stt_prompt(self, audio_data):
        """Prepare raw audio data for STT API request."""
        try:
            tmp_wav_file = self.convert_audio_to_wav(audio_data)

            # Debug the quality of temp_wav_file
            # data, samplerate = sf.read(tmp_wav_file)
            # sd.play(data, samplerate)
            # sd.wait()

            files = {
                "audio": ('audio.wav', tmp_wav_file, 'audio/wav')
            }
            return files
        except Exception as e:
            logger.error(f"Error converting prompt: {e}")
            raise

    def can_make_api_call(self):
        """Check if an API call can be made based on the minimum interval."""
        with self.api_lock:
            current_time = time.time()
            if current_time - self.last_api_call >= self.min_api_interval:
                self.last_api_call = current_time
                return True
            return False

    def playback_audio(self, audio_data):
        """Play back audio data using pydub."""
        try:
            with self.audio_lock:
                # Ensure audio data is a numpy array
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)

                # Normalize and convert to 16-bit PCM
                audio_data = np.int16(audio_data * 32767)

                # Create virtual WAV file
                virtual_wav = io.BytesIO()
                sf.write(virtual_wav,
                         audio_data,
                         samplerate=int(self.audio_capture.sample_rate),  # Ensure integer sample rate
                         format='WAV',
                         subtype='PCM_16')
                virtual_wav.seek(0)

                # Create AudioSegment and play
                try:
                    audio = AudioSegment.from_wav(virtual_wav)
                    print(f"Playing back recording... Duration: {len(audio)/1000:.2f} seconds")
                    print(f"Audio properties - Channels: {audio.channels}, Sample width: {audio.sample_width}, Frame rate: {audio.frame_rate}")
                    play(audio)
                except Exception as e:
                    print(f"Error in AudioSegment creation/playback: {e}")

        except Exception as e:
            logger.error(f"Error playing back audio: {e}")
            print(f"Playback error details - Data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            print(f"Sample rate: {self.audio_capture.sample_rate}")

    def send_stt_prompt(self, audio_data):
        """Send raw audio data to the STT API and return the response."""
        result = None  # Initialize return value
        try:
            # Check if we can process
            with self.processing_lock:
                if self.is_processing or not self.can_make_api_call():
                    logger.info("Skipping API call - already processing or too soon")
                    return None
                self.is_processing = True

            # Convert audio to prompt
            logger.info("Converting audio for STT prompt...")
            prompt = self.convert_stt_prompt(audio_data)
            if prompt is None:
                logger.error("No STT prompt found")
                return None

            # Send request to API
            logger.info("Sending STT prompt to API...")
            stt_response = requests.post(
                url=self.config["stt_config"]["api_url"],
                files=prompt,  # Send raw audio data
                #headers={"Content-Type": "application/octet-stream"},  # Set appropriate content type
                timeout=10
            )
            stt_response.raise_for_status()
            result = stt_response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            result = None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode API response: {e}")
            result = None
        except Exception as e:
            logger.error(f"Error sending STT prompt: {e}")
            result = None
        finally:
            with self.processing_lock:
                self.is_processing = False
            return result  # Always return the result in finally block

    def collect_audio_chunks(self):
        """Collect audio chunks from the audio queue."""
        chunks = []
        try:
            with self.audio_lock:
                while not self.audio_capture.audio_queue.empty():
                    chunks.append(self.audio_capture.audio_queue.get_nowait())
                return chunks
        except queue.Empty:
            return chunks

    def process_audio(self):
        """Process audio data and send it to the STT API."""
        try:
            if not self.audio_capture.is_recording:
                self.audio_capture.start_recording()
                print("Recording... Speak now!")
                return False  # Give time for recording to start

            current_time = time.time()

            # Handle minimum time between utterances
            with self.utterance_lock:
                if current_time - self.last_utterance_time < self.min_utterance_interval:
                    return False

            # Use the correct method name (without underscore)
            self.audio_capture._handle_potential_silence(np.array([]), time.time())

            # Wait longer for utterance completion
            if not self.audio_capture.utterance_complete.wait(timeout=0.5):  # Increased timeout
                return False

            if self.processing_utterance:
                return False

            with self.utterance_lock:
                self.processing_utterance = True
                print("Utterance detected, processing...")

                try:
                    chunks = self.collect_audio_chunks()
                    if not chunks:  # Add explicit check for empty chunks
                        print("No audio chunks collected")
                        return False

                    # Add debug print for audio length
                    full_audio = np.concatenate(chunks)
                    print(f"Processing audio chunk of length: {len(full_audio)}")

                    # Play back the audio before sending to API
                    # print("Playing back the recorded audio...")
                    # self.playback_audio(full_audio)

                    # Optional: Add minimum audio length check
                    if len(full_audio) < self.audio_capture.sample_rate * 0.5:  # Min 0.5 seconds
                        print("Audio too short, waiting for more...")
                        return False

                    stt_response = self.send_stt_prompt(full_audio)
                    if stt_response is not None:
                        self.response_queue.put(stt_response)
                        logger.info("Processed complete utterance")
                        print("\nReady for next recording...")
                        self.last_utterance_time = current_time
                        return True

                finally:
                    self.processing_utterance = False
                    self.current_utterance_processed = True
                    self.utterance_complete.clear()
                    self.audio_capture.current_utterance_processed = False
                    self.audio_capture.utterance_complete.clear()

            return False

        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            self.processing_utterance = False
            return False

    def start_stt_handler(self):
        """Start the STT handler and audio capture."""
        try:
            self.is_running = True
            self.audio_capture.start_recording()
            logger.info("STT handler started successfully")
        except Exception as e:
            logger.error(f"Failed to start STT handler: {e}")
            self.is_running = False
            raise

    def stop_stt_handler(self):
        """Stop the STT handler and audio capture."""
        try:
            self.is_running = False
            with self.processing_lock:
                self.is_processing = False
            self.clear_queues()
            self.audio_capture.stop_recording()
            logger.info("STT handler stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping STT handler: {e}")

    def clear_queues(self):
        """Clear the response queue."""
        try:
            with self.audio_lock:
                while not self.response_queue.empty():
                    self.response_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error clearing queues: {e}")

    def get_response(self):
        """Get the latest response from the response queue."""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # Setup logger
    logger.basicConfig(
        level=logger.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize components
    config_manager = ConfigManager(config_path="/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json")
    config = config_manager.get()

    # Print audio configuration
    print("Audio Configuration:")
    print(f"Volume Threshold: {config['stt_config']['audio_capture_config']['volume_threshold']}")
    print(f"Silence Timeout: {config['stt_config']['audio_capture_config']['silence_timeout']}")
    print(f"Sample Rate: {config['stt_config']['audio_capture_config']['target_sample_rate']}")

    module = STTPromptHandler(config=config)
    module.start_stt_handler()

    try:
        print("\nListening... (Press Ctrl+C to stop)")
        while module.is_running:
            module.process_audio()
            response = module.get_response()
            if response:
                print(response)
            time.sleep(0.01)  # Smaller sleep for more responsive capture
    except KeyboardInterrupt:
        print("\nStopping the program...")
        module.stop_stt_handler()