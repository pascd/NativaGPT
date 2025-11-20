import os
import time
import sounddevice as sd
import numpy as np
import queue
import threading

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path
from scipy.signal import resample, butter, filtfilt
from contextlib import contextmanager

from NativaGPT.lib.coloring_logger import logger

@dataclass
class NoiseReductionConfig:
    noise_threshold: float
    noise_reduce_strength: float
    smoothing_factor: float
    lowpass_cutoff: float
    highpass_cutoff: float

class AudioCapture:
    def __init__(self, config: Dict[str, Any]):
        self.config = config  # Store the config for later use
        self._init_config(config)
        self._init_audio_state()
        self._init_threading()
        self._init_noise_reduction()
        logger.info("AudioCapture initialized")

    def _init_config(self, config: Dict[str, Any]) -> None:
        """Initialize configuration parameters."""
        audio_config = config["stt_config"]["audio_capture_config"]
        self.target_sample_rate = audio_config["target_sample_rate"]
        self.volume_threshold = audio_config["volume_threshold"]
        self.chunk_duration = audio_config["chunk_duration"]
        self.silence_timeout = audio_config["silence_timeout"]
        self.pre_padding = audio_config["pre_padding"]
        self.sample_rate = self.get_default_sample_rate()
        self.buffer_size = int(self.pre_padding * self.sample_rate)
        self.is_processing = False

    def _init_audio_state(self) -> None:
        """Initialize audio state variables."""
        self.buffer: List[np.ndarray] = []
        self.audio_queue: queue.Queue = queue.Queue()
        self.processed_audio_queue: queue.Queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        self.stream: Optional[sd.InputStream] = None
        self.last_sound_timestamp: Optional[float] = None  # Properly initialized
        self.current_utterance_processed = False

    def _init_threading(self) -> None:
        """Initialize threading-related variables."""
        self.processing_lock = threading.Lock()
        self.utterance_complete = threading.Event()

    def _init_noise_reduction(self) -> None:
        """Initialize noise reduction configuration."""
        audio_config = self.config["stt_config"]["audio_capture_config"]
        self.noise_reduction_config = NoiseReductionConfig(
            noise_threshold=audio_config.get("noise_threshold", 0.005),
            noise_reduce_strength=audio_config.get("noise_reduce_strength", 2.0),
            smoothing_factor=audio_config.get("smoothing_factor", 0.95),
            lowpass_cutoff=audio_config.get("lowpass_cutoff", 3000),
            highpass_cutoff=audio_config.get("highpass_cutoff", 100)
        )
        self.noise_profile = None
        self.noise_profile_samples: List[np.ndarray] = []
        self.calibration_duration = 1.0
        self.is_calibrating = True

    @contextmanager
    def processing_context(self) -> Generator[None, None, None]:
        """Context manager for processing state."""
        try:
            with self.processing_lock:
                self.is_processing = True
            yield
        finally:
            with self.processing_lock:
                self.is_processing = False

    def apply_bandpass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter with improved error handling and performance."""
        min_chunk_size = 256
        pad_size = None
        try:
            if len(audio_data) < min_chunk_size:
                pad_size = min_chunk_size - len(audio_data)
                padded_data = np.pad(audio_data, (pad_size // 2, pad_size - pad_size // 2), 'edge')
            else:
                padded_data = audio_data

            nyquist = self.sample_rate / 2
            low = self.noise_reduction_config.highpass_cutoff / nyquist
            high = self.noise_reduction_config.lowpass_cutoff / nyquist

            b, a = butter(1, [low, high], btype='band')
            filtered_data = filtfilt(b, a, padded_data)

            return (filtered_data[pad_size // 2:pad_size // 2 + len(audio_data)]
                    if len(audio_data) < min_chunk_size else filtered_data)

        except Exception as e:
            logger.warning(f"Filtering failed: {e}, returning original audio")
            return audio_data

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict, status: Any) -> None:
        """Improved audio callback with better error handling and state management."""
        if status:
            logger.warning(f"Audio status: {status}")

        try:
            with self.processing_lock:
                if self.is_processing:
                    return

            audio_data = self._preprocess_audio_data(indata)
            volume_norm = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
            current_time = time.time()

            self._handle_audio_event(audio_data, volume_norm, current_time)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _preprocess_audio_data(self, indata: np.ndarray) -> np.ndarray:
        """Preprocess audio data with noise reduction."""
        audio_data = indata.copy()
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return self.reduce_noise(audio_data)

    def _handle_audio_event(self, audio_data: np.ndarray, volume_norm: float, current_time: float) -> None:
        """Handle audio events based on volume threshold and timing."""
        if volume_norm > self.volume_threshold:
            self._handle_sound_detected(audio_data, current_time)
        elif self.last_sound_timestamp is not None:
            self._handle_potential_silence(audio_data, current_time)

    def _handle_potential_silence(self, audio_data: np.ndarray, current_time: float) -> None:
        """Handle potential silence after sound detection."""
        if self.last_sound_timestamp is not None:
            silence_duration = current_time - self.last_sound_timestamp
            if silence_duration >= self.silence_timeout:
                if not self.current_utterance_processed:
                    logger.info(f"Silence detected after {silence_duration:.2f} seconds")
                    self.current_utterance_processed = True
                    self.utterance_complete.set()
                    self.last_sound_timestamp = None
            elif self.is_recording:
                self.audio_queue.put(audio_data)

    def _handle_sound_detected(self, audio_data: np.ndarray, current_time: float) -> None:
        """Handle detected sound above threshold."""
        if self.last_sound_timestamp is None:
            logger.info("Sound detected! Starting new utterance...")
            self.current_utterance_processed = False
            self.utterance_complete.clear()

        self.last_sound_timestamp = current_time
        if self.is_recording:
            self.audio_queue.put(audio_data)

    def get_default_sample_rate(self) -> int:
        """Get the default sample rate from the sound device."""
        return sd.query_devices(kind='input')['default_samplerate']

    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce noise from the audio data."""
        # Placeholder for noise reduction logic
        return audio_data

    def start_recording(self) -> None:
        """Start recording audio."""
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * self.chunk_duration)
        )
        self.stream.start()
        logger.info("Recording started...")

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            logger.info("Recording stopped.")