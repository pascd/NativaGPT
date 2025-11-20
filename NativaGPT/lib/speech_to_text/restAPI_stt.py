import torch
import numpy as np
import base64
import io
import soundfile as sf
import os

from scipy.signal import resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger

# App definitions
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

# Get the project root directory (assuming your config is relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Create the config path relative to project root
config_path = os.path.join(PROJECT_ROOT, "config", "config_default.json")

# Use the resolved path
config_manager = ConfigManager(config_path)
config = config_manager.get()

# Handle config if it's a list (take first item)
if isinstance(config, list):
    config = config[0]

MODEL_ID = config["stt_config"]["rest_api"]["model_id"]
TARGET_SAMPLE_RATE = config["stt_config"]["rest_api"]["target_sample_rate"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Global variable for the model pipeline
pipe = None

def load_model():
    """Loads the ASR model with optimized settings."""
    global pipe
    if pipe is None:
        logger.info(f"Loading model: {MODEL_ID} on {DEVICE}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        logger.info("Model loaded successfully.")
    return pipe

class Test(Resource):
    def get(self):
        return {"message": "API working."}, 200

    def post(self):
        try:
            value = request.get_json()
            if value.get("text"):
                return {'Post Values': value}, 201
            return {"error": "Invalid Request"}, 400
        except Exception as error:
            return {"error": str(error)}, 400

class STTService(Resource):
    def get(self):
        return {"status": "Speech-to-Text service is running"}, 200

    def post(self):
        try:
            # Check if file is in request
            if 'audio' not in request.files:
                return {"error": "No audio file provided"}, 400

            audio_file = request.files['audio']

            # Read audio file
            audio_data, sample_rate = sf.read(audio_file)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample to target sample rate if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
                audio_data = resample(audio_data, num_samples)

            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Prepare input for the model
            audio_input = {
                "array": audio_data,
                "sampling_rate": TARGET_SAMPLE_RATE
            }

            # Load model if not loaded
            pipe = load_model()

            # Transcribe
            with torch.no_grad():
                result = pipe(
                    audio_input,
                    return_timestamps=True
                )

            transcribed_text = result["text"].strip()

            return {
                "transcription": transcribed_text,
                #"timestamps": result.get("chunks", [])
            }, 200

        except Exception as error:
            logger.error(f"Transcription error: {error}")
            return {"error": str(error)}, 400

    def process_base64_audio(self, base64_audio):
        """Process base64 encoded audio data."""
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(base64_audio)

            # Create a bytes buffer
            audio_buffer = io.BytesIO(audio_bytes)

            # Read audio data using soundfile
            audio_data, sample_rate = sf.read(audio_buffer)
            return audio_data, sample_rate

        except Exception as e:
            raise ValueError(f"Error processing audio data: {str(e)}")

# Add resources to API
api.add_resource(Test, '/')
api.add_resource(STTService, '/transcribe')

if __name__ == '__main__':
    # Load model at startup
    load_model()

    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8030))

    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)