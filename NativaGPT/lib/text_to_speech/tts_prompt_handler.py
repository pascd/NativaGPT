import json
import io
import requests

from typing import Dict, Optional, Any, Union
from pydub import AudioSegment
from pydub.playback import play


from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger

class TTSPromptHandler:
    """Handles Text-to-Speech conversion and playback."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS Prompt Handler.

        Args:
            config: Configuration dictionary containing TTS settings
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the required configuration settings."""
        required_configs = [
            "tts_config.api_url",
            "tts_config.headers",
            "tts_config.prompt",
            "tts_config.output_config.output_url",
            "tts_config.speaker_config.speakers_url"
        ]

        for config_path in required_configs:
            current = self.config
            try:
                for key in config_path.split('.'):
                    current = current[key]
            except KeyError:
                raise ValueError(f"Missing required configuration: {config_path}")

    def _make_api_request(self,
                          url: str,
                          headers: Dict[str, str],
                          data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make an API request with error handling.

        Args:
            url: API endpoint URL
            headers: Request headers
            data: Request payload

        Returns:
            Optional[Dict]: Response data or None if request failed
        """
        try:
            response = requests.post(url=url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return None

    def set_output_dir(self) -> Optional[Dict[str, Any]]:
        """Configure output directory settings."""
        config = self.config["tts_config"]["output_config"]
        return self._make_api_request(
            url=config["output_url"],
            headers=self.config["tts_config"]["headers"],
            data=config["prompt"]
        )

    def set_speakers_dir(self) -> Optional[Dict[str, Any]]:
        """Configure speaker directory settings."""
        config = self.config["tts_config"]["speaker_config"]
        return self._make_api_request(
            url=config["speakers_url"],
            headers=self.config["tts_config"]["headers"],
            data=config["prompt"]
        )

    def convert_tts_prompt(self, input_text: str) -> Dict[str, str]:
        """
        Convert input text to TTS prompt format.

        Args:
            input_text: Text to be converted to speech

        Returns:
            Dict[str, str]: Formatted prompt for TTS API
        """
        try:
            prompt_template = self.config["tts_config"]["prompt"].copy()
            return {
                field: value.format(text=input_text)
                for field, value in prompt_template.items()
            }
        except KeyError as e:
            logger.error(f"Missing prompt template field: {e}")
            raise ValueError(f"Invalid prompt template configuration: {e}")

    def send_tts_prompt(self, input_text: str) -> bool:
        """
        Send text to TTS service and play the response.

        Args:
            input_text: Text to be converted to speech

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            prompt = self.convert_tts_prompt(input_text=input_text)
            logger.info(f"Sending TTS prompt: {prompt}")

            response = requests.post(
                url=self.config["tts_config"]["api_url"],
                headers=self.config["tts_config"]["headers"],
                json=prompt
            )
            response.raise_for_status()

            return self.play_tts_response(response)

        except Exception as e:
            logger.error(f"Failed to process TTS request: {str(e)}")
            return False

    def play_tts_response(self, response: requests.Response) -> bool:
        """
        Play audio from TTS response.

        Args:
            response: Response from TTS API

        Returns:
            bool: True if playback successful, False otherwise
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            play(audio)
            return True
        except Exception as e:
            logger.error(f"Failed to play audio: {str(e)}")
            return False

def main():
    """Main execution function."""
    logger.basicConfig(level=logger.INFO)
    logger = logger.getLogger(__name__)

    try:
        config_path = "/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json"
        config_manager = ConfigManager(config_path=config_path)
        config = config_manager.get()

        tts_handler = TTSPromptHandler(config)

        # Initialize TTS settings
        if not tts_handler.set_output_dir():
            logger.error("Failed to set output directory")
            return

        if not tts_handler.set_speakers_dir():
            logger.error("Failed to set speakers directory")
            return

        # Test TTS functionality
        success = tts_handler.send_tts_prompt(input_text="Hello, this is a test message.")
        if not success:
            logger.error("Failed to process TTS request")
            return

        logger.info("TTS test completed successfully")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()