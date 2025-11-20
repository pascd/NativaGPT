import requests
import json
import base64
import os

def encode_file_to_base64(file_path):
    """Reads a file and encodes it to a base64 string."""
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def send_tts_request(api_url, text, speaker_wav_path, language, output_path):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    # Encode the speaker WAV file to base64
    speaker_wav_encoded = encode_file_to_base64(speaker_wav_path)

    if not speaker_wav_encoded:
        print("Failed to encode speaker WAV file. Exiting...")
        return

    payload = {
        "text": text,
        "speaker_wav": speaker_wav_encoded,  # Send encoded string
        "language": language,
        "file_name_or_path": output_path     # Path where the generated audio will be saved
    }

    try:
        print("Sending request to TTS API...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            print("TTS Request Successful! Audio file should be saved at the specified location.")
            response_data = response.json()

            # Assuming the API response returns a confirmation message
            print(f"Server response: {response_data}")

        else:
            print(f"Error: {response.status_code}")
            print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # API endpoint
    api_url = "http://localhost:8020/tts_to_audio/"

    # Example input values
    text_input = input("Enter the text to synthesize: ")
    speaker_wav_input = "/home/pedrodias/Documents/git-repos/xtts-api-server/example/female.wav"
    language_input = input("Enter the language code (e.g., 'en', 'fr', etc.): ")
    output_path = input("Enter the output file path (e.g., '/home/user/output.wav'): ")

    # Send request to TTS API
    send_tts_request(api_url, text_input, speaker_wav_input, language_input, output_path)
