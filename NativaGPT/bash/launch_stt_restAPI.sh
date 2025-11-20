#!/bin/bash

echo "Starting Speech-to-text (STT) restAPI..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Run from project root using module syntax
cd "${PROJECT_ROOT}"
python3 -m NativaGPT.lib.speech_to_text.restAPI_stt