#!/bin/bash
#
# Launches LM Studio's CLI server as a local, OpenAI-compatible LLM backend.
# Requires the LM Studio AppImage (https://lmstudio.ai) and its "lms" CLI.
#
# Override the defaults below by exporting the corresponding env var before
# running this script, e.g.:
#   LMS_APPIMAGE_DIR=/opt/lmstudio LMS_MODEL_NAME=my-model ./launch_llm_lms.sh

echo "Starting LM Studio from CLI..."

# Define LM Studio AppImage path
LMS_APPIMAGE_DIR="${LMS_APPIMAGE_DIR:-$HOME/Downloads}"
LMS_APPIMAGE_NAME="LM-Studio*.AppImage"

# Change directory to where the AppImage is located
cd "${LMS_APPIMAGE_DIR}" || {
    echo "Error: Unable to access ${LMS_APPIMAGE_DIR}"
    exit 1
}

# Check if the AppImage exists
APPIMAGE_PATH=$(ls ${LMS_APPIMAGE_NAME} 2>/dev/null)
if [[ -z "$APPIMAGE_PATH" ]]; then
    echo "Error: LM Studio AppImage not found in ${LMS_APPIMAGE_DIR}"
    exit 1
fi

# Start LM Studio in the background
echo "Launching LM Studio..."
./"${APPIMAGE_PATH}" --no-sandbox &

# Wait for LM Studio to initialize
echo "Waiting for LM Studio to start..."
while ! pgrep -f "LM-Studio" >/dev/null; do
    sleep 3
done

# Define the model
MODEL_NAME="${LMS_MODEL_NAME:-lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf}"

# Check if the model is available, if not, download it
echo "Checking if model is available..."
if ! lms list-models | grep -q "$MODEL_NAME"; then
    echo "Model not found. Downloading..."
    lms download "$MODEL_NAME"
fi

# Load the model
echo "Loading model: ${MODEL_NAME}"
lms load "$MODEL_NAME"

# Wait until the model is fully loaded
echo "Waiting for model to load..."
# while ! lms list-models --loaded | grep -q "$MODEL_NAME"; do
#     echo "Model is not loaded yet. Retrying in 3 seconds..."
#     sleep 3
# done

# Start the server
PORT=8000
echo "Starting LM Studio server on port ${PORT}..."
lms serve --port "${PORT}"

echo "LM Studio is now running with model '${MODEL_NAME}' on port ${PORT}"
