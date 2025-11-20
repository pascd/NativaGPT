#!/bin/bash

echo "Starting Kobold restAPI..."

XTTS_REPO_PATH="/home/pedrodias/Documents/git-repos/xtts-api-server/"

# Change to the root of xtts directory
cd "${XTTS_REPO_PATH}"

# Source the python environment
source venv/bin/activate

# Run the restAPI
python3 -m xtts_api_server