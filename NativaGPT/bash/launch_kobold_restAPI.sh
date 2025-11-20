#!/bin/bash

echo "Starting Kobold restAPI..."

KOBOLD_CONFIG="/home/pedrodias/Documents/koboldcpp/config/config_1.kcpps"
KOBOLD_REPO_PATH="/home/pedrodias/Documents/git-repos/koboldcpp/"

# Run from project root using module syntax
cd "${KOBOLD_REPO_PATH}"
python3 -m koboldcpp --config ${KOBOLD_CONFIG} --gpulayers -1 --usecublas