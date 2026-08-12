#!/bin/bash
#
# Launches KoboldCpp's REST API server as a local, OpenAI-compatible LLM
# backend. Requires a separate clone of https://github.com/LostRuins/koboldcpp
#
# Override the defaults below by exporting the corresponding env var before
# running this script, e.g.:
#   KOBOLD_REPO_PATH=/opt/koboldcpp ./launch_kobold_restAPI.sh

echo "Starting Kobold restAPI..."

KOBOLD_REPO_PATH="${KOBOLD_REPO_PATH:-$HOME/git-repos/koboldcpp}"
KOBOLD_CONFIG="${KOBOLD_CONFIG:-$KOBOLD_REPO_PATH/config/config_1.kcpps}"

# Run from project root using module syntax
cd "${KOBOLD_REPO_PATH}"
python3 -m koboldcpp --config ${KOBOLD_CONFIG} --gpulayers -1 --usecublas