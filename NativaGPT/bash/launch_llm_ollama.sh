#!/bin/bash

echo "Starting Ollama from CLI..."

if ! ollama --version >/dev/null 2>&1; then
    echo "Error: ollama is not installed or not working properly."
    exit 1
fi

# Start ollama
ollama serve

# Run the model
ollama run deepseek-r1:latest "Only give brief suggestions and complete the json commands you receive according to the received message."


