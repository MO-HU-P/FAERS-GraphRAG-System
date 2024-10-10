#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama server to be ready..."
sleep 30 

echo "Pulling nomic-embed-text model..."
if ! ollama pull nomic-embed-text; then
  echo "Failed to pull nomic-embed-text model"
  exit 1
fi

echo "Ollama is ready with nomic-embed-text model."

tail -f /dev/null