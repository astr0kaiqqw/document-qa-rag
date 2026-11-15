#!/usr/bin/env bash
set -euo pipefail
# Starts only a loopback server; does not stop an existing Ollama instance.
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
exec ollama serve
