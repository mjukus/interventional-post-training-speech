#!/bin/bash

# Fail fast if no config is provided
if [ -z "$1" ]; then
    echo "Error: No config provided. Usage: sbatch preprocess_multi.sh <config_name>"
    exit 1
fi
# Extract config name from command line argument
CONFIG=$1

source .venv/bin/activate

# Run with multiple configurations
python3 -m src.preprocess --multirun -cn "preprocess/$CONFIG.yaml"
