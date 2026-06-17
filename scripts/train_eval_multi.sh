#!/bin/bash

EXPERIMENT="joint" # Default experiment name, can be overridden by command line argument
# Fail fast if no model is provided
if [ -z "$1" ]; then
    echo "Error: No model name provided. Usage: sbatch train_eval_multi.sh <model_name> [experiment_name]"
    exit 1
fi
# Extract model name from command line argument
MODEL=$1
if [ -z "$2" ]; then
    echo "No experiment name provided, using default: $EXPERIMENT"
else
    EXPERIMENT=$2
    echo "Using experiment name: $EXPERIMENT"
fi

source .venv/bin/activate

# Run with multiple configurations
python3 -m src.main --multirun dataset.model_name=$MODEL +experiment=$EXPERIMENT