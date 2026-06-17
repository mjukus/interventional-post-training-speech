#!/bin/bash
# Copyright (c) Jack Cox
# SPDX-License-Identifier: MIT

EXPERIMENT="joint" # Default experiment name, can be overridden by command line argument
SEED=42 # Default seed, can be overridden by command line argument
# Fail fast if no model is provided
if [ -z "$1" ]; then
    echo "Error: No model name provided. Usage: sbatch train_eval_multi.sh <model_name> [experiment_name] [seed]"
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
# Extract seed from command line argument if provided
if [ -z "$3" ]; then
    echo "No seed provided, using default: $SEED"
else
    SEED=$3
    echo "Using seed: $SEED"
fi

source .venv/bin/activate

python3 -m src.main +experiment=$EXPERIMENT dataset.model_name=$MODEL seed=$SEED
