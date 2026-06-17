#!/bin/bash
# Copyright (c) Jack Cox
# SPDX-License-Identifier: MIT

MODEL="wavlm" # Default model name, can be overridden by command line argument
# Fail fast if no config is provided
if [ -z "$1" ]; then
    echo "Error: No config provided. Usage: sbatch preprocess_multi.sh <config_name> <dataset_split> [model_name]"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: No dataset split provided. Usage: sbatch preprocess_multi.sh <config_name> <dataset_split> [model_name]"
    exit 1
else
    SET=$2
    echo "Using dataset split: $SET"
fi
if [ -z "$3" ]; then
    echo "No model name provided, using default."
else
    MODEL=$3
    echo "Using model name: $MODEL"
fi
# Extract config name from command line argument
CONFIG=$1

source .venv/bin/activate

if [ "$CONFIG" == "voxceleb" ]; then
    # For VoxCeleb, we don't use set
    python3 -m src.preprocess -cn "preprocess/$CONFIG.yaml" preprocess.pretrained_model.name=$MODEL
else
    python3 -m src.preprocess -cn "preprocess/$CONFIG.yaml" preprocess.pretrained_model.name=$MODEL preprocess.set=$SET
fi