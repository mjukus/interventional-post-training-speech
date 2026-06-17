#!/bin/bash

source .venv/bin/activate

# Generate the dataset metadata and ref audios
python3 interventional-dataset/main.py

# Check for GPU in CUDA_VISIBLE_DEVICES
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    DEVICE="cpu"
else
    DEVICE="cuda"
fi

OUTPUT_DIR="data/synthetic"
SET="train"

# Generate TTS audios
apptainer exec --nv interventional-dataset/f5-tts.sif bash scripts/_generate_exhaustive.sh -d "$DEVICE" $OUTPUT_DIR/$SET"_set.tsv" $OUTPUT_DIR

