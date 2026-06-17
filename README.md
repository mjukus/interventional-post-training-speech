# Learning task-specific subspaces via interventional post-training of speech foundation models

This is the official repository for the paper "[Learning task-specific subspaces via interventional post-training of speech foundation models](https://arxiv.org/abs/2606.17967)" in Proc. Interspeech '26. [![doi: 10.5281/zenodo.20735589](https://zenodo.org/badge/DOI/10.5281/zenodo.20735589.svg)](https://doi.org/10.5281/zenodo.20735588)

## Installation

To install the project, clone the repository:

```bash
git clone https://github.com/mjukus/interventional-post-training-speech
```

We recommend using uv for environment management---installing all the project's dependencies is as simple as:

```bash
cd interventional-post-training-speech
uv sync --extra cpu # For cpu builds of PyTorch
uv sync --extra cu124 # For cuda 12.4 builds of PyTorch
```

## Additional Requirements

Evaluation on speaker verification requires the user to place the VoxCeleb1 test set in the `data/VoxCeleb1` directory. Note this directory already contains annotations for the gender of the speakers, produced by us. Content evaluation requires SpeechCommands, but this will be downloaded by default with torchaudio. Similarly, the test-clean set of LibriTTS will also be downloaded automatically when generating the synthetic interventional dataset.

Generating the synthetic dataset and preprocessing features both rely on an available audio backend, e.g. ffmpeg or libsndfile, which should be installed separately.

## Generating Synthetic Data

Generating the synthetic dataset requires an apptainer image of [F5-TTS](https://github.com/SWivid/F5-TTS), which can be built from their docker image:

```bash
sudo apptainer build f5-tts.sif docker://ghcr.io/swivid/f5-tts:main
```

Place this image at interventional-dataset/f5-tts.sif, and run scripts/generate_dataset.sh. It is possible to run multiple instances of this simultaneously for faster dataset generation. By default, the dataset is written to `data/synthetic`.

## Preprocessing Speech Foundation Model Features

Provided the required synthetic, SpeechCommands and VoxCeleb1 datasets are present in the data directory, preprocessing involves running one of the two preprocessing scripts with the appropriate config as a command line argument, e.g.:

```bash
bash scripts/preprocess.sh interventional [model_name] [set] # optional model and set arguments
bash scripts/preprocess_multi.sh interventional # hydra multirun covers all models and sets
```

Note that the set argument isn't used for VoxCeleb1, as we only use the test set.

## Reproducing Our Experiments

We focus on providing the basic functionality required to reproduce our experiments, but the experiments are customisable through changing configs, or providing command line overrides to hydra entry points. That said, running the basic experiments is simple using the provided scripts:

```bash
bash scripts/train_eval.sh wavlm [experiment] [seed] # Run a single job on one seed
bash scripts/train_eval_multi.sh wavlm [experiment] # Run multiple jobs over 5 seeds
```

The choice of backbone model is a required argument. Options for `experiment` are baseline, content_only, speaker_only, and joint. Refer to the paper for details on these different setups.

These scripts perform training and evaluation sequentially, but the two parts can be separated by using the hydra entry points in `src/train_model.py` and `src/evaluate_model.py`. These can be run with:

```bash
uv run -m src.train_model ...
```

See the [hydra docs](hydra.cc) for more information on command-line overrides, and multirun logic.

## Output Structure

By default, all experiment outputs are written to the `outputs` directory. The top level within this directory is an experiment, while subdirectories for a run are automatically generated from hydra command-line overrides. Outputs for each run have the following structure:

- .hydra
    - configs for the run
- checkpoints
- logs
    - tensorboard/wandb output
- results
    - train
    - eval
- main.log (main log file)

## Logging

By default, metrics for the runs are logged to Tensorboard. To view a run's metrics, use:

```bash
uv run tensorboard --logdir=outputs/[experiment]/[run]/logs
```

## License

This project is licensed under the MIT license. See LICENSE.txt for the full license.