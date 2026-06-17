"""Module to pre-compute features with a pre-trained model.

This module extracts features from a pre-trained model in the S3PRL upstream hub and
saves the result to file, either with each example as a separate file, or mem-mapped
into a single large archive.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from s3prl.nn import S3PRLUpstream
from torch import Tensor
from tqdm import tqdm

from src.evaluation.dataset import SpeechCommandsDataset, VoxCeleb1Test
from src.shared.dataset import AudioDataloader
from src.train.dataset import AudioInterventionalDataset

log = logging.getLogger(__name__)


def extract_batch(
    batch: tuple[Tensor, Tensor, list],
    model: S3PRLUpstream,
    device: torch.device,
) -> tuple[Tensor, Tensor, list]:
    """Extract features for a batch of waveforms.

    Args:
        batch (tuple[Tensor, Tensor, list]): A batch of data, containing padded
        waveforms, lengths, and metadata.
        model (S3PRLUpstream): A pre-trained model from the S3PRL upstream hub to use
        for feature extraction.
        device (torch.device): The device to run the model on.

    Returns:
        tuple[Tensor, Tensor, list]: A tuple containing the extracted features, their
        lengths, and the original metadata.

    """
    waveforms, lengths, metadata = batch
    waveforms = waveforms.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        features, lengths = model(waveforms, lengths)
        lengths = lengths[0]  # All layers have the same lengths
    features = torch.stack(features, dim=1)  # Shape: (B, n_layers, L, D)
    return features, lengths, metadata


def preprocess_features(
    dataloader: AudioDataloader,
    model: S3PRLUpstream,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Pre-process an audio dataset.

    This function loops through a dataloader and extracts features using a pre-trained
    model, and saves the features to disk. Each data example is saved as a separate .pt
    file.

    Args:
        dataloader (AudioDataloader): A dataloader with batches of an audio dataset.
        model (S3PRLUpstream): A pre-trained model from the S3PRL upstream hub to use
        for feature extraction.
        output_dir (Path): The directory to save the extracted features to.
        device (torch.device): The device to run the model on.

    """
    for batch in tqdm(dataloader):
        features, lengths, metadata = extract_batch(batch, model, device)
        for i in range(features.size(0)):
            feature = (
                features[i, :, : lengths[i]].cpu().clone()
            )  # Trim to original length
            audio_path = Path(metadata[i]["path"])
            feature_path = output_dir / audio_path.with_suffix(".pt")
            feature_dir = feature_path.parent
            feature_dir.mkdir(parents=True, exist_ok=True)
            torch.save(feature, feature_path)


def preprocess_features_mmap(
    dataloader: AudioDataloader,
    model: S3PRLUpstream,
    output_dir: Path,
    device: torch.device,
    *,
    shard_size_mb: int = 512,
) -> None:
    """Pre-process an audio dataset into multiple memory-mapped shard files.

    This stores feature tensors in a set of shard binaries under ``output_dir/shards``
    and writes an index that maps each audio ID to (shard_name, offset, size, shape).

    Args:
        dataloader (AudioDataloader): A dataloader with batches of an audio dataset.
        model (S3PRLUpstream): A pre-trained model from the S3PRL upstream hub.
        output_dir (Path): Directory for shards and index file.
        device (torch.device): Device to run model on.
        shard_size_mb (int): Maximum shard size in megabytes.

    """
    if shard_size_mb <= 0:
        msg = f"Shard size must be positive. Got {shard_size_mb}."
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_idx = output_dir / "index.pt"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    max_shard_bytes = shard_size_mb * 1024 * 1024
    index_dict = {}

    shard_index = 0
    current_offset = 0
    current_shard_bytes = 0
    shard_name = f"features_{shard_index:05d}.bin"
    shard_file = (shard_dir / shard_name).open("wb")

    try:
        for batch in tqdm(dataloader, desc="Extracting features"):
            features, lengths, metadata = extract_batch(batch, model, device)
            for i in range(features.size(0)):
                feature = features[i, :, : lengths[i]].cpu().numpy().astype("float32")
                feature_bytes = feature.tobytes()
                nbytes = len(feature_bytes)

                # Check if this sample goes over target shard size.
                if (
                    current_shard_bytes > 0
                    and current_shard_bytes + nbytes > max_shard_bytes
                ):
                    shard_file.close()
                    shard_index += 1
                    shard_name = f"features_{shard_index:05d}.bin"
                    shard_file = (shard_dir / shard_name).open("wb")
                    current_offset = 0
                    current_shard_bytes = 0

                shard_file.write(feature_bytes)

                audio_id = metadata[i]["path"]
                index_dict[audio_id] = (
                    shard_name,
                    current_offset,
                    feature.size,
                    feature.shape,
                )

                current_offset += feature.size
                current_shard_bytes += nbytes
    finally:
        shard_file.close()

    log.info("Saving index file")
    torch.save(index_dict, output_idx)


def main(cfg: DictConfig) -> None:
    """Select the appropriate preprocessing function based on configuration.

    This function reads the configuration and loads the dataset and pre-trained model
    specified in the config. It then calls either `preprocess_features` or
    `preprocess_features_mmap` based on the `mmap` parameter.

    Args:
        cfg (DictConfig): OmegaConf configuration object.

    Raises:
        ValueError: If the specified dataset is not supported.

    """
    cfg = cfg.preprocess
    log.info(OmegaConf.to_yaml(cfg))  # Log config
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir) / cfg.pretrained_model.name

    if cfg.dataset == "interventional_dataset":
        dataset = AudioInterventionalDataset(
            metadata_file=cfg.metadata_file,
            data_dir=data_dir,
            sample_rate=cfg.sample_rate,
            backend=cfg.backend,
        )
        output_dir /= cfg.set
    elif cfg.dataset == "voxceleb":
        dataset = VoxCeleb1Test(
            data_dir=data_dir,
            sample_rate=cfg.sample_rate,
            backend=cfg.backend,
        )
    elif cfg.dataset == "speech-commands":
        dataset = SpeechCommandsDataset(
            data_dir=data_dir,
            url=cfg.url,
            subset=cfg.set,
            download=cfg.download,
        )
        output_dir /= cfg.set
    else:
        msg = f"""Unsupported dataset: {cfg.dataset}. Supported datasets are:
        interventional_dataset, voxceleb, fluent-speech-commands, speech-commands."""
        raise ValueError(msg)
    dataloader = AudioDataloader(dataset, batch_size=cfg.batch_size)
    model = S3PRLUpstream(cfg.pretrained_model.name)
    model.eval()

    # Device management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if cfg.mmap:
        preprocess_features_mmap(
            dataloader,
            model,
            output_dir,
            device,
            shard_size_mb=cfg.shard_size_mb,
        )
    else:
        preprocess_features(dataloader, model, output_dir, device)


if __name__ == "__main__":

    @hydra.main(
        version_base=None,
        config_path="pkg://configs",
        config_name="preprocess/default",
    )
    def run_main(cfg: DictConfig) -> None:
        """Run main with Hydra configuration.

        Args:
            cfg (DictConfig): OmegaConf configuration object passed by Hydra.

        """
        main(cfg)

    run_main()
