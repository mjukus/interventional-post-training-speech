"""Module containing data classes for evaluation datasets.

This module includes dataset, dataloader and data module classes for VoxCeleb1 and
Speech Commands, including both raw audio datasets and datasets of pre-computed
features.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import lightning as L  # noqa: N812 standard import style for Lightning
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchaudio.datasets import SPEECHCOMMANDS

from src.shared.dataset import MemMappedDataset, PrecomputedFeaturesDataloader
from src.shared.utils import as_path


def load_voxceleb_sv_pairs(
    metadata_file: str | Path,
) -> list[tuple[str, str, int]]:
    """Parse a VoxCeleb metadata file.

    Parses a metadata file containing speaker verification pairs into a list of
    labelled enrollment-test pairs.

    Args:
        metadata_file (str | Path): Path to the metadata file containing speaker
        verification pairs.

    Returns:
        list[tuple[str, str, int]]: List of labelled enrollment-test pairs. Each tuple
        contains the enrollment id, test id, and a label (1 for same speaker, 0 for
        different speakers).

    """
    metadata_file = as_path(metadata_file)
    sv_pairs = []
    with metadata_file.open(encoding="utf-8") as f:
        for line in f:
            label, enrollment, test = line.strip().split()
            sv_pairs.append((enrollment, test, int(label)))
    return sv_pairs


def load_voxceleb_gender(metadata_file: str | Path) -> dict[str, str]:
    """Load gender metadata for VoxCeleb samples.

    Args:
        metadata_file (str | Path): Path to the metadata file containing gender
        information for each speaker id in the VoxCeleb test set.

    Returns:
        dict[str, str]: A dictionary mapping speaker IDs to their corresponding
        gender.

    """
    metadata_file = as_path(metadata_file)
    gender_map = {}
    with metadata_file.open(encoding="utf-8") as f:
        for line in f:
            speaker_id, gender = line.strip().split()
            gender_map[speaker_id] = gender
    return gender_map


# ----- DATASETS FOR RAW AUDIO -----


class VoxCeleb1Test(Dataset):
    """VoxCeleb1 test dataset.

    Attributes:
        wav_dir (Path): Path to the directory containing the audio files.
        sample_rate (int): Sample rate.
        backend (str): Audio backend.
        files (list[Path]): List of paths to audio files in the dataset.

    """

    def __init__(
        self,
        data_dir: str | Path,
        sample_rate: int = 16000,
        backend: str = "torchcodec",
    ) -> None:
        """Initialise an instance of the VoxCeleb1 test dataset.

        Args:
            data_dir (str | Path): Path to the root directory of the VoxCeleb1
            dataset.
            sample_rate (int, optional): Sample rate for loading audio files. Default is
            16000 Hz.
            backend (str, optional): Backend to use for loading audio files. Supported
            values are "torchcodec" and "torchaudio". Default is "torchcodec".

        Raises:
            ValueError: If no audio files are found in the specified data directory.
            ValueError: If an unsupported backend is specified.

        """
        self.wav_dir = as_path(data_dir) / "wav"
        self.sample_rate = sample_rate
        if backend not in {"torchcodec", "torchaudio"}:
            msg = f"""Unsupported backend: {backend}. Supported backends are
            'torchcodec' and 'torchaudio'."""
            raise ValueError(msg)
        self.backend = backend
        self.files = list(self.wav_dir.rglob("*.wav"))
        if len(self.files) == 0:
            msg = f"No audio files found in {self.wav_dir}."
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of audio files in the dataset.

        Returns:
            int: The number of audio files in the dataset.

        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, dict]:
        """Load and return a waveform, sample rate, and metadata for a specific sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, int, dict]: A tuple containing the waveform, its sample
            rate and a dictionary containing metadata. In this case, the metadata is
            just the relative file path of the audio file.

        """
        file_path = self.files[idx]
        file_id = file_path.relative_to(self.wav_dir).as_posix()
        metadata = {"path": file_id}
        if self.backend == "torchcodec":
            from torchcodec.decoders import AudioDecoder  # noqa: PLC0415, I001 local import to avoid dependency for torchaudio users

            decoder = AudioDecoder(file_path, sample_rate=self.sample_rate)
            waveform = decoder.get_all_samples()
        else:
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                )
                waveform = resampler(waveform)
        return waveform, self.sample_rate, metadata


class SpeechCommandsDataset(Dataset):
    """Speech Commands dataset.

    Wrapper over torchaudio's Speech Commands dataset with an interface consistent
    with other datasets.

    Attributes:
        dataset (torchaudio.datasets.SPEECHCOMMANDS): An instance of the torchaudio
        SPEECHCOMMANDS dataset.
        noise_files: List of paths to noise files for the "_silence_" class.
        n_noise: Number of noise samples to include in the dataset.
        subset: Label for the subset of the dataset the instance represents.
        url: URL for the dataset.
        rng: Numpy random number generator for sampling noise segments.

    """

    def __init__(
        self,
        data_dir: str | Path,
        url: str = "speech_commands_v0.01",
        subset: str = "training",
        rng: np.random.Generator | None = None,
        **dataset_kwargs,  # noqa: ANN003 kwargs can have many types
    ) -> None:
        """Initialise an instance of the Speech Commands dataset.

        Args:
            data_dir (str | Path): Path to the Speech Commands dataset.
            url (str, optional): URL for the dataset. Supported values are
            "speech_commands_v0.01" and "speech_commands_v0.02". Default is
            "speech_commands_v0.01".
            subset (str, optional): Subset of the dataset to use. Supported values are
            "training", "validation" and "testing". Default is "training".
            rng (np.random.Generator, optional): Numpy random number generator for
            sampling noise segments. If None, a new generator will be created with a
            random seed. Default is None.
            **dataset_kwargs: Additional keyword arguments to pass to the torchaudio
            SPEECHCOMMANDS dataset constructor.

        """
        data_dir = as_path(data_dir)
        root_dir = data_dir.parent
        self.dataset = SPEECHCOMMANDS(root_dir, url, subset=subset, **dataset_kwargs)
        noise_dir = data_dir / url / "_background_noise_"
        self.noise_files = np.fromiter(noise_dir.glob("*.wav"), dtype=Path)
        if subset == "training":
            self.n_noise = 1800
        elif subset == "validation":
            self.n_noise = 200
        else:
            self.n_noise = 0
        self.subset = subset
        self.url = url
        self.rng = np.random.default_rng(rng)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns the total length of the dataset, including noise samples.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.dataset) + self.n_noise

    def __getitem__(self, idx: int) -> tuple[Tensor, int, dict]:
        """Load and return a waveform, sample rate, and metadata for a specific sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, int, dict]: A tuple containing the waveform, its sample
            rate and a dictionary containing metadata. The metadata includes the file
            path, speaker ID, transcript, action, object and location for the sample.
            For noise samples, the metadata includes a generated path and label of
            "_silence_" only.

        """
        if idx < len(self.dataset):
            waveform, sample_rate, *_ = self.dataset[idx]
            metadata = self.dataset.get_metadata(idx)
            metadata_dict = {}
            for i, field in enumerate([
                "path",
                "sample_rate",
                "label",
                "speaker_id",
                "utterance_number",
            ]):
                metadata_dict[field] = metadata[i]
            metadata_dict["path"] = metadata_dict["path"].replace(self.url, self.subset)
        else:
            noise_file = self.rng.choice(a=self.noise_files)

            waveform, sample_rate = torchaudio.load(noise_file)
            # Randomly crop a segment of the noise file
            start = self.rng.integers(0, waveform.size(1) - sample_rate)
            waveform = waveform[:, start : start + sample_rate]
            metadata_dict = {
                "path": f"{self.subset}/_silence_/noise_{idx}",
                "sample_rate": sample_rate,
                "label": "_silence_",
                "speaker_id": "unknown",
                "utterance_number": 0,
            }
        return waveform, sample_rate, metadata_dict


# ----- DATASETS FOR PRECOMPUTED FEATURES -----


class VoxCelebPrecomputedFeaturesDataset(Dataset):
    """Dataset for pre-computed features from VoxCeleb.

    Attributes:
        feature_dir (Path): Path to the directory containing the pre-computed feature
        files.
        feature_files (list[Path]): List of paths to the pre-computed feature files.
        layers (int | None): The index of the layer to select from the
        features. If None, all layers are returned.

    """

    def __init__(self, feature_dir: str | Path, layers: int | None = None) -> None:
        """Initialise the dataset.

        Args:
            feature_dir (str | Path): Path to the directory containing the pre-computed
            feature files.
            layers (int | None, optional): If not None, the index of the layer to select
            from the features. If None, all layers are returned. Default is None.

        Raises:
            ValueError: If no feature files are found in the specified directory.

        """
        self.feature_dir = as_path(feature_dir)
        self.feature_files = list(self.feature_dir.rglob("**/*.pt"))
        if len(self.feature_files) == 0:
            msg = f"No feature files found in {self.feature_dir}."
            raise ValueError(msg)
        self.layers = layers

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """Load and return the features and ID for a specific sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, str]: A tuple containing the features tensor and the ID
            for the sample. The features tensor has shape (n_layers, L, D) if layers is
            None, or (L, D) if layers is an int.

        """
        feature_path = self.feature_files[idx]
        file_name = feature_path.relative_to(self.feature_dir)
        audio_id = file_name.with_suffix(".wav")
        features = torch.load(feature_path)  # shape: (n_layers, L, D)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        audio_id = audio_id.as_posix()
        return features, audio_id


class VoxCelebPrecomputedFeaturesMemMappedDataset(MemMappedDataset):
    """Memory-mapped version of the VoxCelebPrecomputedFeaturesDataset.

    Attributes:
        layers (int | None): The index of the layer to select from the features. If
        None, all layers are returned.

    """

    def __init__(self, feature_dir: str | Path, layers: int | None = None) -> None:
        """Initialise the memory-mapped dataset.

        Args:
            feature_dir (str | Path): Path to the directory containing the pre-computed
            features and index file.
            layers (int | None, optional): If not None, the index of the layer to select
            from the features. If None, all layers are returned. Default is None.

        """
        feature_dir = as_path(feature_dir)
        index_file = feature_dir / "index.pt"
        super().__init__(feature_dir, index_file)
        self.layers = layers

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """Load and return the features and ID for a specific sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, str]: A tuple containing the features tensor and the ID
            for the sample. The features tensor has shape (n_layers, L, D) if layers is
            None, or (L, D) if layers is an int.

        """
        audio_id = self.keys[idx]
        features = self._get_feature_by_id(audio_id)  # shape: (n_layers, L, D)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        return features, audio_id


class SpeechCommandsPrecomputedFeaturesDataset(Dataset):
    """Dataset for pre-computed features from Speech Commands.

    Attributes:
        layers: (int | None): The index of the layer to select from the features. If
        None, all layers are returned.
        samples (list[tuple[Path, int]]): List of tuples containing the paths to the
        feature files and their corresponding label indices.
        class_mappings (dict): Dictionary containing the mappings from labels to
        indices.
        num_classes (int): The number of classes in the dataset.
        weights_per_class (Tensor): Tensor containing the weight for each class.
        sample_weights (list[float]): List of weights for each sample, to be used for
        weighted sampling.

    """

    CLASSES = (
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "_unknown_",
        "_silence_",
    )

    def _setup_class_mappings(self) -> None:
        labels_to_idx = {label: idx for idx, label in enumerate(self.CLASSES)}
        idx_to_label = {idx: label for label, idx in labels_to_idx.items()}
        self.class_mappings = {
            "labels_to_idx": labels_to_idx,
            "idx_to_label": idx_to_label,
        }
        self.num_classes = len(self.CLASSES)

    def __init__(
        self,
        features_dir: str | Path,
        subset: str,
        layers: int | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            features_dir (str | Path): Path to the directory containing the pre-computed
            feature files.
            subset (str): Subset of the dataset to use. Supported values are "training",
            "validation" and "testing".
            layers (int | None, optional): If not None, the index of the layer to select
            from the features. If None, all layers are returned. Default is None.

        """
        features_dir = as_path(features_dir)
        self.layers = layers
        self._setup_class_mappings()

        subset_feature_dir = features_dir / subset
        all_feature_paths = subset_feature_dir.rglob("**/*.pt")
        self.samples = []
        class_counts = torch.zeros(len(self.CLASSES), dtype=torch.long)
        for path in all_feature_paths:
            label = path.parent.name
            if label in self.CLASSES:
                label_idx = self.class_mappings["labels_to_idx"][label]
            else:
                label_idx = self.class_mappings["labels_to_idx"]["_unknown_"]
            self.samples.append((path, label_idx))
            class_counts[label_idx] += 1
        self.weights_per_class = 1 / class_counts
        self.sample_weights = [
            self.weights_per_class[label].item() for _, label in self.samples
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Load and return the features and label for a specific sample.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            tuple[Tensor, Tensor]: The features and label for the sample.

        """
        feature_path, label_idx = self.samples[idx]
        features = torch.load(feature_path)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        label = torch.tensor(label_idx, dtype=torch.long)
        return features, label


class SpeechCommandsPrecomputedFeaturesMemMappedDataset(
    MemMappedDataset,
    SpeechCommandsPrecomputedFeaturesDataset,
):
    """Memory-mapped version of the SpeechCommandsPrecomputedFeaturesDataset."""

    def __init__(
        self,
        features_dir: str | Path,
        subset: str,
        layers: int | None = None,
    ) -> None:
        """Initialise the memory-mapped dataset.

        Args:
            features_dir (str | Path): Path to the directory containing the pre-computed
            features and index file.
            subset (str): Subset of the dataset to use. Supported values are "training",
            "validation" and "testing".
            layers (int | None, optional): If not None, the index of the layer to select
            from the features. If None, all layers are returned. Default is None.

        """
        features_dir = as_path(features_dir)
        subset_dir = features_dir / subset
        index_file = subset_dir / "index.pt"
        super().__init__(subset_dir, index_file)
        self.layers = layers
        self._setup_class_mappings()

        self.samples = []
        class_counts = torch.zeros(len(self.CLASSES), dtype=torch.long)
        for audio_id in self.keys:
            label = Path(audio_id).parent.name
            if label in self.CLASSES:
                label_idx = self.class_mappings["labels_to_idx"][label]
            else:
                label_idx = self.class_mappings["labels_to_idx"]["_unknown_"]
            self.samples.append((audio_id, label_idx))
            class_counts[label_idx] += 1
        self.weights_per_class = 1 / class_counts
        self.sample_weights = [
            self.weights_per_class[label].item() for _, label in self.samples
        ]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Load and return the features and label for a specific sample.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            tuple[Tensor, Tensor]: The features and label for the sample.

        """
        audio_id, label_idx = self.samples[idx]
        features = self._get_feature_by_id(audio_id)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        label = torch.tensor(label_idx, dtype=torch.long)
        return features, label


# ------ DATA MODULES ------


class SpeechCommandsDataModule(L.LightningDataModule):
    """Loads Speech Commands as a Lightning DataModule.

    Attributes:
        feature_dir (Path): Path to the directory containing the pre-computed feature
        files.
        layers (int | None): The index of the layer to select from the features. If
        None, all layers are returned.
        batch_size (int): The batch size to use for the dataloaders.
        dataset_class: The dataset class to use for loading the data.
        dataloader_kwargs: Additional keyword arguments to pass to the dataloaders.

    """

    def __init__(
        self,
        feature_dir: str | Path,
        layers: int | None = None,
        batch_size: int = 32,
        *,
        mmap: bool = False,
        **dataloader_kwargs,  # noqa: ANN003 dataloader kwargs can take many types
    ) -> None:
        """Initialise the DataModule.

        Args:
            feature_dir (str | Path): Path to the directory containing the pre-computed
            feature files.
            layers (int | None, optional): The index of the layer to select from the
            features. If None, all layers are returned. Default is None.
            batch_size (int, optional): The batch size to use for the dataloaders.
            Default is 32.
            mmap (bool, optional): Whether to use memory-mapped datasets. Default is
            False.
            **dataloader_kwargs: Additional keyword arguments to pass to dataloaders.

        """
        super().__init__()
        self.feature_dir = as_path(feature_dir)
        self.layers = layers
        self.batch_size = batch_size
        self.dataset_class = (
            SpeechCommandsPrecomputedFeaturesMemMappedDataset
            if mmap
            else SpeechCommandsPrecomputedFeaturesDataset
        )
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage: str) -> None:
        """Set up the datasets for the specified stage.

        Args:
            stage (str): The stage for which to set up the datasets. Supported values
            are "fit" and "test".

        Raises:
            NotImplementedError: If an unsupported stage is specified.

        """
        if stage in {"fit", "test"}:
            self.train_dataset = self.dataset_class(
                features_dir=self.feature_dir,
                subset="training",
                layers=self.layers,
            )
            self.num_classes = self.train_dataset.num_classes
            self.class_mappings = self.train_dataset.class_mappings
            self.val_dataset = self.dataset_class(
                features_dir=self.feature_dir,
                subset="validation",
                layers=self.layers,
            )
            self.test_dataset = self.dataset_class(
                features_dir=self.feature_dir,
                subset="testing",
                layers=self.layers,
            )
        else:
            msg = f"Unsupported stage: {stage}. Supported stages are 'fit' and 'test'."
            raise NotImplementedError(msg)

    def train_dataloader(self) -> PrecomputedFeaturesDataloader:
        """Return the dataloader for the training dataset.

        Returns:
            PrecomputedFeaturesDataloader: The dataloader for the training dataset.

        """
        sampler = WeightedRandomSampler(
            self.train_dataset.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return PrecomputedFeaturesDataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> PrecomputedFeaturesDataloader:
        """Return the dataloader for the validation dataset.

        Returns:
            PrecomputedFeaturesDataloader: The dataloader for the validation dataset.

        """
        sampler = WeightedRandomSampler(
            self.val_dataset.sample_weights,
            num_samples=len(self.val_dataset),
            replacement=True,
        )
        return PrecomputedFeaturesDataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> PrecomputedFeaturesDataloader:
        """Return the dataloader for the test dataset.

        Returns:
            PrecomputedFeaturesDataloader: The dataloader for the test dataset.

        """
        return PrecomputedFeaturesDataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.dataloader_kwargs,
        )
