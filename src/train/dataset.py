"""Module containing dataset classes for training.

This module contains dataset, dataloader and datamodule classes for training, including
an abstract base class for interventional datasets, an interventional dataloader that
extends the PrecomputedFeaturesDataloader to generate interventional labels, and a batch
sampler for sampling batches from a dataset with dense interventions. It also includes
specific dataset classes for interventional datasets of audio and pre-computed features,
and a Pytorch Lightning data module which returns the training dataloaders.
"""

import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from src.shared.dataset import MemMappedDataset, PrecomputedFeaturesDataloader
from src.shared.utils import as_path

log = logging.getLogger(__name__)


class InterventionalDataset(Dataset, ABC):
    """Base class for datasets that expose metadata for interventions.

    Interventional samplers and dataloaders rely on a tabular ``metadata`` field with
    one row per sample and intervention columns available by name.
    """

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        """Return per-sample metadata used for interventional batching."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the metadata.

        """
        return len(self.metadata)


class InterventionalDataloader(PrecomputedFeaturesDataloader):
    """DataLoader for interventional datasets.

    This DataLoader composes batches of features and generates interventional labels
    based on specified metadata columns. It can handle both nested tensors and padded
    tensors for variable-length features.

    Attributes:
        interventions (list[str]): List of column names in the dataset metadata to use
        as interventions.

    """

    def __init__(
        self,
        dataset: InterventionalDataset,
        interventions: list[str],
        *,
        use_nested_tensors: bool = False,
        **kwargs,  # noqa: ANN003 dataloader kwargs have many types
    ) -> None:
        """Initialize the InterventionalDataloader.

        Args:
            dataset (InterventionalDataset): An interventional dataset, which should
            have metadata available for the specified interventions.
            interventions (list[str]): List of column names in the dataset metadata to
            use as interventions. Interventional labels will be generated based on these
            columns.
            use_nested_tensors (bool, optional): Whether to use nested tensors for
            variable-length inputs. Default: False.
            **kwargs: Additional keyword arguments to pass to the DataLoader
            constructor.

        """
        super().__init__(
            dataset,
            use_nested_tensors=use_nested_tensors,
            **kwargs,
        )
        self.interventions = interventions

    def _collate_labels(self, labels: list[pd.Series]) -> Tensor:
        """Generate pairwise interventional labels from sample metadata.

        Returns:
            Tensor of shape (batch_size, batch_size, num_interventions), where each
            entry [i, j, k] is 1 if samples i and j differ in intervention k, and 0 if
            they are the same in intervention k.

        """
        metadata = pd.DataFrame(labels)
        batch_size = len(labels)
        interventional_labels = torch.ones((
            batch_size,
            batch_size,
            len(self.interventions),
        ))
        for i, intervention in enumerate(self.interventions):
            values = metadata[intervention].to_numpy()
            correspondence_mask = values[:, None] == values[None, :]
            interventional_labels[:, :, i] = 1 - torch.tensor(
                correspondence_mask.astype(int),
            )
        return interventional_labels


class InterventionalDataModule(LightningDataModule):
    """Lightning DataModule for interventional datasets.

    Attributes:
        train_dataset (InterventionalDataset): Training dataset.
        val_dataset (InterventionalDataset): Validation dataset.
        interventions (list[str]): List of column names in the dataset metadata to use
        as interventions.
        use_interventions (list[str]): List of interventions to load as labels.
        batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        dataloader_kwargs: Additional keyword arguments to pass to the DataLoader.
        train_sampler (InterventionalBatchSampler): Batch sampler for training data.
        val_sampler (InterventionalBatchSampler): Batch sampler for validation data.

    """

    def __init__(
        self,
        train_dataset: InterventionalDataset,
        val_dataset: InterventionalDataset,
        interventions: list[str],
        batch_size: int = 128,
        val_batch_size: int | None = None,
        *,
        shuffle: bool = False,
        use_interventions: list[str] | None = None,
        train_batch_shape: list[int] | None = None,
        **dataloader_kwargs,  # noqa: ANN003 dataloader kwargs have many types
    ) -> None:
        """Initialize the InterventionalDataModule.

        Args:
            train_dataset (InterventionalDataset): Dataset with training data.
            val_dataset (InterventionalDataset): Dataset with validation data.
            interventions (list[str]): List of column names in the dataset metadata to
            use as interventions.
            batch_size (int, optional): Batch size. Default: 128.
            val_batch_size (int, optional): Batch size for validation. If None, uses
            training batch size. Default: None.
            shuffle (bool, optional): Whether to shuffle the data. Default: False.
            use_interventions (list[str] | None, optional): List of interventions to use
            for the dataloader. If None, uses all interventions. Default: None.
            **dataloader_kwargs: Additional keyword arguments to pass to the DataLoader.

        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.interventions = interventions
        self.use_interventions = (
            use_interventions if use_interventions is not None else interventions
        )
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.dataloader_kwargs = dataloader_kwargs

        self.train_sampler = InterventionalBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            interventions=self.interventions,
            shuffle=shuffle,
            batch_shape=train_batch_shape,
        )
        self.val_sampler = InterventionalBatchSampler(
            self.val_dataset,
            batch_size=self.val_batch_size,
            interventions=self.interventions,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> InterventionalDataloader:
        """Return the training dataloader.

        Returns:
            InterventionalDataloader: The training dataloader.

        """
        return InterventionalDataloader(
            self.train_dataset,
            self.use_interventions,
            batch_sampler=self.train_sampler,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> InterventionalDataloader:
        """Return the validation dataloader.

        Returns:
            InterventionalDataloader: The validation dataloader.

        """
        return InterventionalDataloader(
            self.val_dataset,
            self.use_interventions,
            batch_sampler=self.val_sampler,
            **self.dataloader_kwargs,
        )


class InterventionalBatchSampler(Sampler):
    """Sample batches of indices from a dataset with dense interventions.

    Sampler that yields a dense interventional batch, from a dataset with dense
    interventions (all combinations of intervention values).

    Attributes:
        dataset (InterventionalDataset): The interventional dataset to sample from.
        batch_size (int): The sizes of the produced batches.
        interventions (list[str]): List of column names in the dataset metadata to use
        as interventions.
        shuffle (bool): Whether to shuffle the data at each epoch.
        intervention_values (list[int]): The number of unique values for each
        intervention.
        indices_matrix (Tensor): A tensor of shape (n1, n2, ...,) where n_i is the
        number of unique values for intervention i, containing the dataset indices
        arranged according to the interventions.
        batch_shape (Tensor): A tensor of shape (num_interventions,) containing the
        size of a batch along each intervention dimension.
        n_intervention_batches (Tensor): A tensor of shape (num_interventions,)
        containing the number of batches along each intervention dimension.
        total_batches (int): The total number of batches per epoch.

    """

    def __init__(
        self,
        dataset: InterventionalDataset,
        batch_size: int,
        interventions: list[str],
        *,
        shuffle: bool = False,
        batch_shape: list[int] | None = None,
    ) -> None:
        """Initialize the InterventionalBatchSampler.

        Args:
            dataset (InterventionalDataset): The dataset to sample from, which should
            have metadata with the specified interventions, maximal interventional
            density, and be arranged with nested interventions (e.g. 10 speakers, 5
            sentences each, with data order speaker1 (sentence1, sentence2, ...),
            speaker2 (sentence1, sentence2, ...), etc.)).
            batch_size: Batch size.
            interventions: List of intervention column names, in order of nesting in the
            dataset.
            shuffle (bool, optional): Whether to shuffle the data. Default: False.
            batch_shape (list[int], optional): Shape of the batch in terms of
            intervention factors, e.g. [2,2] for a batch size of 4 with 2 interventions.
            If not provided, the best factors for the dataset and batch size will be
            determined automatically. Default: None.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.interventions = interventions
        self.shuffle = shuffle
        indices = list(range(len(dataset)))

        # Find the number of unique values for each intervention
        self.intervention_values = []
        for intervention in interventions:
            self.intervention_values.append(
                self.dataset.metadata[intervention].nunique(),
            )
        # Reshape indices into N-D array with a dimension for each intervention
        self.indices_matrix = torch.tensor(indices).reshape(
            self.intervention_values,
        )
        # There may be a way to make this work for non-dense datasets, but the batch
        # construction would be complicated, so for now we required dense interventions. TODO

        # Determine batch shape
        if batch_shape is None:
            self.batch_shape = self._get_factors(batch_size, self.intervention_values)
        else:
            self.batch_shape = torch.tensor(batch_shape)
        if self.batch_shape.prod() != batch_size:
            msg = (
                f"Invalid batch shape {self.batch_shape} for batch size {batch_size}. "
                "The product of the batch shape must equal the batch size."
            )
            raise ValueError(msg)
        if any(self.batch_shape > torch.tensor(self.intervention_values)):
            msg = (
                f"Invalid batch shape {self.batch_shape} for intervention values "
                f"{self.intervention_values}. Each batch dimension must be less than or"
                " equal to the corresponding number of intervention values."
            )
            raise ValueError(msg)
        if len(self.batch_shape) != len(self.interventions):
            msg = (
                f"Invalid batch shape {self.batch_shape} for interventions "
                f"{self.interventions}. The batch shape must have the same number of "
                "dimensions as the number of interventions."
            )
            raise ValueError(msg)

        # Calculate total number of batches
        self.n_intervention_batches = (
            torch.tensor(self.intervention_values) // self.batch_shape
        )
        self.total_batches = int(self.n_intervention_batches.prod())
        log.info("Total batches per epoch: %d", self.total_batches)

    def _get_factors(self, batch_size: int, values: list[int]) -> Tensor:
        """Find optimal factors for each intervention.

        Optimal factors multiply to the batch size and are as close to each other as
        possible.
        """
        intervention_factors = {}
        for i, value in enumerate(values):
            factors = [
                f
                for f in range(2, batch_size + 1)
                if batch_size % f == 0 and value % f == 0
            ]
            if not factors:
                msg = (
                    "Cannot find suitable factor for intervention "
                    f"{self.interventions[i]} with class size {value} and batch size "
                    f"{batch_size}"
                )
                raise ValueError(msg)
            intervention_factors[self.interventions[i]] = factors

        factor_combinations = list(itertools.product(*intervention_factors.values()))
        factor_combinations = [
            i for i in factor_combinations if torch.tensor(i).prod() == batch_size
        ]
        if factor_combinations == []:
            msg = (
                f"No valid combination of factors of class sizes {values} that multiply"
                f" to batch size {batch_size}. Consider adjusting the batch size."
            )
            raise ValueError(msg)

        # Select combination with smallest std deviation (most balanced)
        factor_combination_stds = [
            torch.tensor(c).float().std().item() for c in factor_combinations
        ]
        selected_factors = factor_combinations[
            factor_combination_stds.index(min(factor_combination_stds))
        ]
        return torch.tensor(selected_factors)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices.

        Returns:
            Iterator[list[int]]: An iterator that yields lists of dataset indices for
            each batch.

        """
        # Shuffle along each intervention dimension if specified
        indices_matrix = self.indices_matrix.clone()
        if self.shuffle:
            for dim in range(indices_matrix.dim()):
                idx = torch.randperm(indices_matrix.size(dim))
                indices_matrix = indices_matrix.index_select(dim, idx)

        # Discard samples that don't fit into a full batch
        used_data_size = self.batch_shape * self.n_intervention_batches
        reduced_indices = indices_matrix[tuple(slice(0, s) for s in used_data_size)]
        # A batch is a block from the N-D array, defined by selected_factors, e.g. for a
        # (4,6) array:
        # [[1 1 2 2 3 3],
        #  [1 1 2 2 3 3],
        #  [4 4 5 5 6 6],
        #  [4 4 5 5 6 6]]
        # We reshape the indices array into components for an intervention dimension and
        # selected factor, so for this example a 4-D array of shape (2,2,3,2). Each
        # batch is a slice of size (1,2,1,2) from this 4-D array.
        intermediate_shape = []
        for d, b in zip(self.n_intervention_batches, self.batch_shape, strict=True):
            intermediate_shape.extend([d, b])
        transformed_indices = reduced_indices.reshape(intermediate_shape)
        grid_indices = list(range(0, 2 * len(self.interventions), 2))
        block_indices = list(range(1, 2 * len(self.interventions), 2))
        batch_indices = transformed_indices.permute(
            *grid_indices,
            *block_indices,
        ).reshape(-1, self.batch_size)

        # Restrict to start_batch and end_batch if specified
        for i in range(self.total_batches):
            yield batch_indices[i].tolist()

    def __len__(self) -> int:
        """Return the total number of batches per epoch.

        Returns:
            int: The number of batches.

        """
        return self.total_batches


class PrecomputedPretrainedFeaturesDataset(InterventionalDataset):
    """Dataset class for pre-computed features for an interventional dataset.

    Attributes:
        _metadata (pd.DataFrame): The dataset metadata loaded from the metadata file.
        feature_dir (Path): Directory containing pre-computed feature files.
        layers (int | None): Specific layers to select from the features. If None, all
        layers are used.

    """

    def __init__(
        self,
        feature_dir: str | Path,
        metadata_file: str | Path,
        layers: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            feature_dir (str | Path): Directory containing pre-computed feature files.
            metadata_file (str | Path): File containing dataset metadata.
            layers (int | None): Specific layers to select from the features. If None,
            all layers are used. Default: None.

        """
        metadata_file = as_path(metadata_file)
        self._metadata = pd.read_csv(metadata_file, sep="\t", quoting=3)
        self.feature_dir = as_path(feature_dir)
        self.layers = layers

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the dataset metadata.

        Returns:
            pd.DataFrame: The dataset metadata loaded from the metadata file.

        """
        return self._metadata

    def __getitem__(self, idx: int) -> tuple[Tensor, pd.Series]:
        """Return the features and metadata for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, pd.Series]: A tuple containing the features tensor and the
            corresponding metadata row as a pandas Series.

        """
        row = self.metadata.iloc[idx]
        feature_path = self.feature_dir / Path(row["path"]).with_suffix(".pt")
        features = torch.load(feature_path)  # shape: (n_layers, L, D)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        return features, row


class PrecomputedPretrainedFeaturesMemMappedDataset(
    MemMappedDataset,
    PrecomputedPretrainedFeaturesDataset,
):
    """Memory-mapped version of the PrecomputedPretrainedFeaturesDataset.

    Attributes:
        _metadata (pd.DataFrame): The dataset metadata loaded from the metadata file.
        feature_dir (Path): Directory containing pre-computed feature files.
        layers (int | None): Specific layers to select from the features. If None, all
        layers are used.

    """

    def __init__(
        self,
        feature_dir: str | Path,
        metadata_file: str | Path,
        layers: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            feature_dir (str | Path): Directory containing pre-computed feature files.
            metadata_file (str | Path): File containing dataset metadata.
            layers (int | None): Specific layers to select from the features. If None,
            all layers are used. Default: None.

        """
        feature_dir = as_path(feature_dir)
        index_file = feature_dir / "index.pt"
        metadata_file = as_path(metadata_file)
        super().__init__(feature_dir, index_file)
        self._metadata = pd.read_csv(metadata_file, sep="\t", quoting=3)
        self.layers = layers

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset, determined by the length of the
            metadata.

        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[Tensor, pd.Series]:
        """Return the features and metadata for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, pd.Series]: A tuple containing the features tensor and the
            corresponding metadata row as a pandas Series.

        """
        row = self.metadata.iloc[idx]
        sample_id = row["path"]
        features = self._get_feature_by_id(sample_id)  # shape: (n_layers, L, D)
        if self.layers is not None:
            features = features[self.layers]  # Select specified layers
        return features, row


class AudioInterventionalDataset(Dataset):
    """Dataset class for interventional datasets of audio.

    Attributes:
        _metadata (pd.DataFrame): The dataset metadata loaded from the metadata file.
        data_dir (Path): Directory containing audio files.
        sample_rate (int): Sample rate to load audio at.
        backend (str): Backend to use for loading audio. Supported options are
        "torchcodec" and "torchaudio".

    """

    def __init__(
        self,
        metadata_file: str | Path,
        data_dir: str | Path,
        sample_rate: int = 16000,
        backend: str = "torchcodec",
    ) -> None:
        """Initialize the dataset.

        Args:
            metadata_file (str | Path): Path to file containing dataset metadata.
            data_dir (str | Path): Path to directory containing audio files.
            sample_rate (int): Sample rate to load audio at. Default: 16000.
            backend (str): Backend to use for loading audio. Supported options are
            "torchcodec" and "torchaudio". Default: "torchcodec".

        Raises:
            ValueError: If the metadata file format is not supported.

        """
        metadata_file = as_path(metadata_file)
        if metadata_file.suffix == ".tsv":
            self._metadata = pd.read_csv(metadata_file, sep="\t", quoting=3)
        elif metadata_file.suffix == ".csv":
            self._metadata = pd.read_csv(metadata_file, sep=",", quoting=3)
        else:
            msg = "Unsupported metadata file format. Use .tsv or .csv"
            raise ValueError(msg)
        self.data_dir = as_path(data_dir)
        self.sample_rate = sample_rate
        self.backend = backend

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the metadata.

        """
        return len(self._metadata)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, pd.Series]:
        """Return the audio sample, sample rate, and metadata for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, int, pd.Series]: A tuple containing the audio tensor, the
            sample rate, and the corresponding row of metadata.

        Raises:
            ValueError: If the backend specified for loading audio is not supported.

        """
        row = self._metadata.iloc[idx]
        audio_path = self.data_dir / row["path"]
        if self.backend == "torchcodec":
            from torchcodec.decoders import AudioDecoder  # noqa: PLC0415, I001 local import to avoid dependency for torchaudio users

            decoder = AudioDecoder(audio_path, sample_rate=self.sample_rate)
            # Check if start_time and end_time columns are present
            if "start_time" in row and "end_time" in row:
                samples = decoder.get_samples_played_in_range(
                    row["start_time"],
                    row["end_time"],
                )
            else:
                samples = decoder.get_all_samples()
        elif self.backend == "torchaudio":
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                )
                waveform = resampler(waveform)
            if "start_time" in row and "end_time" in row:
                start_sample = int(row["start_time"] * self.sample_rate)
                end_sample = int(row["end_time"] * self.sample_rate)
                samples = waveform[:, start_sample:end_sample]
            else:
                samples = waveform
        else:
            msg = f"Unsupported backend: {self.backend}."
            raise ValueError(msg)
        return samples, self.sample_rate, row
