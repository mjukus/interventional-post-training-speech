"""Module containing data classes shared between training and evaluation.

This module defines AudioDataloader and PrecomputedFeaturesDataloader classes, with
custom collate functions for batching variable-length audio data, and the
MemMappedDataset class, a base class for datasets that load features from memory-mapped
files.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class AudioDataloader(DataLoader):
    """Load batched variable-length audio datasets.

    A DataLoader wrapper for audio datasets, with a custom collate function that pads
    variable-length waveforms and collects their lengths and metadata.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32, **kwargs) -> None:  # noqa: ANN003 dataloader kwargs have many types
        """Initialize the dataloader.

        Args:
            dataset (Dataset): The dataset to load batches from.
            batch_size (int): The number of samples per batch.
            **kwargs: Additional keyword arguments to pass to the DataLoader
            constructor.

        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[Tensor, int, dict]],
    ) -> tuple[Tensor, Tensor, list]:
        """Construct a batch from a list of samples, padding waveforms to length.

        Args:
            batch (list): A list of samples.

        Returns:
            tuple[Tensor, Tensor, list]: A batch of padded waveforms, their lengths, and
            metadata.

        """
        waveforms = []
        lengths = []
        metadata_list = []
        for samples, _, metadata in batch:
            waveform = samples.data  # shape: (C, L)
            length = waveform.size(1)
            waveforms.append(waveform.squeeze(0))  # shape: (L)
            lengths.append(int(length))
            metadata_list.append(metadata)
        padded_waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms,
            batch_first=True,
        )  # shape: (B, L_max)
        lengths = torch.tensor(lengths)  # shape: (B,)
        return padded_waveforms, lengths, metadata_list


class PrecomputedFeaturesDataloader(DataLoader):
    """Dataloader for precomputed features.

    Loads precomputed features, with a collate function that allows for variable length
    batch items implemented with jagged-layout nested tensors.

    Attributes:
       use_nested_tensors: (bool, optional): Whether to use nested tensors for
       variable-length inputs. Default: False

    """

    DIM_WITH_LAYERS = 3  # Dimension of features tensor with multiple pre-trained layers

    def __init__(
        self,
        dataset: Dataset,
        *,
        use_nested_tensors: bool = False,
        **kwargs,  # noqa: ANN003 dataloader kwargs can take many types
    ) -> None:
        """Initialise an instance of the dataloader.

        Args:
            dataset (Dataset): A torch dataset of pre-computed features.
            use_nested_tensors (bool): Whether or not to use nested tensors for
            variable-length inputs. If False, the collate function will pad the features
            to the length of the longest item in the batch. If True, the collate
            function will return nested tensors with jagged layout. Default is False.
            **kwargs: Additional keyword arguments to pass to the parent DataLoader.

        """
        super().__init__(
            dataset,
            collate_fn=self.collate_fn,
            **kwargs,
        )
        self.use_nested_tensors = use_nested_tensors

    def _collate_labels(self, labels: list) -> Tensor | list:
        if labels and isinstance(labels[0], Tensor):
            return torch.stack(labels)  # shape: (B, ...)
        return labels

    def collate_fn(
        self,
        batch: list[tuple[Tensor, Any]],
    ) -> tuple[Tensor, Tensor | None, Tensor | list[Any]]:
        """Collate a batch from a list of features and labels.

        Args:
            batch (list): A list of tuples, where each tuple contains a features tensor
            and a label.

        Returns:
            tuple[Tensor, Tensor | None, list[str] | Tensor]: Batch features,
            optional unpadded sequence lengths and batch labels.

        """
        features = []
        lengths = []
        labels = []
        for sample_features, label in batch:
            lengths.append(sample_features.size(-2))
            if sample_features.dim() == self.DIM_WITH_LAYERS:
                l_first_features = sample_features.transpose(
                    0,
                    1,
                )  # shape: (L, n_layer, D), transposed
            else:
                l_first_features = sample_features
            features.append(l_first_features)
            labels.append(label)
        if self.use_nested_tensors:
            composed_features = torch.nested.nested_tensor(
                features,
                layout=torch.jagged,
            )  # shape: (B, j1, [n_layers,] D)
            lengths = None
        else:
            composed_features = torch.nn.utils.rnn.pad_sequence(
                features,
                batch_first=True,
            )  # shape: (B, L_max, [n_layers,] D)
            lengths = torch.tensor(lengths, dtype=torch.long)  # shape: (B,)
        if composed_features.dim() == self.DIM_WITH_LAYERS + 1:
            composed_features = composed_features.transpose(
                1,
                2,
            )  # shape: (B, n_layers, L_max, D)
        collated_labels = self._collate_labels(labels)
        return composed_features, lengths, collated_labels


class MemMappedDataset(Dataset):
    """Load a memory-mapped dataset.

    Dataset wrapper for memory-mapped datasets, allowing for efficient loading of large
    datasets that don't fit in memory. The dataset is expected to be stored in shard
    binaries under ``<feature_dir>/shards``, with an index file that maps sample IDs to
    (shard_name, offset, size, shape). The __getitem__ method should be implemented in
    a subclass, to match the specific structure and labels of the dataset.

    Attributes:
        shard_dir (Path): Path to the directory containing shard files.
        index (dict): A dictionary mapping sample IDs to shard entry metadata.
        keys (list): A list of sample IDs in the dataset.
        shard_maps (dict[str, np.memmap]): Open memmaps for each shard file.

    """

    METADATA_DIM = 4  # Expected number of elements in each index entry

    def __init__(
        self,
        feature_dir: Path,
        index_file: Path,
        *,
        copy_on_read: bool = True,
    ) -> None:
        """Initialize the MemMappedDataset.

        Args:
            feature_dir (Path): Feature directory containing shard files.
            index_file (Path): Path to the index file.
            copy_on_read (bool): Whether to copy data from the memmap on read. If False,
            returned tensors will share memory with the memmap, and should not be
            modified.

        """
        self.shard_dir = feature_dir / "shards"
        if not self.shard_dir.is_dir():
            msg = f"Expected shard directory at {self.shard_dir}."
            raise FileNotFoundError(msg)
        self.index = torch.load(index_file, weights_only=False)  # Load index metadata
        self.keys = list(self.index.keys())
        self.copy_on_read = copy_on_read
        self.shard_maps = {}

        if len(self.keys) == 0:
            return

        first_entry = self.index[self.keys[0]]
        if len(first_entry) != self.METADATA_DIM:
            msg = (
                "Expected shard index entries of the form "
                "(shard_name, offset, size, shape)."
            )
            raise ValueError(msg)

    def _get_shard_map(self, shard_name: str) -> np.memmap:
        if shard_name not in self.shard_maps:
            shard_path = self.shard_dir / shard_name
            self.shard_maps[shard_name] = np.memmap(
                shard_path,
                dtype=np.float32,
                mode="r",
            )
        return self.shard_maps[shard_name]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.keys)

    def _get_feature_by_id(self, sample_id: str) -> Tensor:
        if sample_id not in self.index:
            msg = f"ID {sample_id} not found in memmap index."
            raise KeyError(msg)
        shard_name, offset, size, shape = self.index[sample_id]
        data_map = self._get_shard_map(shard_name)
        flat_feature = data_map[offset : offset + size]
        if self.copy_on_read:
            flat_feature = flat_feature.copy()
        return torch.from_numpy(flat_feature).reshape(shape)

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset by index.

        The __getitem__ method should be implemented in a subclass, to match the
        specific structure and labels of the dataset. This method should use the
        _get_feature_by_id method to retrieve features from the memory-mapped file based
        on sample IDs.

        Args:
            idx (int): The index of the sample to retrieve.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """
        msg = "The __getitem__ method should be implemented in a subclass."
        raise NotImplementedError(msg)
