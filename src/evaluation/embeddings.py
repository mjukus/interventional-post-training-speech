"""Module for handling pre-extracted embeddings datasets and dataloaders."""

import lightning as L  # noqa: N812 standard import for lightning
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class PreExtractedEmbeddingsDataset(Dataset):
    """Dataset wrapping pre-extracted embeddings from a frozen model.

    This dataset holds pre-computed embeddings and labels, allowing training
    classifiers without re-extracting features from a frozen model each epoch.

    Attributes:
        embeddings (Tensor): Pre-extracted embeddings.
        labels (Tensor): Labels for each embedding.

    """

    def __init__(self, embeddings: Tensor, labels: Tensor) -> None:
        """Initialise the pre-extracted embeddings dataset.

        Args:
            embeddings (Tensor): Pre-extracted embeddings of shape
            (num_samples, embedding_dim).
            labels (Tensor): Labels for each embedding of shape (num_samples,).

        Raises:
            ValueError: If embeddings and labels have different first dimension.

        """
        if embeddings.size(0) != labels.size(0):
            msg = (
                f"Embeddings and labels must have the same number of samples."
                f"Got {embeddings.size(0)} embeddings and {labels.size(0)} labels."
            )
            raise ValueError(msg)
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            int: The number of samples in the dataset.

        """
        return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return an embedding and its corresponding label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple of (embedding, label).

        """
        return self.embeddings[idx], self.labels[idx]


class PreExtractedEmbeddingsDataModule(L.LightningDataModule):
    """Lightning DataModule for pre-extracted embeddings.

    Attributes:
        train_embeddings (Tensor): Pre-extracted training embeddings.
        train_labels (Tensor): Training labels.
        val_embeddings (Tensor): Pre-extracted validation embeddings.
        val_labels (Tensor): Validation labels.
        test_embeddings (Tensor): Pre-extracted test embeddings.
        test_labels (Tensor): Test labels.
        dataloader_kwargs (dict): Additional keyword arguments for dataloaders.

    """

    def __init__(
        self,
        train_embeddings: Tensor,
        train_labels: Tensor,
        val_embeddings: Tensor,
        val_labels: Tensor,
        test_embeddings: Tensor | None = None,
        test_labels: Tensor | None = None,
        **dataloader_kwargs,  # noqa: ANN003 dataloader kwargs take many types
    ) -> None:
        """Initialise the pre-extracted embeddings datamodule.

        Args:
            train_embeddings (Tensor): Pre-extracted training embeddings of shape
                (num_train, embedding_dim).
            train_labels (Tensor): Training labels of shape (num_train,).
            val_embeddings (Tensor): Pre-extracted validation embeddings of shape
                (num_val, embedding_dim).
            val_labels (Tensor): Validation labels of shape (num_val,).
            test_embeddings (Tensor | None, optional): Pre-extracted test embeddings.
                If None, defaults to val_embeddings. Default is None.
            test_labels (Tensor | None, optional): Test labels. If None, defaults to
                val_labels. Default is None.
            dataloader_kwargs (dict): Additional keyword arguments for dataloaders.

        """
        super().__init__()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_labels = val_labels
        self.test_embeddings = (
            test_embeddings if test_embeddings is not None else val_embeddings
        )
        self.test_labels = test_labels if test_labels is not None else val_labels
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for the specified stage.

        Args:
            stage (str | None, optional): The stage for which to set up the datasets.
                Supported values are "fit", "validate", and "test". Default is None.

        """
        if stage in ("fit", "validate", None):
            self.train_dataset = PreExtractedEmbeddingsDataset(
                self.train_embeddings,
                self.train_labels,
            )
            self.val_dataset = PreExtractedEmbeddingsDataset(
                self.val_embeddings,
                self.val_labels,
            )
        if stage in ("test", None):
            self.test_dataset = PreExtractedEmbeddingsDataset(
                self.test_embeddings,
                self.test_labels,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns:
            DataLoader: The dataloader for the training dataset.

        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns:
            DataLoader: The dataloader for the validation dataset.

        """
        return DataLoader(
            self.val_dataset,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns:
            DataLoader: The dataloader for the test dataset.

        """
        return DataLoader(
            self.test_dataset,
            **self.dataloader_kwargs,
        )
