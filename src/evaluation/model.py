"""Model class for evaluation on keyword spotting.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import lightning as L  # noqa: N812 standard alias for lightning
import torch
from torch import Tensor, nn

from src.shared.utils import as_path


class KeywordSpottingClassifier(L.LightningModule):
    """Keyword spotting classifier model, implemented as a linear layer.

    Attributes:
        class_mappings (dict): Mapping from indices to class labels.
        subspace (int): Which subspace of disentangled representations to use.
        prediction_dir (Path): Path to the directory to save keyword spotting
        predictions.
        classifier (nn.Linear): Linear layer for keyword spotting classification.
        criterion (nn.CrossEntropyLoss): Loss function for training the classifier.
        predictions (list[Tensor]): List of predictions for the current epoch.
        labels (list[Tensor]): List of labels for the current epoch.

    """

    def __init__(
        self,
        embedding_dim: int,
        output_class_num: int,
        class_mappings: dict,
        subspace: int,
        *,
        save_dir: str | Path = "results",
    ) -> None:
        """Initialize the KeywordSpottingClassifier.

        Args:
            embedding_dim (int): Dimension of input embeddings or disentangled features.
            output_class_num (int): Number of output classes.
            class_mappings (dict): Mapping from indices to class labels.
            subspace (int): Which subspace of disentangled representations to use.
            save_dir (str | Path, optional): Directory to save predictions. Default is
            "results".

        """
        super().__init__()
        self.class_mappings = class_mappings
        self.subspace = subspace
        save_dir = as_path(save_dir)
        self.prediction_dir = save_dir / "ks_predictions" / f"subspace_{self.subspace}"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = nn.Linear(embedding_dim, output_class_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the classifier.

        """
        return self.classifier(x)

    def _label_to_str(self, label: int) -> str:
        return self.class_mappings["idx_to_label"][label]

    def _save_predictions(
        self,
        predictions: list[Tensor],
        labels: list[Tensor],
        stage: str = "train",
    ) -> None:
        # Save predictions and labels for the entire epoch
        predictions_cat = torch.cat(predictions, dim=0)
        labels_cat = torch.cat(labels, dim=0)
        preds_str = list(map(self._label_to_str, predictions_cat.cpu().tolist()))
        labels_str = list(map(self._label_to_str, labels_cat.cpu().tolist()))
        file_path = (
            self.prediction_dir / f"{stage}_predictions_epoch_{self.current_epoch}.txt"
        )
        with file_path.open("w") as f:
            for pred, label in zip(preds_str, labels_str, strict=True):
                f.write(f"Predicted: {pred}, True: {label}\n")

    def _get_loss_and_acc(
        self,
        logits: Tensor,
        labels: Tensor,
        stage: str = "train",
    ) -> tuple[Tensor, Tensor, Tensor]:
        loss = self.criterion(logits, labels)
        predicted_class = torch.argmax(logits, dim=-1)
        correct_predictions = (predicted_class == labels).float()
        batch_acc = correct_predictions.mean()
        self.log(
            f"eval/ks/{stage}/loss/subspace {self.subspace}",
            loss,
            batch_size=labels.size(0),
        )
        self.log(
            f"eval/ks/{stage}/acc/subspace {self.subspace}",
            batch_acc,
            on_epoch=True,
            on_step=False,
            batch_size=labels.size(0),
            prog_bar=True,
        )
        return loss, batch_acc, predicted_class

    def _step(
        self,
        batch: tuple[Tensor, Tensor],
        stage: str = "train",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x, labels = batch
        logits = self(x)
        loss, batch_acc, predicted_class = self._get_loss_and_acc(
            logits,
            labels,
            stage=stage,
        )
        return loss, batch_acc, predicted_class, labels

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Perform a training step.

        Args:
            batch (tuple[Tensor, Tensor]): A batch of data, containing input features
            and labels.

        Returns:
            Tensor: The loss for the batch.

        """
        loss, _, _, _ = self._step(batch)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Initialise the validation epoch."""
        self.predictions = []
        self.labels = []

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Perform a validation step.

        Args:
            batch (tuple[Tensor, Tensor]): A batch of data, containing input features
            and labels.

        Returns:
            Tensor: The loss for the batch.

        """
        loss, _, predicted_class, labels = self._step(batch, stage="val")
        self.predictions.append(predicted_class)
        self.labels.append(labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Save predictions at the end of the validation epoch."""
        self._save_predictions(self.predictions, self.labels, stage="val")

    def on_test_epoch_start(self) -> None:
        """Initialise the test epoch."""
        self.predictions = []
        self.labels = []

    def test_step(self, batch: tuple[Tensor, Tensor]) -> None:
        """Perform a test step.

        Args:
            batch (tuple[Tensor, Tensor]): A batch of data, containing input features
            and labels.

        """
        _, _, predicted_class, labels = self._step(batch, stage="test")
        self.predictions.append(predicted_class)
        self.labels.append(labels)

    def on_test_epoch_end(self) -> None:
        """Save predictions at the end of the test epoch."""
        self._save_predictions(self.predictions, self.labels, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer to use for training.

        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)
