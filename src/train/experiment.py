"""Module containing the experiment class for training.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, overload

import lightning as L  # noqa: N812 standard import for lightning
import matplotlib.pyplot as plt
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch import Tensor
from torch.optim import Optimizer

from src.shared.utils import visualise_latent_space
from src.train.model import DisentanglementModel


class OptimizerFactory(Protocol):
    """Factory protocol that builds an optimizer from model parameters."""

    def __call__(self, params: Iterable[Tensor]) -> Optimizer:
        """Call the factory to create an optimizer instance.

        Args:
            params (Iterable[Tensor]): The parameters of the model to optimize.

        Returns:
            Optimizer: An instance of a PyTorch optimizer initialized with the given
            parameters.

        """
        ...


class SchedulerFactory(Protocol):
    """Factory protocol for optional LR scheduler construction."""

    @overload
    def __call__(self, optimizer: Optimizer) -> LRSchedulerTypeUnion: ...

    @overload
    def __call__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
    ) -> LRSchedulerTypeUnion: ...


class InterventionalDisentanglementExperiment(L.LightningModule):
    """Experiment module for Interventional Contrastive Disentanglement Network.

    Attributes:
        model (DisentanglementModel): The model to be trained/evaluated.
        optimizer (OptimizerFactory): A factory that returns an optimizer instance
        when called with model parameters.
        scheduler (SchedulerFactory): An optional factory that returns a scheduler
        instance when called with the optimizer.
        latent_dir (Path): Directory in which to save latent embeddings.

    """

    def __init__(
        self,
        model: DisentanglementModel,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory | None = None,
        output_dir: Path | str = "results",
    ) -> None:
        """Initialize the experiment module.

        Args:
            model (DisentanglementModel): The model to be trained/evaluated.
            optimizer (OptimizerFactory): A factory that returns an optimizer
            instance when called with model parameters.
            scheduler (SchedulerFactory): An optional factory that returns a
            scheduler instance when called with the optimizer.
            output_dir (Path): The directory to save outputs to.

        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        output_dir = Path(output_dir)
        self.latent_dir = output_dir / "latent_embeddings"

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Pass input through the model.

        Args:
            x (Tensor): Input tensor, shape (B, L, D).
            lengths (Tensor): Optional tensor of shape (B,).

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: The output tensors from the model.

        """
        return self.model(x, lengths)

    def training_step(
        self,
        batch: tuple[Tensor, Tensor | None, Tensor],
    ) -> Tensor:
        """Perform a training step.

        Args:
            batch (tuple): A batch from the dataloader, containing a feature tensor, an
            optional lengths tensor, and a label tensor.

        Returns:
            Tensor: The loss for the batch.

        """
        x, lengths, labels = batch
        self.curr_device = x.device
        z, h_utt, h_recon = self.forward(x, lengths)
        loss = self.model.loss_function(z, h_utt, h_recon, labels)
        self.log_dict({f"train/{key}": val.item() for key, val in loss.items()})
        return loss["loss"]

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor | None, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Perform a validation step.

        Args:
            batch (tuple): A batch from the dataloader, containing a feature tensor, an
            optional lengths tensor, and a label tensor.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The loss for the batch.

        """
        x, lengths, labels = batch
        self.curr_device = x.device
        z, h_utt, h_recon = self.forward(x, lengths)
        loss = self.model.loss_function(z, h_utt, h_recon, labels)
        self.log_dict({f"val/{key}": val.item() for key, val in loss.items()})
        # Visualise each subspace for one batch
        if batch_idx == 0:
            for j in range(self.model.subspaces):
                z_j = z[:, j, :]
                labels_j = labels[:, :, j]
                # Convert interventional labels to class labels
                class_labels = labels_j.argmin(dim=0)
                fig = visualise_latent_space(
                    z_j,
                    labels=class_labels,
                    save_path=self.latent_dir
                    / f"subspace_{j}"
                    / f"epoch{self.current_epoch}.npz",
                )
                for logger in self.loggers:
                    if hasattr(logger, "log_image"):
                        logger.log_image(
                            key=f"val/Subspace {j}/t-SNE",
                            images=[fig],
                        )
                plt.close()
        return loss["loss"]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers and schedulers for training.

        Returns:
            OptimizerLRScheduler: A tuple containing the optimizer and an optional
            scheduler.

        """
        optimizer = self.optimizer(self.model.parameters())
        if self.scheduler is None:
            return optimizer
        # Scheduler setup
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler_sig = inspect.signature(self.scheduler)
        if "total_steps" in scheduler_sig.parameters:
            scheduler = self.scheduler(optimizer, total_steps=total_steps)
            interval = "step"
        else:
            scheduler = self.scheduler(optimizer)
            interval = "epoch"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "name": "sched/lr",
                "monitor": "val/loss",
                "interval": interval,
            },
        }
