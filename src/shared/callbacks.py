"""Module containing custom PyTorch Lightning callbacks.

This module defines the `LossWeightScheduler` callback, which allows for dynamic
adjustment of loss weights during training.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class LossWeightScheduler(Callback):
    """Adjust and monitor loss weights during training.

    Attributes:
        weight (str): The name of the weight attribute to adjust in the model.
        start (float): Starting value of the weight.
        end (float): Ending value of the weight.
        mode (str): Mode of adjustment. Options are "const", "linear".
        epochs (int, optional): Epoch count over which to adjust the weight.

    """

    def __init__(
        self,
        weight: str,
        start: float = 0.0,
        end: float | None = None,
        mode: str = "const",
        epochs: int | None = None,
    ) -> None:
        """Initialise the callback.

        Args:
            weight (str): The name of the weight attribute to adjust in the model.
            start (float): Starting value of the weight.
            end (float): Ending value of the weight.
            mode (str): Mode of adjustment. Options are "const", "linear".
            epochs (int, optional): Epoch count over which to adjust the weight. Default
            is None, which means it will use the total number of epochs.

        Raises:
            ValueError: If mode is "linear" and end is not specified.

        """
        self.weight = weight
        self.start = start
        if mode == "linear" and end is None:
            msg = "End value must be specified for linear mode."
            raise ValueError(msg)
        self.end = end
        self.mode = mode
        self.epochs = epochs

    @staticmethod
    def _adjust_linear(
        start: float,
        end: float,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Calculate the new weight value for linear mode.

        Args:
            start (float): The starting value of the weight.
            end (float): The ending value of the weight.
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs over which to adjust the
            weight.

        Returns:
            float: The new weight value.

        """
        if epoch < total_epochs:
            return start + (end - start) * (epoch / total_epochs)
        return end

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Adjust the weight at the start of each training epoch.

        Args:
            trainer (Trainer): A Trainer.
            pl_module (LightningModule): A LightningModule.

        Raises:
            AttributeError: If the model does not have the specified weight attribute.
            ValueError: If mode is "linear" and total epochs cannot be determined.
            ValueError: If mode is "linear" and end value is not specified.

        """
        if not hasattr(pl_module.model, self.weight):
            msg = f"Model does not have attribute '{self.weight}' to adjust."
            raise AttributeError(msg)
        if self.mode == "linear":
            epoch = trainer.current_epoch
            total_epochs = self.epochs or trainer.max_epochs
            if total_epochs is None:
                msg = "Total number of epochs could not be determined."
                raise ValueError(msg)
            if self.end is None:
                msg = "End value must be specified for linear mode."
                raise ValueError(msg)
            new_weight = self._adjust_linear(self.start, self.end, epoch, total_epochs)
        else:
            new_weight = self.start

        setattr(pl_module.model, self.weight, new_weight)
        pl_module.log(f"sched/{self.weight}", new_weight, on_step=False, on_epoch=True)
