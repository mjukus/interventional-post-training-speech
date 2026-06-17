"""Module with model and data set up and training logic.

This module contains functions for setting up loggers, callbacks, and the training loop
for the Interventional Contrastive Disentanglement Network. It uses Hydra for
configuration management and Lightning for training.
"""

import logging
import os
from pathlib import Path

import lightning as L  # noqa: N812 standard import for lightning
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.shared.callbacks import LossWeightScheduler
from src.shared.utils import setup_loggers
from src.train.experiment import InterventionalDisentanglementExperiment
from src.train.model import DisentanglementModel

log = logging.getLogger(__name__)


def _setup_callbacks(
    checkpoint_dir: Path,
    loss_weights: DictConfig,
    patience: int,
) -> tuple[list, EarlyStopping | None]:
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_{epoch}-{step}",
            monitor="val/loss",
            save_last=True,
        ),  # There seems to be a bug where this saves best checkpoints and the last checkpoint is a copy of that, rather than from the last epoch.
        LossWeightScheduler("contrastive_weight", **loss_weights.contrastive),
        LossWeightScheduler("orth_weight", **loss_weights.orth),
        LossWeightScheduler("recon_weight", **loss_weights.recon),
    ]
    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=patience,
            mode="min",
        )
        callbacks.append(early_stopping)
    return callbacks, early_stopping


def train_model(cfg: DictConfig) -> None:
    """Set up and train a model.

    Args:
        cfg (DictConfig): Configuration object containing all settings for training.

    """
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("CPU count: %d", os.cpu_count())
    log.info("Thread count: %d", torch.get_num_threads())
    log.info("Working directory: %s", Path.cwd())

    # Set up directories and loggers
    output_dir = Path(cfg.output_dir) / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    loggers = setup_loggers(
        cfg.job_id,
        cfg.job_name,
        cfg.logging,
        output_dir,
        job_config=cfg,
    )
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "last.ckpt"

    # Seeding
    L.seed_everything(cfg.seed, workers=True)

    # Data setup
    data_module = instantiate(cfg.dataset.data_module)

    # Instantiate model and setup weighted sum if needed
    model = instantiate(cfg.model)

    # Set up experiment, callbacks, and trainer
    optimizer_factory = instantiate(cfg.optimizer)
    scheduler_factory = instantiate(cfg.scheduler)
    experiment = InterventionalDisentanglementExperiment(
        model=model,
        optimizer=optimizer_factory,
        scheduler=scheduler_factory,
        output_dir=output_dir,
    )
    callbacks, early_stopping = _setup_callbacks(
        checkpoint_dir,
        cfg.loss_weights,
        cfg.patience,
    )
    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    # Train model
    if checkpoint_path.exists():
        log.info("Resuming from checkpoint: %s", checkpoint_path)
        trainer.fit(experiment, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        log.info("No checkpoint found. Starting training from scratch.")
        trainer.fit(experiment, datamodule=data_module)
    experiment.eval()

    if early_stopping is not None and early_stopping.stopping_reason_message:
        log.info(
            "Early stopping triggered:\n%s",
            early_stopping.stopping_reason_message,
        )


if __name__ == "__main__":
    import hydra

    from src.shared.utils import hydra_dirname

    if not OmegaConf.has_resolver("dirname"):
        OmegaConf.register_new_resolver("dirname", resolver=hydra_dirname)

    @hydra.main(version_base=None, config_path="pkg://configs", config_name="default")
    def main(cfg: DictConfig) -> None:
        """Hydra entry point for training.

        Args:
            cfg (DictConfig): Configuration object containing all settings for training.

        """
        try:
            train_model(cfg)
        except Exception:
            log.exception("An error occurred during training.")
            raise

    main()
