"""Module containing utility functions needed for training and evaluation.

This module includes functions for converting strings to paths and nested tensors to
padded tensors, visualising latent spaces with t-SNE, processing Weights and Biases job
ids, setting up loggers, and generating Hydra directory names.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from matplotlib.figure import Figure
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.manifold import TSNE
from torch import Tensor


def as_path(path: str | Path) -> Path:
    """Return a Path regardless of whether input is str or Path."""
    return Path(path)


def convert_nested_to_padded(
    x: Tensor,
    padding_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Convert a nested tensor to a padded tensor.

    Args:
        x (Tensor): A nested tensor with jagged layout.
        padding_value (float): The value with which to pad.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the padded tensor and the original
        sequence lengths.

    Raises:
        ValueError: If the input tensor is not a nested tensor.

    """
    if x.is_nested:
        offsets = x.offsets()
        lengths = offsets[1:] - offsets[:-1]
        padded_size = list(x.shape)
        padded_size[1] = max(lengths).item()
        padded_tensor = torch.nested.to_padded_tensor(
            x,
            output_size=padded_size,
            padding=padding_value,
        )
        return padded_tensor, lengths
    msg = "Input tensor is not a nested tensor."
    raise ValueError(msg)


def plot_tsne(
    x_embedded: np.ndarray,
    labels: np.ndarray | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Plot t-SNE embeddings, and optionally save the figure.

    Args:
        x_embedded (np.ndarray): 2D array of shape (n_samples, 2) containing t-SNE
        embeddings.
        labels (np.ndarray | None): Optional array of shape (n_samples,) containing
        class labels.
        save_path (Path | None): Optional path to save the figure. If None, the figure
        is not saved.

    Returns:
        Figure: The matplotlib figure containing the t-SNE plot.

    """
    fig, ax = plt.subplots(figsize=(8, 8))
    if labels is not None:
        n_classes = len(np.unique(labels))
        # Use a palette with enough colors for all classes
        palette = sns.color_palette("husl", n_colors=n_classes)
        sns.scatterplot(
            x=x_embedded[:, 0],
            y=x_embedded[:, 1],
            hue=labels,
            style=labels,
            palette=palette,
            alpha=0.7,
            ax=ax,
            legend=False,
        )
    else:
        sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], alpha=0.7, ax=ax)
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def visualise_latent_space(
    x: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor | None = None,
    *,
    save_path: Path | None = None,
) -> Figure:
    """Visualise a latent space using PCA and t-SNE.

    Args:
        x (np.ndarray | torch.Tensor): The latent representations to visualise.
        labels (np.ndarray | torch.Tensor | None): Optional labels for coloring the
        plot. Default is None.
        save_path (Path | None): Optional path to save the t-SNE embeddings to. Default
        is None.

    """
    x_numpy = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    x_embedded = TSNE().fit_transform(x_numpy)
    labels_numpy = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    )
    fig = plot_tsne(x_embedded, labels_numpy)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if labels_numpy is not None:
            np.savez(save_path, X_embedded=x_embedded, labels=labels_numpy)
        else:
            np.savez(save_path, X_embedded=x_embedded)
    return fig


def get_wandb_job_id(job_id: str, job_name: str) -> str:
    """Process a experiment job id to a valid wandb id.

    Weights and Biases job ids cannot contain :;,#?/' characters and are limited to 128
    characters, so we process our job ids to fit these requirements.

    Args:
        job_id (str): The original job id, which may contain invalid characters and be
        too long.
        job_name (str): The name of the job, which will be used as a prefix in the
        processed id.

    Returns:
        str: A valid Weights and Biases job id.

    """
    max_len = 128
    exp_id = re.sub(r"[:;,#\/?]", "_", job_id)
    if len(exp_id) > max_len:
        parameters = job_id.replace(f"{job_name}_", "")
        exp_id = hash(parameters)
        exp_id = f"{job_name}_{exp_id}"
    return exp_id


def setup_loggers(
    job_id: str,
    job_name: str,
    log_config: DictConfig,
    output_dir: Path,
    *,
    job_config: DictConfig | None = None,
) -> list[Logger]:
    """Set up loggers based on the provided configuration.

    Args:
        job_id (str): The unique identifier for the training job.
        job_name (str): The name of the training job.
        log_config (DictConfig): Configuration for logging.
        output_dir (Path): The directory where logs should be saved.
        job_config (DictConfig | None): Optional configuration for the job, passed to
        the logger.

    Returns:
        list[Logger]: A list of configured loggers.

    """
    loggers = []
    loggers.append(CSVLogger(save_dir=output_dir))
    if log_config.logger == "wandb":
        wandb.login()
        job_id = get_wandb_job_id(job_id, job_name)
        config = (
            OmegaConf.to_container(job_config, resolve=True)
            if job_config is not None
            else OmegaConf.to_container(log_config, resolve=True)
        )
        wandb_logger = WandbLogger(
            id=job_id,
            save_dir=log_config.log_dir,
            project=log_config.project_name,
            config=config,
        )
        loggers.append(wandb_logger)
    elif log_config.logger == "tensorboard":
        tb_logger = TensorBoardLogger(
            save_dir=log_config.log_dir,
            name=job_name,
        )
        loggers.append(tb_logger)
    return loggers


def hydra_dirname(
    task_overrides: ListConfig,
    exclude_patterns: ListConfig,
    separator: str = ",",
) -> str:
    """Generate a Hydra directory name.

    Take a list of task overrides and exclude patterns to generate a directory name.

    Args:
        task_overrides (ListConfig): A list of Hydra command line overrides.
        exclude_patterns (ListConfig): A list of regex patterns to exclude.
        separator (str): The separator to use when joining the valid overrides.

    Returns:
        str: The generated Hydra dirname.

    """
    valid_overrides = []
    for override in task_overrides:
        should_exclude = any(
            re.search(pattern, override) for pattern in exclude_patterns
        )
        if not should_exclude:
            valid_overrides.append(override)
    return separator.join(valid_overrides)
