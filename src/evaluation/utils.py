"""Utility functions for evaluation.

This module contains a function to plot speaker verification scores on a violin plot.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def plot_sv_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Path | None = None,
    **kwargs,  # noqa: ANN003 allow arbitrary kwargs
) -> Figure:
    """Plot violin plot of speaker verification scores.

    Args:
        scores (np.ndarray): Array of shape (num_samples, num_subspaces).
        labels (np.ndarray): Array of shape (num_samples,) with ground truth labels.
        save_path (Path, optional): Path to save the plot. If None, the plot is not
        saved. Default is None.
        **kwargs: Additional keyword arguments to pass to sns.violinplot.

    Returns:
        Figure: The matplotlib figure object containing the plot.

    """
    # Prepare data for plotting
    n_scores = scores.shape[0]
    n_subspaces = scores.shape[1]
    scores = scores.flatten()
    labels_repeated = np.repeat(labels, n_subspaces)
    subspace_ids = np.tile(np.arange(n_subspaces), n_scores)
    df = pd.DataFrame({
        "Score": scores,
        "Label": labels_repeated,
        "Subspace": subspace_ids,
    })
    df = df.assign(
        Label=lambda x: x["Label"].map({0: "Impostor", 1: "Genuine"}),
        inplace=True,
    )  # Map from binary to string labels
    # Plot violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Label", y="Score", hue="Subspace", ax=ax, **kwargs)
    ax.set_title("Speaker Verification Scores by Subspace")
    ax.set_xlabel("Subspace")
    ax.set_ylabel("Cosine Similarity Score")
    if save_path:
        plt.savefig(save_path)
    return fig
