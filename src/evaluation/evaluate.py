"""Module containing functions for speaker verification evaluation.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch.loggers import Logger
from s3prl.downstream.sv_voxceleb1.utils import EER
from torch import Tensor, nn
from torch.nn import CosineSimilarity

from src.evaluation.utils import plot_sv_scores

log = logging.getLogger("evaluate")


def _eer(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> tuple:
    """Calculate EER, FAR, FRR, and accuracy.

    Calculates EER, FAR, FRR, and accuracy at the specified threshold. If no threshold
    is provided, calculates EER and threshold using the S3PRL EER function. If a
    threshold is provided, calculates FAR, FRR, and accuracy at that threshold.

    Args:
        labels: Binary labels (1 for same speaker, 0 for different speakers).
        scores: Similarity scores.
        threshold: Optional threshold to calculate EER at. If None, calculate EER and
        threshold using the EER function.

    Returns:
        eer: Equal Error Rate.
        threshold: Threshold at which EER is calculated.
        far: False Acceptance Rate.
        frr: False Rejection Rate.
        acc: Accuracy.

    """
    eer = None
    if threshold is None:
        eer, threshold = EER(labels, scores)
    far = np.sum((scores >= threshold) & (labels == 0)) / np.sum(labels == 0)
    frr = np.sum((scores < threshold) & (labels == 1)) / np.sum(labels == 1)
    acc = np.mean((scores >= threshold) == labels)
    if eer is None and not np.isnan(frr):
        eer, threshold = EER(labels, scores)
    elif eer is None:
        eer = np.float64("nan")
    return eer, threshold, far, frr, acc


def calculate_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    category: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate EER for each category and overall."""
    rows = []
    eer_overall, thres_overall, far_overall, frr_overall, acc_overall = _eer(
        labels,
        scores,
    )
    rows.append({
        "Category": "Overall",
        "EER": eer_overall,
        "Threshold": thres_overall,
        "FAR": far_overall,
        "FRR": frr_overall,
        "Accuracy": acc_overall,
        "Sample Count": len(labels),
    })
    if category is not None:
        categories = np.unique(category)
        for cat in categories:
            cat_indices = category == cat
            n_samples = np.sum(cat_indices)
            eer_cat, thres_cat, far, frr, acc = _eer(
                labels[cat_indices],
                scores[cat_indices],
                threshold=thres_overall,
            )
            rows.append({
                "Category": cat,
                "EER": eer_cat,
                "Threshold": thres_cat,
                "FAR": far,
                "FRR": frr,
                "Accuracy": acc,
                "Sample Count": n_samples,
            })
    return pd.DataFrame(rows)


def log_dataframe(logger: Logger, df: pd.DataFrame) -> None:
    """Log values from a DataFrame to a logger."""
    for _, row in df.iterrows():
        subspace = row["Subspace"]
        category = row["Category"]
        eer = row["EER"]
        threshold = row["Threshold"]
        far = row["FAR"]
        frr = row["FRR"]
        acc = row["Accuracy"]
        n_samples = row["Sample Count"]
        logger.log_metrics(
            {
                f"eval/subspace {subspace}/{category}/EER": eer,
                f"eval/subspace {subspace}/{category}/Threshold": threshold,
                f"eval/subspace {subspace}/{category}/FAR": far,
                f"eval/subspace {subspace}/{category}/FRR": frr,
                f"eval/subspace {subspace}/{category}/Accuracy": acc,
                f"eval/subspace {subspace}/{category}/Sample Count": n_samples,
            },
        )


def evaluate_speaker(
    z: dict[str, Tensor],
    sv_pairs: list[tuple[str, str, int]],
    subspaces: int,
    metric: nn.Module = CosineSimilarity,
    scores_path: Path | None = None,
    plot_path: Path | None = None,
    eer_path: Path | None = None,
    metadata: dict | None = None,
    logger: Logger | None = None,
) -> None:
    """Evaluate representations on speaker verification.

    Args:
        z: Dictionary of embeddings for each sample.
        sv_pairs: List of tuples containing (enrollment sample, test sample, label).
        subspaces: Number of subspaces in the representation.
        metric: Similarity metric to use for calculating scores.
        scores_path: Optional path to save the scores.
        plot_path: Optional path to save a violin plot.
        eer_path: Optional path to save EER results as a CSV.
        metadata: Optional dictionary mapping sample IDs to categories for analysis.
        logger: Optional logger to log results to.

    """
    # Calculate scores for each subspace using the specified metric
    log.info("Calculating scores for each subspace using %s", metric.__name__)
    scores = np.zeros((len(sv_pairs), subspaces))
    labels = np.zeros((len(sv_pairs)), dtype=np.int32)
    category = np.empty((len(sv_pairs)), dtype="<U2")
    similarity = metric(dim=-1)
    for i, (enrollment, test, label) in enumerate(sv_pairs):
        z_enroll = z[enrollment]
        z_test = z[test]
        scores[i, :] = similarity(z_enroll, z_test)
        labels[i] = label
        if metadata is not None:
            enrollment_id = enrollment.split("/")[0]
            test_id = test.split("/")[0]
            category[i] = metadata[enrollment_id] + metadata[test_id]
    # Save scores if a path is provided
    if scores_path:
        with scores_path.open("w") as f:
            f.write(" ".join([f"Subspace_{j}" for j in range(subspaces)]) + " label\n")
            for i in range(scores.shape[0]):
                f.write(
                    " ".join([f"{scores[i, j]}" for j in range(subspaces)])
                    + f" {labels[i]}\n",
                )
    # Plot violin plot of scores
    if plot_path or logger:
        fig = plot_sv_scores(scores, labels, save_path=plot_path)
        if logger and hasattr(logger, "log_image"):
            logger.log_image(key="eval/Speaker Verification Scores", images=[fig])
    # Evaluate EER for each subspace
    log.info("Evaluating EER for each subspace")
    subspace_dfs = []
    for j in range(subspaces):
        scores_j = scores[:, j]
        eer_df = calculate_eer(
            labels,
            scores_j,
            category if metadata is not None else None,
        )
        eer_df.insert(0, "Subspace", j)  # Add subspace column
        subspace_dfs.append(eer_df)
    full_df = pd.concat(subspace_dfs, ignore_index=True)
    log.info("EER results:\n%s", full_df.to_string(index=False))
    if logger:
        log_dataframe(logger, full_df)
    if eer_path:
        eer_path = eer_path.with_suffix(".csv")
        full_df.to_csv(eer_path, index=False)
        log.info("Saved EER results to %s", eer_path)
