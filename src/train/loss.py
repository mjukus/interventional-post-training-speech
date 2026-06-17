"""Module defining loss functions for training.

This module contains the implementation of the interventional contrastive loss and
orthogonality loss used for training.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import logging

import torch
import torch.nn.functional as F  # noqa: N812 Standard import style for torch.nn.functional
from torch import Tensor, nn

log = logging.getLogger(__name__)


def compute_cross_entropy(p: Tensor, q: Tensor) -> Tensor:
    """Compute the cross-entropy H(p, q) between two probability distributions.

    Args:
        p (Tensor): Ground-truth probability distribution.
        q (Tensor): Predicted probability distribution.

    Returns:
        Tensor: The batch cross-entropy loss. Scalar.

    """
    q = F.log_softmax(q, dim=1)
    loss = -torch.sum(p * q, dim=1)
    return loss.mean()


class InterventionalContrastiveLoss(nn.Module):
    """Compute the interventional contrastive loss.

    Adaptation of the supervised contrastive loss from Khosla et al., 2020 to multiple
    interventional subspaces.

    Attributes:
        temperature (float): Temperature parameter for scaling the logits.
        additive_margin (float): Additive margin for positive pairs, following Wang et
        al., 2018.

    """

    def __init__(self, temperature: float = 0.1, additive_margin: float = 0.0) -> None:
        """Initialise the loss.

        Args:
            temperature (float): Temperature hyperparameter. Default is 0.1.
            additive_margin (float): Additive margin hyperparameter. Default is 0.0 (no
            margin).

        """
        super().__init__()
        self.temperature = temperature
        self.additive_margin = additive_margin

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """Compute the interventional contrastive loss.

        Args:
            z (Tensor): Latent representations of shape (batch_size, latent_dim).
            labels (Tensor): Interventional labels of shape (batch_size, batch_size),
            where labels[i, j] = 1 if samples i and j are in the same interventional
            class, and 0 otherwise.

        Returns:
            Tensor: The computed interventional contrastive loss. Scalar.

        """
        z = F.normalize(z, p=2, dim=1)

        # Indicator matrix for positive pairs
        contrastive_labels = 1 - labels
        contrastive_labels.fill_diagonal_(0)  # Remove self-self pairs
        # Only the anchors which have positive pairs in the batch are valid
        batch_prime = (contrastive_labels.sum(dim=1) > 0).squeeze()
        contrastive_labels_prime = contrastive_labels[batch_prime]
        # Compute ground-truth categorical distribution
        p_i = contrastive_labels_prime.sum(dim=1, keepdim=True)
        p = contrastive_labels_prime / p_i
        # Compute logits
        logits = torch.matmul(z, z.T) / self.temperature
        logits.fill_diagonal_(-1e9)  # Mask self-self pairs
        logits = logits[batch_prime]
        # Additive-margin softmax (Wang et al., 2018) subtracts the margin from positive
        # (or target) logits only.
        if self.additive_margin != 0.0:
            logits -= contrastive_labels_prime * self.additive_margin / self.temperature
        if p_i.sum() == 0:
            msg = (
                "No positive pairs in batch; returning zero loss. Consider adjusting"
                " the batching strategy to ensure some positive pairs are present."
            )
            log.warning(msg)
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        return compute_cross_entropy(p, logits)


class OrthogonalityLoss(nn.Module):
    """Compute the orthogonality loss between two subspaces.

    Orthogonality loss to encourage different subspaces to be independent, following
    Bousmalis et al., 2016.
    """

    def __init__(self) -> None:
        """Initialise the loss."""
        super().__init__()

    def forward(self, z_j: Tensor, z_k: Tensor) -> Tensor:
        """Compute the orthogonality loss between two subspaces z_j and z_k.

        Args:
            z_j (Tensor): Latent representations from subspace j.
            z_k (Tensor): Latent representations from subspace k.

        Returns:
            Tensor: The computed orthogonality loss.

        """
        z_j_norm = F.normalize(z_j, p=2, dim=1)
        z_k_norm = F.normalize(z_k, p=2, dim=1)
        # Compute pairwise distances
        distance = torch.matmul(z_j_norm, z_k_norm.T)
        squared_distance = distance**2
        return torch.mean(squared_distance)
