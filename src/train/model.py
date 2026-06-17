"""Module containing full model classes for training."""

from abc import ABC, abstractmethod

import lightning as L  # noqa: N812 standard import for lightning
import torch
from torch import Tensor

from src.shared.encoder_pooling import MeanPooling, PoolingLayer
from src.train.disentanglement_network import DisentanglementNetwork
from src.train.loss import InterventionalContrastiveLoss, OrthogonalityLoss


class DisentanglementModel(L.LightningModule, ABC):
    """Generic parent class for disentanglement models.

    Subclasses provide model-specific latent computation and loss, while this base class
    handles optional layer weighting, pooling, and latent reshaping.

    Attributes:
        pooling_layer (PoolingLayer): The pooling layer to pool frame-level features
        to utterance-level.
        subspaces (int): The number of subspaces to use.
        residual_subspace (bool): Whether to include a residual subspace.
        weighted_sum (LearnableWeightedSum | None): Optional learnable weighted sum for
        combining pretrained layers.

    """

    def __init__(
        self,
        pooling_layer: PoolingLayer,
        subspaces: int,
        *,
        residual_subspace: bool = False,
    ) -> None:
        """Initialize the disentanglement model.

        Args:
            pooling_layer (PoolingLayer): The pooling layer to pool frame-level features
            to utterance-level.
            subspaces (int): The number of subspaces to use.
            residual_subspace (bool): Whether to include a residual subspace.

        """
        super().__init__()
        self.pooling_layer = pooling_layer
        self.subspaces = subspaces
        self.residual_subspace = int(residual_subspace)

    @property
    def total_subspaces(self) -> int:
        """Total number of subspaces."""
        return self.subspaces + self.residual_subspace

    def _pool_utterance(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        return self.pooling_layer(x, lengths=lengths)

    def _reshape_latent(self, z: Tensor) -> Tensor:
        """Reshape flat latent vectors into (B, subspaces, subspace_dim)."""
        return z.view(z.size(0), self.total_subspaces, -1)

    @abstractmethod
    def _get_outputs(
        self,
        h_utt: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Pass the input through the model.

        Args:
            x (Tensor): Input tensor.
            lengths (Tensor, optional): Optional tensor of sequence lengths.

        """
        h_utt = self._pool_utterance(x, lengths=lengths)
        z, h_recon = self._get_outputs(h_utt)
        z = self._reshape_latent(z)
        return z, h_utt, h_recon

    def encode(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Encode the input into the latent space.

        Args:
            x (Tensor): Input tensor.
            lengths (Tensor, optional): Optional tensor of sequence lengths.

        """
        h_utt = self._pool_utterance(x, lengths=lengths)
        z, _ = self._get_outputs(h_utt)
        return self._reshape_latent(z)

    def predict_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Predict embeddings for a batch of input data."""
        x, lengths, _ = batch
        return self.encode(x, lengths=lengths)

    @abstractmethod
    def loss_function(
        self,
        z: Tensor,
        h_utt: Tensor,
        h_recon: Tensor | None,
        labels: Tensor,
    ) -> dict:
        """Compute the loss function for the model."""
        raise NotImplementedError


class InterventionalContrastiveDisentanglement(DisentanglementModel):
    """Interventional Contrastive Disentanglement Network.

    Attributes:
        disentanglement_network (DisentanglementNetwork): The disentanglement network.
        reconstruction (bool): Whether the disentanglement network reconstructs the
        utterance embedding.
        reconstruction_loss (nn.MSELoss | None): The reconstruction loss function, if
        applicable.
        contrastive_loss (InterventionalContrastiveLoss): The contrastive loss function.
        orthogonality_loss (OrthogonalityLoss): The orthogonality loss function.
        contrastive_weight (float): Weight for the contrastive loss term.
        orth_weight (float): Weight for the orthogonality loss term.
        recon_weight (float): Weight for the reconstruction loss term.

    """

    def __init__(
        self,
        pooling_layer: PoolingLayer,
        disentanglement_network: DisentanglementNetwork,
        subspaces: int,
        *,
        residual_subspace: bool = True,
        temperature: float = 0.1,
        additive_margin: float = 0.0,
    ) -> None:
        """Initialise the model.

        Args:
            pooling_layer (PoolingLayer): The pooling layer to pool frame-level features
            to utterance-level with.
            disentanglement_network (DisentanglementNetwork): The disentanglement
            network to compute latent representations and (optionally) reconstructions
            with.
            subspaces (int): The number of subspaces to use.
            residual_subspace (bool): Whether to include a residual subspace. Default is
            True.
            temperature (float): Temperature for the contrastive loss. Default is 0.1.
            additive_margin (float): Additive margin for the contrastive loss. Default
            is 0.0.

        """
        super().__init__(
            pooling_layer=pooling_layer,
            subspaces=subspaces,
            residual_subspace=residual_subspace,
        )
        self.disentanglement_network = disentanglement_network
        if disentanglement_network.latent_dim % self.total_subspaces != 0:
            msg = "Latent dimension must be divisible by number of subspaces."
            raise ValueError(msg)
        self.contrastive_loss = InterventionalContrastiveLoss(
            temperature=temperature,
            additive_margin=additive_margin,
        )
        self.orthogonality_loss = OrthogonalityLoss()
        self.contrastive_weight = 1.0
        self.orth_weight = 1.0
        self.recon_weight = 1.0
        self.reconstruction = False

    def _get_outputs(self, h_utt: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.reconstruction:
            z, h_recon = self.disentanglement_network(h_utt)
        else:
            z = self.disentanglement_network(h_utt)
            h_recon = None
        return z, h_recon

    def loss_function(
        self,
        z: Tensor,
        h_utt: Tensor,
        h_recon: Tensor | None,
        labels: Tensor,
    ) -> dict:
        """Compute the overall loss for the model.

        This includes contrastive loss, orthogonality loss, and (if applicable)
        reconstruction loss terms.

        Args:
            z (Tensor): Latent representations, shape (B, subspaces, subspace_dim).
            h_utt (Tensor): Pooled utterance representations, shape (B, input_dim).
            h_recon (Tensor | None): Optional reconstructed utterance representations,
            shape (B, input_dim).
            labels (Tensor): Interventional labels, shape (B, B, subspaces).

        Returns:
            dict: Dictionary containing individual loss components and total loss.

        """
        contrastive_loss = 0.0
        orthogonality_loss = torch.tensor(0.0, device=z.device)
        loss_dict = {}
        for j in range(self.subspaces):
            # Contrastive loss
            z_j = z[:, j, :]
            labels_j = labels[:, :, j]
            contrastive_loss_j = self.contrastive_loss(z_j, labels_j)
            loss_dict[f"contrastive_loss/subspace_{j}"] = contrastive_loss_j
            contrastive_loss += contrastive_loss_j
            # Orthogonality loss
            for k in range(self.subspaces + self.residual_subspace):
                if k > j:  # Avoid double counting
                    z_k = z[:, k, :]
                    orthogonality_loss_j_k = self.orthogonality_loss(z_j, z_k)
                    loss_dict[f"orthogonality_loss/subspaces_{j}_{k}"] = (
                        orthogonality_loss_j_k
                    )
                    orthogonality_loss += orthogonality_loss_j_k
        loss_dict["contrastive_loss"] = contrastive_loss
        loss_dict["orthogonality_loss"] = orthogonality_loss
        total_loss = (
            self.contrastive_weight * contrastive_loss
            + self.orth_weight * orthogonality_loss
        )
        reconstruction_loss = torch.tensor(0.0, device=z.device)
        loss_dict["reconstruction_loss"] = reconstruction_loss
        loss_dict["loss"] = total_loss
        return loss_dict


class StraightThroughModel(DisentanglementModel):
    """Baseline with pooling only.

    This model isn't trained, and accordingly only supports mean pooling.
    """

    def __init__(self, pooling_layer: MeanPooling) -> None:
        """Initialise the model.

        Args:
            pooling_layer (MeanPooling): The mean pooling layer.

        """
        if not isinstance(pooling_layer, MeanPooling):
            msg = "StraightThroughModel only supports mean pooling."
            raise TypeError(msg)
        super().__init__(
            pooling_layer=pooling_layer,
            subspaces=1,
            residual_subspace=False,
        )

    def _get_outputs(
        self,
        h_utt: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        return h_utt, None

    def loss_function(
        self,
        z: Tensor,
        h_utt: Tensor,
        h_recon: Tensor | None,
        labels: Tensor,
    ) -> dict:
        """StraightThroughModel has no trainable objective."""
        msg = "StraightThroughModel has no trainable objective."
        raise NotImplementedError(msg)
