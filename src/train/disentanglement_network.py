"""Module containing disentanglement network classes for training.

This module contains disentanglement network classes for training, including a base
class for disentanglement networks and specific implementations of multi-layer
perceptron and auto-encoder architectures.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class DisentanglementNetwork(nn.Module, ABC):
    """Base class for disentanglement networks."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        """Initialize the disentanglement network.

        Args:
            input_dim (int): The dimensionality of the input features.
            latent_dim (int): The dimensionality of the latent space.

        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def _encode(self, h: Tensor) -> Tensor:
        """Subclass-specific latent encoding implementation."""

    def encode(self, h: Tensor) -> Tensor:
        """Encode an input to latent space.

        Args:
            h (Tensor): Input tensor, shape (B, input_dim).

        Returns:
            Tensor: Latent representation, shape (B, latent_dim).

        """
        return self._encode(h)


class MLPDisentanglementNetwork(DisentanglementNetwork):
    """Disentanglement network with multi-layer perceptron architecture.

    Attributes:
        encoder (nn.Sequential): The MLP encoder network.

    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int | list[int] = 256,
        latent_dim: int = 128,
    ) -> None:
        """Initialize the MLP disentanglement network.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int | list[int]): The dimensionality of the hidden layers. If an
            int is provided, a single hidden layer of that dimensionality will be used.
            If a list is provided, each element will specify the dimensionality of a
            hidden layer. Default is 256.
            latent_dim (int): The dimensionality of the latent space. Default is 128.

        """
        super().__init__(input_dim, latent_dim)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        modules = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            modules.append(nn.Linear(in_dim, h_dim))
            modules.append(nn.ReLU())
            in_dim = h_dim
        modules.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*modules)

    def _encode(self, h: Tensor) -> Tensor:
        return self.encoder(h)

    def forward(self, h: Tensor) -> Tensor:
        """Pass input through the model.

        Args:
            h (Tensor): Input tensor, shape (B, input_dim).

        Returns:
            Tensor: Latent representation, shape (B, latent_dim).

        """
        return self.encode(h)
