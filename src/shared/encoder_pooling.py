"""Module defining pooling layers for variable-length sequences.

This module contains an implementation of mean pooling for variable-length sequences.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from src.shared.utils import convert_nested_to_padded


class PoolingLayer(nn.Module, ABC):
    """Base class for pooling layers."""

    @abstractmethod
    def _pool(self, x: Tensor, lengths: Tensor | None = None) -> Tensor: ...

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Pass input through the pooling layer.

        Args:
            x (Tensor): Input tensor of shape (B, sequence_length, feature_dim).
            lengths (Tensor | None): Optional tensor of shape (B,) containing
            the lengths of each sequence in the batch, for masking padded sequences.

        Returns:
            Tensor: Output tensor of shape (B, embedding_dim) containing
            pooled embeddings.

        """
        return self._pool(x, lengths)


class MeanPooling(PoolingLayer):
    """Pool variable-length sequences into fixed-size embeddings using mean pooling.

    Attributes:
        linear_projection (bool): Whether to apply a linear projection after pooling.
        linear (nn.Linear): Optional linear layer for projection after mean pooling.

    """

    def __init__(
        self,
        input_dim: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Initialise the mean pooling module.

        Args:
            input_dim (int | None): Dimensionality of input features, optional.
            embedding_dim (int | None): Dimensionality of the output embedding,
            optional. If different from input_dim, a linear projection will be applied
            after mean pooling.

        Raises:
            ValueError: If input_dim is specified but embedding_dim is not, or vice
            versa.

        """
        super().__init__()
        self.linear_projection = False
        if input_dim != embedding_dim:
            if input_dim is None or embedding_dim is None:
                msg = "Both input_dim and embedding_dim must be specified, or neither."
                raise ValueError(msg)
            self.linear_projection = True
            self.linear = nn.Linear(input_dim, embedding_dim)

    def _pool(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if x.is_nested:
            # Tensor.mean and Tensor.sum are not supported nested operations.
            x, lengths = convert_nested_to_padded(x, padding_value=0.0)
        elif lengths is None:
            lengths = x.size(1) * torch.ones(x.size(0), dtype=torch.long).to(x.device)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(
            0,
        ) < lengths.unsqueeze(1)
        x *= mask.unsqueeze(-1)  # Apply mask to zero out padded values
        x = x.sum(dim=1) / lengths.unsqueeze(1)  # Mean over time dimension
        if self.linear_projection:
            x = self.linear(x)
        return x
