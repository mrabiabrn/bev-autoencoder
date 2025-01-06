

# Code mainly from: https://github.com/facebookresearch/detr (Apache-2.0 license)
# TODO: Refactor & add docstring's

from typing import Union
import torch
from torch import nn

from .encoder import RVAEEncoder
from .decoder import RVAEDecoder


class RVAEModel(nn.Module):
    """Raster-Vector Autoencoder in of SLEDGE."""

    def __init__(self, config):
        """
        Initialize Raster-Vector Autoencoder.
        :param config: configuration dataclass of RVAE.
        """
        super().__init__(RVAEModel)

        self._config = config

        self._raster_encoder = RVAEEncoder(config)
        self._vector_decoder = RVAEDecoder(config)

    @staticmethod
    def _reparameterize(mu, log_var) -> torch.Tensor:
        """
        Reparameterization method for variational autoencoder's.
        :param latent: dataclass for mu, logvar tensors.
        :return: combined latent tensor.
        """
        assert mu.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features, encode_only: bool = False):
        # features : B, 4, 256, 256

        # encoding
        mu, log_var = self._raster_encoder(features)
        latent = self._reparameterize(mu, log_var)
        if encode_only:
            return latent

        # decoding
        #predictions["sledge_vector"] = self._vector_decoder(latent)
        return latent #predictions

    def get_encoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._raster_encoder

    def get_decoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._vector_decoder