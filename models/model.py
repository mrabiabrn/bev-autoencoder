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
        super().__init__()

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

    def forward(self, features, encode_only=False):
        # features : B, 4, H, W.  H = W = 256

        predictions = {}                                   

        # encoding
        mu, log_var = self._raster_encoder(features)        # B, L, 8, 8
        latent = self._reparameterize(mu, log_var)          # B, L, 8, 8

        if encode_only:
            return latent
        predictions["latent"] = {"mu": mu, "log_var": log_var, "latent": latent}

        # decoding
        predictions["vector"] = self._vector_decoder(latent)

        return predictions 