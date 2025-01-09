import torch
import torchvision

import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from typing import Tuple, Union

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class RVAEEncoder(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, config):
        """
        Initialize encoder module.
        :param config: config of RVAE or VAE.
        """
        super().__init__()

        # TODO: support more backbones.
        backbone = getattr(torchvision.models, config.backbone)(
            replace_stride_with_dilation=[False, False, False],
            weights="DEFAULT",
            norm_layer=FrozenBatchNorm2d,
        )
        if config.num_input_channels != 3:
            backbone.conv1 = nn.Conv2d(config.num_input_channels, 64, 7, stride=2, padding=3, bias=False)

        self._backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        output_channels = 512 if config.backbone in ["resnet18", "resnet34"] else 2048

        # TODO: add params to config
        self._group_norm = nn.GroupNorm(num_groups=32, num_channels=output_channels, eps=1e-6, affine=True)
        self._latent = nn.Conv2d(output_channels, 2 * config.latent_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, raster: torch.Tensor) -> Tuple:
        # raster : B, 4, H, W

        cnn_features = self._backbone(raster)["0"]          # B, D, H//32, W//32  --> D = 2048
        normed_features = self._group_norm(cnn_features)    # B, D, H//32, W//32

        latent = self._latent(normed_features)              # B, 2L, H//32, W//32 --> L = latent_channel
        mu, log_var = torch.chunk(latent, 2, dim=1)         # B, L, H//32, W//32  (each) 

        return (mu, log_var)
