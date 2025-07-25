from __future__ import annotations

import json
from pathlib import PosixPath
from typing import Any, Dict, Optional, Tuple, Union
import dataclasses
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.position_encoding import build_position_encoding
from .utils.transformer import Transformer

from .indices import LineIndex, AgentIndex
from enum import Enum


class RVAEDecoder(nn.Module):
    """Decoder module of Raster-Vector-Autoencoder."""

    def __init__(self, config):
        """
        Initialize decoder module.
        :param config: config dataclass of RVAE
        """
        super().__init__()

        self._config = config
        self._num_queries_list = config.num_queries_list

        self._transformer = Transformer(
            d_model=config.d_model,
            nhead=config.num_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation=config.activation,
            normalize_before=config.normalize_before,
            return_intermediate_dec=False,
        )
        self._line_head = LineHead(
            d_input=config.d_model,
            d_ffn=config.head_d_ffn,
            num_layers=config.head_num_layers,
            num_line_poses=config.num_line_poses,
            frame=config.frame,
        )
        self._line_head_div = LineHead(
            d_input=config.d_model,
            d_ffn=config.head_d_ffn,
            num_layers=config.head_num_layers,
            num_line_poses=10, #config.num_line_poses,
            frame=config.frame,
        )
        self._vehicle_head = BoundingBoxHead(
            d_input=config.d_model,
            d_ffn=config.head_d_ffn,
            num_layers=config.head_num_layers,
            frame=config.frame,
            enum=AgentIndex,
            max_velocity=config.vehicle_max_velocity,
            multi_class=config.multiclass,
        )
        # self._pedestrian_head = BoundingBoxHead(
        #     d_input=config.d_model,
        #     d_ffn=config.head_d_ffn,
        #     num_layers=config.head_num_layers,
        #     frame=config.frame,
        #     enum=AgentIndex,
        #     max_velocity=config.pedestrian_max_velocity,
        # )
        # self._static_object_head = BoundingBoxHead(
        #     d_input=config.d_model,
        #     d_ffn=config.head_d_ffn,
        #     num_layers=config.head_num_layers,
        #     frame=config.frame,
        #     enum=StaticObjectIndex,
        # )
        # self._green_line_head = LineHead(
        #     d_input=config.d_model,
        #     d_ffn=config.head_d_ffn,
        #     num_layers=config.head_num_layers,
        #     num_line_poses=config.num_line_poses,
        #     frame=config.frame,
        # )
        # self._red_line_head = LineHead(
        #     d_input=config.d_model,
        #     d_ffn=config.head_d_ffn,
        #     num_layers=config.head_num_layers,
        #     num_line_poses=config.num_line_poses,
        #     frame=config.frame,
        # )
        #self._ego_head = EgoHead(config.d_model)

        self._patch_projection = nn.Linear(config.d_patches, config.d_model)
        self._query_embedding = nn.Embedding(sum(self._num_queries_list), config.d_model)
        self._position_encoding = build_position_encoding(config)

        if config.split_latent:
            self._type_encoding = nn.Embedding(2, config.d_model)


    def forward(self, latent: torch.Tensor) -> Tuple:
        """
        Forward pass of decoder module.
        :param latent: tensor of latent variable (after reparameterization)
        :return: sledge vector dataclass
        """
        b, device = latent.shape[0], latent.device   # (b, d_model, h, w)

        if self._config.split_latent:
            static_latent, dynamic_latent = torch.chunk(latent, 2, dim=1)

            static_patches, dynamic_patches = (
                patchify(static_latent, self._config.patch_size),
                patchify(dynamic_latent, self._config.patch_size),
            )

            num_patches = self._config.num_patches
            decoder_mask = create_mask(
                num_patches,
                num_patches,
                sum(self._num_queries_list[:2]),  # TODO: lane queries
                sum(self._num_queries_list[2:]),
                device,
            )

            type_embed = self._type_encoding.weight[None, ...].repeat(b, 1, 1)   # (b, 2, d_model)
            type_embed = type_embed.repeat_interleave(num_patches, 1)            # (b, 2*p*p, d_model)
            type_embed = type_embed.permute(1, 0, 2)                             # (2*p*p, b, d_model)

            pos_embed = self._position_encoding(static_patches)     # (b, d_model, p, p)
            pos_embed = pos_embed.flatten(-2).permute(2, 0, 1)      # (p*p, b, d_model)
            pos_embed = pos_embed.repeat(2, 1, 1)                   # (2*p*p, b, d_model)
            pos_embed += type_embed

            static_patches = static_patches.flatten(-2).permute(2, 0, 1)    # (p*p, b, d_patches)
            dynamic_patches = dynamic_patches.flatten(-2).permute(2, 0, 1)  # (p*p, b, d_patches)
            patches = torch.cat([static_patches, dynamic_patches], dim=0)   # (2*p*p, b, d_patches)
            projected_patches = self._patch_projection(patches)             # (2*p*p, b, d_model)

        else:
            decoder_mask = None
            patches = patchify(static_latent, self._config.patch_size)
            patches = patches.flatten(-2).permute(2, 0, 1)          # (p*p, b, d_patches)
            projected_patches = self._patch_projection(patches)     # (p*p, b, d_model)

            pos_embed = self._position_encoding(static_patches)     # (b, d_model, p, p)
            pos_embed = pos_embed.flatten(-2).permute(2, 0, 1)      # (p*p, b, d_model)

        query_embed = self._query_embedding.weight[:, None].repeat(1, b, 1)  # (num_queries, b, d_model)

        # print("decoder mask shape:", decoder_mask.shape if decoder_mask is not None else "None")
        # print("projected patches shape:", projected_patches.shape)
        # print("query embed shape:", query_embed.shape)
        # print("pos embed shape:", pos_embed.shape)
        # print("num queries list:", self._num_queries_list)
        hs = self._transformer(
            src=projected_patches,
            query_embed=query_embed,                
            pos_embed=pos_embed,
            memory_mask=decoder_mask,
        )[0].permute(1, 0, 2)

        hs_line, hs_div, hs_vehicle = hs.split(           # hs_pedestrian, hs_static, hs_green, hs_red, hs_ego 
            self._num_queries_list, dim=1
        )
        # hs_line --> (b, num_line_queries, d_model)
        line_element = self._line_head(hs_line)
        vehicle_element = self._vehicle_head(hs_vehicle)
        div_element = self._line_head_div(hs_div)
        # pedestrian_element = self._pedestrian_head(hs_pedestrian)
        # static_object_element = self._static_object_head(hs_static)
        # green_line_element = self._green_line_head(hs_green)
        # red_line_element = self._red_line_head(hs_red)
        # ego_element = self._ego_head(hs_ego)

        return {
                    'LANES': 
                        {
                            'vector': line_element[0],
                            'mask': line_element[1],
                        },
                    'LANE_DIVIDERS':
                        {
                            'vector': div_element[0],
                            'mask': div_element[1],
                        },
                    'VEHICLES': 
                        {
                            'vector': vehicle_element[0],
                            'mask': vehicle_element[1],
                        },
                    'hs_vehicle': hs_vehicle,
                    }

    def decode(self, latent: torch.Tensor) -> Tuple:
        """Alias for .forward()"""
        return self.forward(latent)


class FFN(nn.Module):
    """Feed-forward network implementation."""

    def __init__(
        self,
        d_input: int,
        d_ffn: int,
        d_output: int,
        num_layers: int,
    ):
        """
        Initialize FNN.
        :param d_input: dimensionality of input
        :param d_ffn: dimensionality hidden layers
        :param d_output: dimensionality of output
        :param num_layers: number of hidden layers
        """
        super(FFN, self).__init__()

        self._num_layers = num_layers
        h = [d_ffn] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([d_input] + h, h + [d_output]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of FNN."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self._num_layers - 1 else layer(x)
        return x


class LineHead(nn.Module):
    """Decoder head for lines in RVAEDecoder (lines, traffic lights)."""

    def __init__(
        self,
        d_input: int,
        d_ffn: int,
        num_layers: int,
        num_line_poses: int,
        frame: Tuple[float, float],
    ):
        """
        Initialize line head.
        :param d_input: dimensionality of input.
        :param d_ffn: dimensionality of ffn hidden layers.
        :param num_layers: number of hidden layers in fnn.
        :param num_line_poses: number of output poses of lines.
        :param frame: output frame of predicted patch in meter.
        """
        super(LineHead, self).__init__()

        self._ffn_states = FFN(d_input, d_ffn, num_line_poses * len(LineIndex), num_layers)
        self._ffn_mask = nn.Linear(d_input, 1)

        self._num_line_poses = num_line_poses

        frame_transform = torch.tensor(frame, dtype=torch.float32, requires_grad=False)
        frame_transform = frame_transform[None, :] / 2
        self.register_buffer("_frame_transform", frame_transform)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of LineHead."""

        batch_size, num_queries = queries.shape[:2]
        states = (
            self._ffn_states(queries).tanh().reshape(batch_size, num_queries, self._num_line_poses, len(LineIndex))
            #* self._frame_transform
        )

        states = states
        mask = self._ffn_mask(queries).squeeze(dim=-1)

        return  (states, mask)



class BoundingBoxHead(nn.Module):
    """Decoder head for bounding boxes in RVAEDecoder (vehicles, pedestrian, static objects)."""

    def __init__(
        self,
        d_input: int,
        d_ffn: int,
        num_layers: int,
        frame: Tuple[float, float],
        enum: Enum,
        max_velocity: Optional[float] = None,
        multi_class: bool = False,
    ):
        """
        Initialize bounding box head.
        :param d_input: dimensionality of input.
        :param d_ffn: dimensionality of ffn hidden layers.
        :param num_layers: number of hidden layers in fnn.
        :param frame: output frame of predicted patch in meter.
        :param enum: integer enum class of bounding box states.
        :param max_velocity: max velocity of predicted bounding box, defaults to None.
        """
        super(BoundingBoxHead, self).__init__()

        #if enum == AgentIndex:
        #    assert max_velocity is not None

        self._ffn_states = FFN(d_input, d_ffn, len(enum), num_layers)
        num_classes = 9 if multi_class else 1
        self._ffn_mask = nn.Linear(d_input, num_classes)  # including null class

        self.multiclass = multi_class

        self._enum = enum
        self._max_velocity = max_velocity

        frame_transform = torch.tensor(frame, dtype=torch.float32, requires_grad=False)
        frame_transform = frame_transform[None, :] / 2
        self.register_buffer("_frame_transform", frame_transform)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of BoundingBoxHead."""

        states = self._ffn_states(queries)

        # states[..., :2] = states[..., :2].tanh() #* self._frame_transform
        # states[..., 2] = states[..., 2].tanh() #* np.pi
        # states[..., -2:] = states[..., -2:].tanh()
        states = states.tanh()

        if self._max_velocity is not None:
            states[..., -1] = states[..., -1].sigmoid() * self._max_velocity

        if self.multiclass:
            mask = self._ffn_mask(queries)  # B, Q, C
        else:
            mask = self._ffn_mask(queries).squeeze(dim=-1)
        return (states, mask)


class EgoHead(nn.Module):
    """Simple decoder head for ego vehicle attributes."""

    def __init__(
        self,
        d_input: int,
    ):
        """
        Initialize ego head.
        :param d_input: dimensionality of input.
        """

        super(EgoHead, self).__init__()

        # TODO: Maybe refactor this class
        self._ffn_states = nn.Linear(d_input, 1)

    def forward(self, ego_query) -> Tuple[torch.Tensor, torch.Tensor]:

        states = self._ffn_states(ego_query).squeeze()
        mask = torch.ones_like(states)  # dummy

        return (states, mask)


# TODO: as class method in decoder object?
def create_mask(
    num_static_patches: int,
    num_dynamic_patches: int,
    num_static_queries: int,
    num_dynamic_queries: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create lane-agent mask for transformer in RVAEDecoder.
    :param num_static_patches: number of patches (key-val) of static latent.
    :param num_dynamic_patches: number of patches (key-val) of dynamic latent.
    :param num_static_queries: number of queries for static elements.
    :param num_dynamic_queries: number of queries for dynamic elements.
    :param device: torch device where to init mask.
    :return: mask for RVAEDecoder
    """
    num_patches = num_static_patches + num_dynamic_patches
    num_queries = num_static_queries + num_dynamic_queries

    mask = torch.full((num_queries, num_patches), float("-inf"), device=device)

    mask[:num_static_queries, :num_static_patches] = 0
    mask[num_static_queries:, num_static_patches:] = 0

    return mask


# TODO: as class method in decoder object?
def patchify(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Create patches of latent tensor, similar to a vision transformer.
    :param latent: tensor of the latent variable.
    :param patch_size: size of patches.
    :return: patched latent tensor.
    """
    assert latent.ndim == 4, "Latent tensor should be 4-dimensional"
    batch, channels, height, width = latent.shape

    num_patches = height // patch_size
    patches = latent.unfold(dimension=2, size=patch_size, step=patch_size)
    patches = patches.unfold(dimension=3, size=patch_size, step=patch_size)
    patches = patches.permute(0, 1, 4, 5, 2, 3)
    patches = patches.reshape(batch, -1, num_patches, num_patches)

    return patches