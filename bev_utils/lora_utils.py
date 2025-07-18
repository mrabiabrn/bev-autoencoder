# Adapted from https://github.com/JamesQFreeman/LoRA-ViT


import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        layer_norm_q: nn.Module = None,
        layer_norm_v: nn.Module = None,
        layer_norm_k: nn.Module = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

        self.layernorm_q = layer_norm_q
        self.layernorm_v = layer_norm_v
        self.layernorm_k = layer_norm_k

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(self.layernorm_q(x)))
        new_v = self.linear_b_v(self.linear_a_v(self.layernorm_v(x)))
        new_k = self.linear_b_k(self.linear_a_k(self.layernorm_k(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        qkv[:, :, self.dim : 2 * self.dim] += new_k
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None, use_layer_norm=False, use_qkv=False):
        super(LoRA_ViT_timm, self).__init__()

        if r == 0:
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
            
        else:
            if lora_layer:
                self.lora_layer = lora_layer
            else:
                self.lora_layer = list(range(len(vit_model.blocks)))

            # dim = vit_model.head.in_features
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in vit_model.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(vit_model.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                w_a_linear_k = nn.Identity()
                w_b_linear_k = nn.Identity()
                if use_qkv:
                    w_a_linear_k = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_k = nn.Linear(r, self.dim, bias=False)
                layer_norm_q = nn.Identity()
                layer_norm_v = nn.Identity()
                layer_norm_k = nn.Identity()
                if use_layer_norm:
                    layer_norm_q = nn.LayerNorm(self.dim)
                    layer_norm_v = nn.LayerNorm(self.dim)
                    if use_qkv:
                        layer_norm_k = nn.LayerNorm(self.dim)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv_timm(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    w_a_linear_k,
                    w_b_linear_k,
                    layer_norm_q,
                    layer_norm_v,
                    layer_norm_k,
                )
            self.reset_parameters()
            self.lora_vit = vit_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit.forward_features(x)[:, 1:]
