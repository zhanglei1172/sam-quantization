# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from triton_int.nn.fused import W8A8BFP32OFP32_GeLu_Q
from triton_int.nn.linear import W8A8BFP32OFP32Linear
from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class INTMLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        assert act == nn.GELU
        self.lin1 = W8A8BFP32OFP32_GeLu_Q(embedding_dim, mlp_dim)
        self.lin2 = W8A8BFP32OFP32Linear(mlp_dim, embedding_dim)
        # self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))

    @staticmethod
    @torch.no_grad()
    def from_float(
        module,
        input_scale: float,
        lin2_input_scale: float,
    ):
        int8_module = MLPBlock(
            module.lin1.in_features, module.lin2.in_features, module.act.__class__
        )
        int8_module.lin1 = W8A8BFP32OFP32_GeLu_Q.from_float(
            module.lin1, input_scale, lin2_input_scale
        )
        int8_module.lin2 = W8A8BFP32OFP32Linear.from_float(
            module.lin2, lin2_input_scale
        )

        return int8_module


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-3, keepdim=True)
        s = (x - u).pow(2).mean(-3, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
