# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act=FuncEnum.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = (act)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(ops.elementwise(self.act)(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(shape=[num_channels], dtype="float16")
        self.bias = nn.Parameter(shape=[num_channels], dtype="float16")
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = ops.reduce_mean(dim=-3, keepdim=True)(x)
        _tmp = x - u
        s = ops.reduce_mean(dim=-3, keepdim=True)((_tmp) * (_tmp))
        x = (_tmp) / ops.elementwise(FuncEnum.SQRT)(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        x = ops.reshape()(self.weight, shape=[-1, 1, 1]) * x + ops.reshape()(self.bias, shape=[-1, 1, 1])
        return x
