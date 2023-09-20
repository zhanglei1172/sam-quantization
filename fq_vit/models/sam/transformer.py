# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from ..ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        quant=False,
        calibrate=False,
        cfg=None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    quant=quant,
                    calibrate=calibrate,
                    cfg=cfg,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, cfg=cfg
        )
        self.norm_final_attn = QIntLayerNorm(embedding_dim)
        self.qact1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact4 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.qact3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN,
            permute=False,
        )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = self.qact1(queries + point_embedding)
        k = self.qact2(keys + image_pe)
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.qact3(queries + attn_out)
        queries = self.qact4(
            self.norm_final_attn(queries, self.qact3.quantizer, self.qact4.quantizer)
        )

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        quant=False,
        calibrate=False,
        cfg=None,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        if not skip_first_layer_pe:
            self.qact_pos = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A,
                permute=False,
            )
            self.qact1 = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A,
                permute=False,
            )
        self.qact2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN,
        )
        self.qact3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact4 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact5 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact6 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN,
        )
        self.qact7 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact8 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact9 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN,
        )
        self.qact10 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact11 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact12 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact13 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A_LN,
            observer_str=cfg.OBSERVER_A_LN,
            quantizer_str=cfg.QUANTIZER_A_LN,
        )
        self.qact14 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.self_attn = Attention(
            embedding_dim, num_heads, quant=quant, calibrate=calibrate, cfg=cfg
        )
        self.norm1 = QIntLayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            quant=quant,
            calibrate=calibrate,
            cfg=cfg,
        )
        self.norm2 = QIntLayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation, quant, calibrate, cfg)
        self.norm3 = QIntLayerNorm(embedding_dim)

        self.norm4 = QIntLayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            quant=quant,
            calibrate=calibrate,
            cfg=cfg,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.qact2(self.self_attn(q=queries, k=queries, v=queries))
        else:
            q = self.qact1(queries + self.qact_pos(query_pe))
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = self.qact2(queries + attn_out)
        queries = self.qact3(
            self.norm1(queries, self.qact2.quantizer, self.qact3.quantizer)
        )

        # Cross attention block, tokens attending to image embedding
        q = self.qact4(queries + query_pe)
        k = self.qact5(keys + key_pe)
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.qact6(queries + attn_out)
        queries = self.qact7(
            self.norm2(queries, self.qact6.quantizer, self.qact7.quantizer)
        )

        # MLP block
        mlp_out = self.qact8(self.mlp(queries))
        queries = self.qact9(queries + mlp_out)
        queries = self.qact10(
            self.norm3(queries, self.qact9.quantizer, self.qact10.quantizer)
        )

        # Cross attention block, image embedding attending to tokens
        q = self.qact11(queries + query_pe)
        k = self.qact12(keys + key_pe)
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.qact13(keys + attn_out)
        keys = self.qact14(
            self.norm4(keys, self.qact13.quantizer, self.qact14.quantizer)
        )

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        quant=False,
        calibrate=False,
        cfg=None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = QLinear(
            embedding_dim,
            self.internal_dim,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W,
        )
        self.k_proj = QLinear(
            embedding_dim,
            self.internal_dim,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W,
        )
        self.v_proj = QLinear(
            embedding_dim,
            self.internal_dim,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W,
        )
        self.qact1_1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.qact1_2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.qact1_3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.qact2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.out_proj = QLinear(
            self.internal_dim,
            embedding_dim,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W,
        )
        self.qact3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.qact_attn1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
            permute=False,
        )
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S,
        )

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        q = self.qact1_1(q)
        k = self.k_proj(k)
        k = self.qact1_2(k)
        v = self.v_proj(v)
        v = self.qact1_3(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = self.qact_attn1(attn)

        # attn = torch.softmax(attn, dim=-1)
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.qact2(out)
        out = self.out_proj(out)
        out = self.qact3(out)

        return out
