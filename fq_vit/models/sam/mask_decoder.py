# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from fq_vit.models.ptq.layers import QAct, QConvTranspose2d, QLinear
from typing import List, Tuple, Type

from .common import QIntLayerNorm2D


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        quant=False,
        calibrate=False,
        cfg=None
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        # self.qact_embed = QAct(
        #     quant=quant,
        #     calibrate=calibrate,
        #     bit_type=cfg.BIT_TYPE_A,
        #     calibration_mode=cfg.CALIBRATION_MODE_A,
        #     observer_str=cfg.OBSERVER_A,
        #     quantizer_str=cfg.QUANTIZER_A,
        # )
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.qact_embed1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.qact_embed2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.pos_src_qact = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.pos_src_qact = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )
        self.output_upscaling = nn.ModuleList(
            [
                QConvTranspose2d(
                    transformer_dim,
                    transformer_dim // 4,
                    kernel_size=2,
                    stride=2,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W,
                ),
                QIntLayerNorm2D(transformer_dim // 4),
                activation(),
                QConvTranspose2d(
                    transformer_dim // 4,
                    transformer_dim // 8,
                    kernel_size=2,
                    stride=2,
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_W,
                    calibration_mode=cfg.CALIBRATION_MODE_W,
                    observer_str=cfg.OBSERVER_W,
                    quantizer_str=cfg.QUANTIZER_W,
                ),
                activation(),
            ]
        )
        self.qacts = nn.ModuleList(
            [
                QAct(
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A,
                    observer_str=cfg.OBSERVER_A,
                    quantizer_str=cfg.QUANTIZER_A,
                )
                if _ != 0
                else QAct(
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                    observer_str=cfg.OBSERVER_A_LN,
                    quantizer_str=cfg.QUANTIZER_A_LN,
                )
                for _ in range(5)
            ]
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(
                    transformer_dim,
                    transformer_dim,
                    transformer_dim // 8,
                    3,
                    quant=quant,
                    calibrate=calibrate,
                    cfg=cfg,
                )
                for i in range(self.num_mask_tokens)
            ]
        )
        self.qact1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            quant=quant,
            calibrate=calibrate,
            cfg=cfg,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = self.qact_embed1(
            torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        )

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings  # modified
        src = self.qact_embed2(src + dense_prompt_embeddings)
        pos_src = self.pos_src_qact(image_pe)  # modified
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        x = src
        for i, mod in enumerate(self.output_upscaling):
            if isinstance(mod, QIntLayerNorm2D):
                x = self.qacts[i](mod(x, self.qacts[i - 1].quantizer))
            else:
                x = self.qacts[i](mod(x))

        # upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding = x
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = self.qact1(
            (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        quant=False,
        calibrate=False,
        cfg=None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            QLinear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.qacts1 = nn.ModuleList(
            [
                QAct(
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A,
                    observer_str=cfg.OBSERVER_A,
                    quantizer_str=cfg.QUANTIZER_A,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.qacts2 = nn.ModuleList(
            [
                QAct(
                    quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A,
                    observer_str=cfg.OBSERVER_A,
                    quantizer_str=cfg.QUANTIZER_A,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.qact3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A,
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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = self.qacts2[i](F.relu(self.qacts1[i](layer(x))))
            else:
                x = self.qact3(layer(x))
        if self.sigmoid_output:
            x = self.qact4(F.sigmoid(x))
        return x
