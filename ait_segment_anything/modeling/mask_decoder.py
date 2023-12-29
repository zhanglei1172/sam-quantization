# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.compiler.ops.common.epilogue import FuncEnum
from aitemplate.compiler.ops.tensor.dynamic_slice import MAX_INT32

from typing import List, Tuple, Type
from ait_segment_anything.modeling.transformer import put_debug_nodes, debug_node
# from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation=FuncEnum.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
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

        self.iou_token = nn.Embedding((1, transformer_dim), dtype="float16")
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(
            (self.num_mask_tokens, transformer_dim), dtype="float16"
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2dBias(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            # LayerNorm2d(transformer_dim // 4),
            nn.LayerNorm(transformer_dim // 4),
            ActivationModule(activation),
            nn.ConvTranspose2dBias(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            ActivationModule(activation),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        # multimask_output: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (Tensor): the embeddings from the image encoder
          image_pe (Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          Tensor: batched predicted masks
          Tensor: batched predictions of mask quality
        """

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        # mask_slice = slice(1, None)
        # else:
        # mask_slice = slice(0, 1)
        masks = ops.dynamic_slice()(
            masks, start_indices=[0, 0, 0, 0], end_indices=[None, 1, None, None]
        )
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]
        # iou_pred = ops.dynamic_slice()(
        #     iou_pred, start_indices=[0, 0], end_indices=[None, 1]
        # )

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = ops.concatenate()(
            [self.iou_token.tensor(), self.mask_tokens.tensor()], dim=0
        )

        output_tokens = ops.expand()(
            ops.unsqueeze(0)(output_tokens),
            shape=[sparse_prompt_embeddings.shape()[0], -1, -1],
        )
        tokens = ops.concatenate()([output_tokens, sparse_prompt_embeddings], dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = ops.reshape()(
            ops.expand()(
                ops.unsqueeze(1)(image_embeddings), [-1, tokens.shape()[0], -1, -1, -1]
            ),
            [
                -1,
                image_embeddings.shape()[1],
                image_embeddings.shape()[2],
                image_embeddings.shape()[2],
            ],
        )
        src = src + dense_prompt_embeddings
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        pos_src = ops.reshape()(
            ops.expand()(
                ops.unsqueeze(1)(image_pe), [-1, tokens.shape()[0], -1, -1, -1]
            ),
            [
                -1,
                image_pe.shape()[1],
                image_pe.shape()[2],
                image_pe.shape()[2],
            ],
        )
        b, c, h, w = src.shape()

        # Run the transformer
        src_ = src
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = ops.reshape()(
            ops.dynamic_slice()(
                hs, start_indices=[0, 0, 0], end_indices=[None, 1, None]
            ),
            [b, -1],
        )
        mask_tokens_out = ops.dynamic_slice()(
            hs,
            start_indices=[0, 1, 0],
            end_indices=[None, 1 + self.num_mask_tokens, None],
        )

        # Upscale mask embeddings and predict masks using the mask tokens
        # src = ops.reshape()(ops.transpose()(src, dim0=1, dim1=2), shape=[b, c, h, w])
        src = ops.reshape()(src, shape=[b, h, w, c])

        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                (
                    self.output_hypernetworks_mlps[i](
                        ops.dynamic_slice()(
                            mask_tokens_out,
                            start_indices=[0, 0, 0],
                            end_indices=[None, 1, None],
                        )
                    )
                )
            )
        hyper_in = ops.concatenate()(hyper_in_list, dim=-2)
        b, h, w, c = upscaled_embedding.shape()
        masks = ops.reshape()(
            ops.bmm_rcr()(hyper_in, ops.reshape()(upscaled_embedding, [b, h * w, c])),
            [b, -1, h, w],
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
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                ops.elementwise(FuncEnum.RELU)(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        if self.sigmoid_output:
            x = ops.elementwise(FuncEnum.SIGMOID)(x)
        return x


class ActivationModule(nn.Module):
    def __init__(self, activation) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return ops.elementwise(self.activation)(x)
