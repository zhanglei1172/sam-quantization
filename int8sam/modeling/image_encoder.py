# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
from functools import partial
from triton_int.functional.quantization import quantize_per_tensor_absmax
from triton_int.kernels.utils import bmm_autotune
from triton_int.nn.bmm import BMM_S8T_S8N_F32T, BMM_S8T_S8T_S8T
from triton_int.nn.fused import LayerNormQ
from triton_int.nn.linear import (
    DanymicW8A8BFP32OFP32Linear,
    W8A8B8O8Linear,
    W8A8B8O8LinearReLU,
    W8A8BFP32OFP32Linear,
)
from typing import Optional, Tuple, Type

from .common import INTMLPBlock, LayerNorm2d

# from segment_anything.flash_4 import _attention_rel_h_rel_w


@bmm_autotune()
@triton.jit
def kernel_bmm_s8t_s8n_f32t(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    scale,
    # Matrix dimensions
    M,
    N,
    K,
    X,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_batch_a,
    stride_batch_b,
    stride_batch_c,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + pid_batch * stride_batch_a
    )
    b_ptrs = (
        b_ptr
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + (pid_batch % X) * stride_batch_b
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0
        )
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here

    c = accumulator.to(tl.float32) * scale
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
        + pid_batch * stride_batch_c
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c.to(c_ptr.dtype.element_ty), mask=c_mask)


def custom_bmm_s8t_s8n_f32t(a, b, scale: float, out=None, dtype=torch.float32):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # Check constraints.
    tmp_shape = a.shape[:-1]
    # a = a.view(-1, a.shape[-1])
    # b = b.view(-1, b.shape[-1])
    assert len(a.shape) == 4 and len(b.shape) == 3
    assert a.shape[1] == b.shape[0]
    assert a.shape[3] == b.shape[2], "Incompatible dimensions"
    B, X, M, K = a.shape
    X, N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.zeros((B, X, M, N), device=a.device, dtype=dtype)
    else:
        c = out.fill_(0)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
        B * X,
    )
    kernel_bmm_s8t_s8n_f32t[grid](
        a,
        b,
        c,
        scale,
        M,
        N,
        K,
        X,
        a.stride(1),
        b.stride(0),
        c.stride(1),
        a.stride(2),
        a.stride(3),
        b.stride(1),
        b.stride(2),
        c.stride(2),
        c.stride(3),
    )
    return c.view(*tmp_shape, N)


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

    @staticmethod
    @torch.no_grad()
    def from_float(module, layer_scales):
        int8_module = ImageEncoderViT(
            module.img_size,
            module.patch_embed.proj.kernel_size[0],
            module.patch_embed.proj.in_channels,
            module.patch_embed.proj.out_channels,
            len(module.blocks),
            module.blocks[0].attn.num_heads,
            module.blocks[0].mlp.lin1.out_features
            / module.blocks[0].norm1.weight.numel(),
            module.neck[0].out_channels,
            module.blocks[0].attn.qkv.bias is not None,
            module.blocks[0].norm1.__class__,
            module.blocks[0].mlp.act.__class__,
            module.pos_embed is not None,
            module.blocks[0].attn.use_rel_pos,
            True,
            module.blocks[0].window_size,
            [7, 15, 23, 31],
        )
        for i, blk in enumerate(module.blocks):
            int8_module.blocks[i] = Block.from_float(blk, layer_scales[i])
        int8_module.pos_embed = module.pos_embed
        int8_module.patch_embed = module.patch_embed
        int8_module.neck = module.neck

        return int8_module


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        ln_keywargs = {}
        if isinstance(norm_layer, partial):
            ln_keywargs = norm_layer.keywords
            norm_layer = norm_layer.func
        assert norm_layer == nn.LayerNorm, "Only support LayerNorm."
        self.norm1 = LayerNormQ(dim, **ln_keywargs)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = LayerNormQ(dim, **ln_keywargs)
        self.mlp = INTMLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    @staticmethod
    @torch.no_grad()
    def from_float(module, layer_scales):
        dim = module.norm1.weight.numel()
        int8_module = Block(
            dim,
            module.attn.num_heads,
            module.mlp.lin1.out_features // dim,
            module.attn.qkv.bias is not None,
            module.norm1.__class__,
            module.mlp.act.__class__,
            module.attn.use_rel_pos,
            True,
            module.window_size,
            (
                (module.attn.rel_pos_h.shape[0] + 1) // 2,
                (module.attn.rel_pos_w.shape[0] + 1) // 2,
            )
            if module.attn.use_rel_pos
            else None,
        )
        int8_module.norm1 = LayerNormQ.from_float(
            module.norm1, layer_scales["qkv_input_scale"]
        )
        int8_module.norm2 = LayerNormQ.from_float(
            module.norm2, layer_scales["fc1_input_scale"]
        )
        int8_module.attn = Attention.from_float(
            module.attn,
            layer_scales["qkv_input_scale"],
            layer_scales["qkv_output_scale"],
            layer_scales["out_input_scale"],
        )
        int8_module.mlp = INTMLPBlock.from_float(
            module.mlp, layer_scales["fc1_input_scale"], layer_scales["fc2_input_scale"]
        )

        return int8_module


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = W8A8B8O8Linear(dim, dim * 3)
        self.proj = W8A8BFP32OFP32Linear(dim, dim)
        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8T_S8T(1.0)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim, dtype=torch.int8),
                requires_grad=False,
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim, dtype=torch.int8),
                requires_grad=False,
            )
            self.rel_pos_h_bmm = BMM_S8T_S8N_F32T(1.0)
            self.rel_pos_w_bmm = BMM_S8T_S8N_F32T(1.0)
            self.register_buffer("rel_h_scale", torch.tensor(1.0))
            self.register_buffer("rel_w_scale", torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.qk_bmm(q, k)

        if self.use_rel_pos:
            attn = self.add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)

        x = (
            self.pv_bmm(attn.mul_(127).round_().to(torch.int8), v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x

    @staticmethod
    @torch.no_grad()
    def from_float(
        module,
        input_scale: float,
        qkv_out_scale: float,
        out_input_scale: float,
    ):
        int8_module = Attention(
            module.qkv.in_features,
            module.num_heads,
            module.qkv.bias is not None,
            module.use_rel_pos,
            True,
            ((module.rel_pos_h.shape[0] + 1) // 2, (module.rel_pos_w.shape[0] + 1) // 2)
            if module.use_rel_pos
            else None,
        )
        if module.use_rel_pos:
            int8_h, scale_h = quantize_per_tensor_absmax(module.rel_pos_h.data)
            int8_w, scale_w = quantize_per_tensor_absmax(module.rel_pos_w.data)
            int8_module.rel_pos_h.data.copy_(int8_h)
            int8_module.rel_pos_w.data.copy_(int8_w)
            int8_module.rel_h_scale = qkv_out_scale * scale_h
            int8_module.rel_w_scale = qkv_out_scale * scale_w
        int8_module.qkv = W8A8B8O8Linear.from_float(
            module.qkv, input_scale, qkv_out_scale
        )
        int8_module.proj = W8A8BFP32OFP32Linear.from_float(module.proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            qkv_out_scale * module.scale, qkv_out_scale
        )
        int8_module.pv_bmm = BMM_S8T_S8T_S8T.from_scale(
            1.0 / 127, qkv_out_scale, out_input_scale
        )

        return int8_module

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        # rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        # or not use torch.einsum:
        # rel_h = torch.matmul(r_q, Rh.transpose(1, 2))
        rel_h = custom_bmm_s8t_s8n_f32t(r_q, Rh, self.rel_h_scale.item())
        # rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        # rel_w = torch.matmul(r_q, Rw.transpose(1, 2))
        rel_w = custom_bmm_s8t_s8n_f32t(r_q, Rw, self.rel_w_scale.item())

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
        k_size / q_size, 1.0
    )
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
        q_size / k_size, 1.0
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
