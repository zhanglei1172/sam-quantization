import torch
import torch.nn as nn

from segment_anything.modeling.image_encoder import Attention, get_rel_pos

import math
from typing import Optional, Tuple

# from .quant_linear import QuantLinear


def make_quant_attn(model):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    for name, m in model.named_modules():
        if not isinstance(m, Attention):
            continue

        qkv_proj = m.qkv

        attn = QuantAttention(
            qkv_proj,
            m.proj,
            m.num_heads,
            m.scale,
            m.use_rel_pos,
            m.rel_pos_h if m.use_rel_pos else None,
            m.rel_pos_w if m.use_rel_pos else None,
        )

        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name[len(parent_name) + 1 :]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            parent = model
            child_name = name

        # print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)


def add_decomposed_rel_pos(
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

    # B, _, dim = q.shape
    # r_q = q.reshape(B, q_h, q_w, dim)
    # rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    # or not use torch.einsum:
    rel_h = torch.matmul(q, Rh.transpose(1, 2))
    # rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_w = torch.matmul(q, Rw.transpose(1, 2))

    return rel_h, rel_w


class QuantAttention(nn.Module):
    """
    Modified version of LlamaAttention that fuses the q, k, v projections.
    """

    def __init__(
        self,
        qkv_proj,
        o_proj,
        num_heads,
        scale,
        use_rel_pos,
        rel_pos_h=None,
        rel_pos_w=None,
    ):
        super().__init__()
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.scale = scale
        self.rel_pos_h = rel_pos_h
        self.rel_pos_w = rel_pos_w
        self.use_rel_pos = use_rel_pos

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv_proj(x)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).reshape(3, B * self.num_heads, H, W, -1)[0]

        # attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            x = forward(
                qkv,
                rel_h,
                rel_w,
                self.num_heads,
                q.shape[-1],
                sm_scale=self.scale,
            ).to(x.dtype)
        else:
            raise NotImplementedError

        # upcast attention to fp32
        # attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(qkv.dtype)

        # Get output
        # x = (
        #     (attn @ v)
        #     .view(B, self.num_heads, H, W, -1)
        #     .permute(0, 2, 3, 1, 4)
        #     .reshape(B, H, W, -1)
        # )
        out = self.o_proj(x)

        return out


import torch

import math
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel1(
    INP,
    POS_EMB1,
    POS_EMB2,
    seq_len,
    batch_size,
    head_num,
    hidden_dim,
    sm_scale,
    Out,
    emb_len,
    stride_e,
    qkv_offset,
    seq_stride,
    batch_stride,
    head_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    head_batch_idx = tl.program_id(1)
    head_idx = head_batch_idx % head_num
    batch_idx = head_batch_idx // head_num
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hidden = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        INP
        + start_m * seq_stride
        + head_idx * head_stride
        + batch_idx * batch_stride
        + (offs_m[:, None] * seq_stride + offs_hidden[None, :])
    )
    k_ptrs = (
        INP
        + head_idx * head_stride
        + batch_idx * batch_stride
        + qkv_offset
        + (offs_n[:, None] * seq_stride + offs_hidden[None, :])
    )
    v_ptrs = (
        INP
        + head_idx * head_stride
        + batch_idx * batch_stride
        + qkv_offset * 2
        + (offs_n[:, None] * seq_stride + offs_hidden[None, :])
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # l_i = tl.where(offs_m < N_CTX, l_i, 1)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(
        q_ptrs,
        mask=(offs_hidden[None, :] < hidden_dim)
        & (offs_m[:, None] < seq_len - start_m),
        other=0.0,
    )
    q = (q * qk_scale).to(INP.dtype.element_ty)
    b_ptrs1_ = POS_EMB1 + head_batch_idx * stride_e
    b_ptrs2_ = POS_EMB2 + head_batch_idx * stride_e

    for start_n in range(0, seq_len, BLOCK_N):
        _offs_emb1 = (offs_n + start_n) // emb_len
        _offs_emb2 = (offs_n + start_n) % emb_len
        b1_ptrs = b_ptrs1_ + (
            (offs_m + start_m)[:, None] * emb_len + (_offs_emb1)[None, :]
        )
        b2_ptrs = b_ptrs2_ + (
            (offs_m + start_m)[:, None] * emb_len + (_offs_emb2)[None, :]
        )

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(
            k_ptrs + start_n * seq_stride,
            mask=(offs_hidden[None, :] < hidden_dim)
            & (offs_n[:, None] < seq_len - start_n),
            other=0,
        )
        v = tl.load(
            v_ptrs + start_n * seq_stride,
            mask=(offs_hidden[None, :] < hidden_dim)
            & (offs_n[:, None] < seq_len - start_n),
            other=0,
        )
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), allow_tf32=True)

        b1 = tl.load(
            b1_ptrs,
            mask=((offs_m[:, None] < seq_len - start_m))
            & ((offs_n[None, :] < seq_len - start_n)),
            other=0.0,
        ).to(
            tl.float32
        )  # [:, :, None]
        b2 = tl.load(
            b2_ptrs,
            mask=((offs_m[:, None] < seq_len - start_m))
            & ((offs_n[None, :] < seq_len - start_n)),
            other=0.0,
        ).to(
            tl.float32
        )  # [:, None, :]
        qk += b1 * 1.44269504
        qk += b2 * 1.44269504
        qk = tl.where(
            (seq_len - start_n > offs_n[None, :])
            & (seq_len - start_m > offs_m[:, None]),
            qk,
            float("-inf"),
        )
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(INP.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    # write back l and m
    acc = acc / l_i[:, None]
    # write back O
    O_ptrs = (
        Out
        + start_m * seq_stride // 3
        + head_idx * head_stride
        + batch_idx * batch_stride // 3
        + (offs_m[:, None] * (seq_stride // 3) + offs_hidden[None, :])
    )
    tl.store(
        O_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_hidden[None, :] < hidden_dim)
        & (offs_m[:, None] < seq_len - start_m),
    )


def forward(inp, pos_emb1, pos_emb2, head_num, hidden_dim, sm_scale):
    # only support for Ampere now
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError(
            "Flash attention currently only supported for compute capability >= 80"
        )
    BLOCK_M = 64
    BLOCK_N = 64
    # shape constraints

    BLOCK_HEADDIM = max(triton.next_power_of_2(hidden_dim), 16)
    # assert Lk in {16, 32, 64, 128}
    batch, h, w, qkvhd = inp.shape
    o = torch.full(
        [batch, h, w, qkvhd // 3],
        float("inf"),
        dtype=inp.dtype,
        device=inp.device,
    )
    grid = (triton.cdiv(h * w, BLOCK_M), head_num * batch, 1)
    qkv_offset = qkvhd // 3
    num_warps = 4 if hidden_dim <= 64 else 8
    _fwd_kernel1[grid](
        inp,
        pos_emb1,
        pos_emb2,
        h * w,
        batch,
        head_num,
        hidden_dim,
        sm_scale,
        o,
        pos_emb1.shape[-1],
        pos_emb1.stride(0),
        qkv_offset,
        inp.stride(2),
        inp.stride(0),
        hidden_dim,  # head_num stride
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_HEADDIM,
        num_warps=num_warps,
        num_stages=1,
    )

    return o


# @pytest.mark.parametrize("batch, seqlen_q, nheads, d,", [(1, 2, 1024, 64)])
# @pytest.mark.parametrize("causal", [True])
@torch.no_grad()
def test_op(batch, h, w, qkvhd, head_num=8, dtype=torch.float16):
    device = "cuda"
    torch.manual_seed(20)
    x = torch.empty((batch, h, w, qkvhd), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )
    sm_scale = 0.5
    hidden_dim = qkvhd // 3 // head_num
    pos_emb1 = torch.empty(
        (batch * head_num, h, w, h), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)
    pos_emb2 = torch.empty(
        (batch * head_num, h, w, h), dtype=dtype, device=device
    ).normal_(mean=0, std=0.5)
    tri_out = forward(
        x,
        pos_emb1,
        pos_emb2,
        head_num,
        hidden_dim,
        # pos_emb.to(device),
        sm_scale=sm_scale,
    ).to(dtype)
    # reference implementation
    qkv = (x).reshape(batch, h * w, 3, head_num, hidden_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, batch * head_num, h * w, hidden_dim).unbind(0)
    attn = (q * sm_scale) @ k.transpose(-2, -1)

    attn = nn.functional.softmax(
        (
            attn.view(batch * head_num, h, w, h, w)
            + pos_emb1.unsqueeze(-1)
            + pos_emb2.unsqueeze(-2)
        ).view(batch * head_num, h * w, h * w),
        dim=-1,
        dtype=torch.float32,
    ).to(qkv.dtype)
    ref_out = (
        (attn @ v)
        .view(batch, head_num, h, w, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch, h, w, qkvhd // 3)
    )

    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print("max diff: ", (ref_out - tri_out).abs().max().item())
    print(
        torch.nn.functional.cosine_similarity(ref_out.ravel(), tri_out.ravel(), dim=-1)
    )


if __name__ == "__main__":
    test_op(25, 14, 14, 3 * 16 * 80, head_num=16, dtype=torch.bfloat16)
    test_op(1, 64, 64, 3 * 16 * 80, head_num=16, dtype=torch.bfloat16)
