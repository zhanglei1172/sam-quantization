"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

This version was modified to fuse an addition of two attention masks into one
attn_bias = (rel_h_ + rel_w_).view(q_.size(0), q_.size(1), rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4))

We use attn_mask and attn_bias interchangeably.

This modification was designed by Christian Puhrsch and Daniel Haziza

"""

import torch

import triton
import triton.language as tl

import os
import pathlib

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

def _autotune(configs, function):
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
            t0 = benchmark.Timer(
                stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
            )
        except:
            return None
        return t0.blocked_autorange().mean * 1e6

    best = None
    best_config = None
    for config in configs:
        BLOCK_M, BLOCK_N, num_warps, num_stages = config
        t_config = benchmark_torch_function_in_microseconds(
            function, BLOCK_M, BLOCK_N, num_warps, num_stages)
        if t_config is not None:
            if best is not None:
                if t_config < best:
                    best = t_config
                    best_config = config
            else:
                best = t_config
                best_config = config
        print(str(config), " :", str(t_config))
    return best, best_config


def _attention_rel_h_rel_w_kernel_aligned_device(inp, rel_h, rel_w, sm_scale, head_num, hidden_dim, o,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 num_warps,
                                                 num_stages):
    assert (inp.dtype == torch.bfloat16 or inp.dtype == torch.float16)
    assert o.dtype == inp.dtype
    assert rel_h.dtype == inp.dtype
    assert rel_w.dtype == inp.dtype
    # assert rel_h_w.size(-1) == 2 * BLOCK_N
    batch, h, w, qkvhd = inp.shape

    BLOCK_HEADDIM = max(triton.next_power_of_2(hidden_dim), 16)
    # assert Lk in {16, 32, 64, 128}
    grid = (triton.cdiv(h * w, BLOCK_M), head_num * batch, 1)
    qkv_offset = qkvhd // 3
    _fwd_kernel1[grid](
        inp,
        rel_h,
        rel_w,
        h * w,
        batch,
        head_num,
        hidden_dim,
        sm_scale,
        o,
        rel_h.shape[-1],
        rel_h.stride(0),
        qkv_offset,
        inp.stride(2),
        inp.stride(0),
        hidden_dim,  # head_num stride
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_HEADDIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o



def _load_best_configs():
    device_name = torch.cuda.get_device_name()
    if not device_name.startswith('NVIDIA A100'):
        print("Warning: Custom flash attention kernels were written specifically for A100.")
    import importlib
    from importlib_resources import files
    saved_configs = files("segment_anything")
    saved_configs = saved_configs / "configs" / "flash_4_configs_a100.p"
    if not device_name.startswith('NVIDIA A100'):
        cwd = pathlib.Path.cwd()
        saved_configs = cwd / "flash_4_configs.p"
        print(f"We will try to read previously created kernel configurations from {saved_configs}.")
        print("You can disable this kernel by setting segment_anything_USE_FLASH_4=0")
        return {}
    if saved_configs.is_file():
        import pickle
        with open(saved_configs, 'rb') as f:
            print(f"Loading best configs from file {saved_configs}")
            return pickle.load(f)


def _save_best_configs(best_configs):
    import importlib
    from importlib_resources import files
    saved_configs = files("segment_anything")
    saved_configs = saved_configs / "configs" / "flash_4_configs_a100.p"
    device_name = torch.cuda.get_device_name()
    if not device_name.startswith('NVIDIA A100'):
        saved_configs = pathlib.Path.cwd() / "flash_4_configs.p"
        print("Warning: Custom flash attention kernels were written specifically for A100.")
        print(f"Storing configs for {device_name} locally under {saved_configs}")
    with open(saved_configs, 'wb') as f:
        import pickle
        print(f"Saving best configs to file {saved_configs}")
        pickle.dump(best_configs, f)


def _create_best_configs_key(inp, rel_h, rel_w, o):
    key = (inp.size(),   rel_h.size(), rel_w.size(),   o.size(),
           inp.stride(), rel_h.stride(), rel_w.stride(), o.stride())
    return key


BEST_CONFIGS = None

lib = torch.library.Library("customflash", "FRAGMENT")
lib.define("custom_flash_aligned(Tensor inp, Tensor rel_h, Tensor rel_w, float sm_scale, int head_num, int hidden_dim) -> Tensor")


# All that's needed for torch.compile support
@torch.library.impl(lib, "custom_flash_aligned", "Meta")
def _attention_rel_h_rel_w_kernel_aligned_meta(inp, rel_h, rel_w, sm_scale, head_num, hidden_dim):
    return inp[..., :head_num*hidden_dim].contiguous()


@torch.library.impl(lib, "custom_flash_aligned", "CUDA")
def _attention_rel_h_rel_w_kernel_aligned(inp, rel_h, rel_w, sm_scale, head_num, hidden_dim):
    # This is likely not needed, but without it the kernel
    # is guaranteed to fail. If the inputs are already contiguous
    # these are cheap checks via is_contiguous and do nothing.
    inp = inp.contiguous()
    b, h, w, _ = inp.shape
    o = torch.empty(b, h, w, head_num*hidden_dim, dtype=inp.dtype, device=inp.device, memory_format=torch.contiguous_format)

    global BEST_CONFIGS
    if BEST_CONFIGS is None:
        BEST_CONFIGS = _load_best_configs()
    key = _create_best_configs_key(inp, rel_h, rel_w, o)
    if key not in BEST_CONFIGS:
        print("key ", key, " not found. Running autotune. This might take a while.")
        import functools
        import itertools
        configs = []
        for (BLOCK_M, BLOCK_N, num_warps) in itertools.product([64, 128], [64, 128], [1, 2, 4, 8]):
            for num_stages in range(1, num_warps + 1):
                configs.append((BLOCK_M, BLOCK_N, num_warps, num_stages))
        print("all configs len: ", len(configs))
        best, best_config = _autotune(configs, functools.partial(_attention_rel_h_rel_w_kernel_aligned_device,
                                                                 inp, rel_h, rel_w, sm_scale, head_num, hidden_dim, o))
        BEST_CONFIGS[key] = best_config
        print("Found best_config ", best_config,
              " with time ", best, " for key ", key)
        _save_best_configs(BEST_CONFIGS)
    best_config = BEST_CONFIGS[key]
    if best_config is None:
        return torch.tensor([])

    _attention_rel_h_rel_w_kernel_aligned_device(inp,
                                                 rel_h,
                                                 rel_w,
                                                 sm_scale,
                                                 head_num,
                                                 hidden_dim,
                                                 o,
                                                 best_config[0],
                                                 best_config[1],
                                                 best_config[2],
                                                 best_config[3])

    return o


USE_CUSTOM_KERNEL = bool(int(os.environ.get('segment_anything_USE_FLASH_4', 1)))


def _attention_rel_h_rel_w(inp, rel_h, rel_w, head_num, hidden_dim, sm_scale):
    """
    Writing this as a composite allows torch.compile to fuse
    the needed padding into previous operations and memory
    allocations.
    """
    


    return torch.ops.customflash.custom_flash_aligned(
        inp, rel_h, rel_w, sm_scale, head_num, hidden_dim)


    
