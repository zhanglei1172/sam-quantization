import numpy as np
import torch
from torch.fx.node import Argument

from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.transformer import wrapped_atten

import time
from aitemplate.backend.target import Target
from aitemplate.compiler.base import Tensor, _TorchConstantTensorData
from aitemplate.compiler.ops import chunk
from aitemplate.frontend import nn
from aitemplate.utils.torch_utils import torch_dtype_to_string
from functools import partial
from fx2ait.converters.ait_converters import ConverterOutput
from fx2ait.converters.converter_registry import ait_converter
from fx2ait.example.benchmark_utils import benchmark_function, verify_accuracy
from fx2ait.utils import make_str_ait_friendly
from typing import Any, Dict, Tuple


@ait_converter(wrapped_atten)
def multi_head_attention_func(
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> ConverterOutput:
    # TODO fix arg/kwargs matching
    query = kwargs["q"] if "q" in kwargs else args[1]
    key = kwargs["k"] if "k" in kwargs else args[2]
    value = kwargs["v"] if "v" in kwargs else args[3]
    bsz, seq_len_q, dim = query.shape()
    _, seq_len, _ = key.shape()

    submod = args[0]
    dtype = torch_dtype_to_string(submod.out_proj.weight.dtype)
    assert (
        submod.embedding_dim % submod.num_heads == 0
    ), f"embed_dim {submod.embedding_dim} must be divisible by num_heads {submod.num_heads}"
    head_size = submod.embedding_dim // submod.num_heads
    if head_size % 4 != 0:
        raise ValueError(
            f"The head size {head_size} (ie. embed_dim ({submod.embedding_dim}) / num_heads ({submod.num_heads}) "
            " must be divisible by 4. Please fix the model or consider using the complete_video_view_all_page_types preset",
        )

    attn = nn.CrossAttention(
        dim=submod.embedding_dim,
        seq_len=seq_len_q.value(),
        seq_len_kv=seq_len.value(),
        num_heads=submod.num_heads,
        qkv_bias=True,
        has_residual=False,
        dim_inner=submod.out_proj.in_features,
    )

    # Bind constant tensor for MHA module
    # qkv_weight, qkv_bias = None, None
    for k, v in submod.named_parameters():
        ait_data = _TorchConstantTensorData(v.data.contiguous().cuda())
        if "q_proj" in k:
            if "weight" in k:
                q_w = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                q_w._bind_data(ait_data)
            elif "bias" in k:
                q_b = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                q_b._bind_data(ait_data)
        elif "k_proj" in k:
            if "weight" in k:
                k_w = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                k_w._bind_data(ait_data)
            elif "bias" in k:
                k_b = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                k_b._bind_data(ait_data)
        elif "v_proj" in k:
            if "weight" in k:
                v_w = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                v_w._bind_data(ait_data)
            elif "bias" in k:
                v_b = Tensor(
                    shape=v.shape,
                    dtype=dtype,
                    name=make_str_ait_friendly(f"{target}.{k}"),
                )
                v_b._bind_data(ait_data)
        elif "out_proj" in k:
            if "weight" in k:
                tensor = attn.proj.weight.tensor()
            elif "bias" in k:
                tensor = attn.proj.bias.tensor()
            tensor._attrs["name"] = make_str_ait_friendly(f"{target}.{k}")
            tensor._bind_data(ait_data)


    attn.proj_q.weight._tensor = q_w
    attn.proj_k.weight._tensor = k_w
    attn.proj_v.weight._tensor = v_w
    attn.proj_q.bias._tensor = q_b
    attn.proj_k.bias._tensor = k_b
    attn.proj_v.bias._tensor = v_b


    res = attn(query, key, value)
    # make output of MHA a list to match the output type of pytorch MHA
    return res


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model_ = sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda")
model = model_.mask_decoder
# model.forward = partial(model.forward)
del model_
torch.cuda.empty_cache()


def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result


def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
    )
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    import pickle

    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
    return result


@torch.no_grad()
def profile_warmup(input_image_batch, use_compile=False):
    global model
    with torch.autograd.profiler.record_function("compilation and warmup"):
        if use_compile:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
        for i in range(3):
            features_batch = model(input_image_batch)


@torch.no_grad()
def profile_run(input_image_batch):
    with torch.autograd.profiler.record_function("run"):
        features_batch = model(input_image_batch)


@torch.no_grad()
def benchmark(use_compile=False):
    global model
    if use_compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    for i in range(3):
        features_batch = model(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    tic = time.time()
    for i in range(100):
        features_batch = model(*args)
    end_event.record()
    torch.cuda.synchronize()
    toc = time.time()
    elapsed_time = start_event.elapsed_time(end_event)
    print(toc - tic)
    print(elapsed_time)

@torch.no_grad()
def benchmark_acc(use_compile=False):
    global model
    if use_compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    for i in range(3):
        features_batch = model(*args)
    verify_accuracy(model, args)

@torch.no_grad()
def profile_pipeline(input_image_batch, use_compile=False):
    profile_warmup(input_image_batch, use_compile=use_compile)
    profile_run(input_image_batch)

dtype = torch.float16
model.to(dtype)

if __name__ == "__main__":
    profiler_path = "./profiler"
    # input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    args = [
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
        torch.randn(1, 5, 256).cuda().to(dtype),
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
    ]
    # profiler_runner(profiler_path, profile_pipeline, input_image_batch, use_compile=True)
    benchmark_acc(use_compile=False)
