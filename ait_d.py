import numpy as np
import torch

from ait_segment_anything.build_sam import sam_model_registry as ait_sam_model_registry
from ait_segment_anything.modeling.transformer import get_debug_nodes
from segment_anything.build_sam import sam_model_registry

import time

# from segment_anything.modeling.transformer import wrapped_atten
from aitemplate.compiler import compile_model, transform
from aitemplate.compiler.base import Tensor
from aitemplate.testing import detect_target


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h_d"
model = ait_sam_model_registry[mt](checkpoint=None)  # .to("cuda")
model_ = sam_model_registry["vit_h"](
    checkpoint=model_type["vit_h"]
).mask_decoder  # .to("cuda")
# model.forward = partial(model.forward)
# del model_
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


def bench_func(func, args):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    tic = time.time()
    for i in range(1000):
        features_batch = func(*args)
    end_event.record()
    torch.cuda.synchronize()
    toc = time.time()
    elapsed_time = start_event.elapsed_time(end_event)
    print(toc - tic)
    print(elapsed_time)


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


@torch.no_grad()
def benchmark_acc_debug():
    global model, model_
    # input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    inputs_ait = [
        Tensor(x.shape, name=f"input_{i}", is_input=True) for i, x in enumerate(args)
    ]
    model.name_parameter_tensor()
    Y = model(*inputs_ait)
    Y = (Y[0], *get_debug_nodes())
    mark_output(Y)
    target = detect_target(use_fp16_acc=True)
    exe_module = compile_model(Y, target, "./tmp", "ait_sam_decoder")
    y_ait = [
        torch.empty(*[x.value() for x in y.shape()], device="cuda", dtype=torch.float16)
        for y in Y
    ]

    params_pt = model_.named_parameters()
    params_ait = {k: None for k in exe_module.get_constant_names()}
    for name, param in params_pt:
        ait_name = (
            name.replace(".", "_")
            .replace("q_proj", "proj_q")
            .replace("k_proj", "proj_k")
            .replace("v_proj", "proj_v")
            .replace("out_proj", "proj")
        )
        if ait_name not in params_ait:
            print(ait_name)
            continue
        if len(param.shape) == 4 and "weight" in name:
            params_ait[ait_name] = (
                param.permute((0, 2, 3, 1)).contiguous().to(dtype).cuda()
            )
        else:
            params_ait[f"{ait_name}"] = param.to(dtype).cuda()
    for k, v in params_ait.items():
        assert v is not None, k
        exe_module.set_constant_with_tensor(
            k,
            v,
        )
    kwargs = {f"input_{i}": args[i] for i in range(len(args))}
    exe_module.run_with_tensors(kwargs, y_ait)
    print(y_ait)
    # torch.save(y_ait, "y_ait.pth")
    model_.to("cuda")
    y_ref = model_(*(arg.to(torch.float32) for arg in args))[0].to(torch.float16)
    print(y_ref - y_ait[0])
    print(1)


@torch.no_grad()
def benchmark_speed():
    global model, model_
    # input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    inputs_ait = [
        Tensor(x.shape, name=f"input_{i}", is_input=True) for i, x in enumerate(args)
    ]
    model.name_parameter_tensor()
    Y = model(*inputs_ait)
    Y = (Y[0], *get_debug_nodes())
    mark_output(Y)
    target = detect_target(use_fp16_acc=True)
    exe_module = compile_model(Y, target, "./tmp", "ait_sam_decoder")
    y_ait = [
        torch.empty(*[x.value() for x in y.shape()], device="cuda", dtype=torch.float16)
        for y in Y
    ]

    params_pt = model_.named_parameters()
    params_ait = {k: None for k in exe_module.get_constant_names()}
    for name, param in params_pt:
        ait_name = (
            name.replace(".", "_")
            .replace("q_proj", "proj_q")
            .replace("k_proj", "proj_k")
            .replace("v_proj", "proj_v")
            .replace("out_proj", "proj")
        )
        if ait_name not in params_ait:
            print(ait_name)
            continue
        if len(param.shape) == 4 and "weight" in name:
            params_ait[ait_name] = (
                param.permute((0, 2, 3, 1)).contiguous().to(dtype).cuda()
            )
        else:
            params_ait[f"{ait_name}"] = param.to(dtype).cuda()
    for k, v in params_ait.items():
        assert v is not None, k
        exe_module.set_constant_with_tensor(
            k,
            v,
        )
    kwargs = {f"input_{i}": args[i] for i in range(len(args))}
    bench_func(exe_module.run_with_tensors, [kwargs, y_ait])


@torch.no_grad()
def profile_pipeline(input_image_batch, use_compile=False):
    profile_warmup(input_image_batch, use_compile=use_compile)
    profile_run(input_image_batch)


dtype = torch.float16
# model.to(dtype)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    profiler_path = "./profiler"
    # input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    args = [
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
        torch.randn(1, 5, 256).cuda().to(dtype),
        torch.randn(1, 256, 64, 64).cuda().to(dtype),
    ]
    # profiler_runner(profiler_path, profile_pipeline, input_image_batch, use_compile=True)
    benchmark_speed()
    # benchmark_acc_debug()
