import torch
import time
from segment_anything.build_sam import sam_model_registry


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model_ = sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda")
model = model_.image_encoder
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
def profile_warmup(input_image_batch):
    with torch.autograd.profiler.record_function("compilation and warmup"):
        for i in range(3):
            features_batch = model(input_image_batch)


@torch.no_grad()
def profile_run(input_image_batch):
    with torch.autograd.profiler.record_function("run"):
        features_batch = model(input_image_batch)


@torch.no_grad()
def benchmark():
    input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)
    for i in range(3):
        features_batch = model(input_image_batch)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    tic = time.time()
    for i in range(100):
        features_batch = model(input_image_batch)
    end_event.record()
    torch.cuda.synchronize()
    toc = time.time()
    elapsed_time = start_event.elapsed_time(end_event)
    print(toc - tic)
    print(elapsed_time)


@torch.no_grad()
def profile_pipeline(input_image_batch):
    profile_warmup(input_image_batch)
    profile_run(input_image_batch)

dtype = torch.bfloat16
model.to(dtype)

if __name__ == "__main__":
    profiler_path = "./profiler"
    # input_image_batch = torch.randn(1, 3, 1024, 1024).cuda().to(dtype)

    # profiler_runner(profiler_path, profile_pipeline, input_image_batch)
    benchmark()
