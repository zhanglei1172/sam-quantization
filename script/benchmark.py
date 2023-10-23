import torch

import json
import time
import tqdm
from ppq.utils.TensorRTUtil import Benchmark, Profiling


def load_torch_model():
    from segment_anything.build_sam import sam_model_registry

    model_type = {
        "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
        "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
        "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
    }
    mt = "vit_h"
    model = sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda")
    model = model.image_encoder
    model.eval()

    return model


def torch_infer(model, inputs):
    model(inputs)


@torch.no_grad()
def benchmark_torch(steps=1000):
    inputs = torch.randn(1, 3, 1024, 1024).to("cuda")
    model = load_torch_model()
    torch.cuda.synchronize()
    tick = time.time()
    for _ in tqdm.tqdm(range(steps), desc="Pytorch is running..."):
        torch_infer(model, inputs)
    torch.cuda.synchronize()
    tok = time.time()
    print(f"Time span: {tok - tick  : .4f} sec")
    return tok - tick


def benchmark_ort(engine_file, steps=1000):
    import numpy as np

    import onnxruntime
    inputs = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        engine_file, providers=["CUDAExecutionProvider"]
    )
    output_name = sess.get_outputs()[0].name

    tick = time.time()
    for _ in tqdm.tqdm(range(steps), desc="Pytorch is running..."):
        ort_outs = sess.run(
            output_names=[output_name],
            input_feed={"input.1": inputs},
        )
    tok = time.time()
    print(f"Time span: {tok - tick  : .4f} sec")
    return tok - tick


steps = 1000

# Benchmark(engine_file="Output/INT8.engine", steps=1000)
# Benchmark(engine_file="Output/FP32.engine", steps=1000)


# b_int8 = Benchmark(engine_file="trt_out/INT8.engine", steps=steps) / steps
# b_fp16 = Benchmark(engine_file="Output/FP16.engine", steps=steps) / steps
# b_fp32 = Benchmark(engine_file="Output/FP32.engine", steps=steps) / steps
# b_torch = benchmark_torch(steps=steps) / steps
b_ort_int8 = benchmark_ort(engine_file="Output/quantized.onnx", steps=steps) / steps

# print(f"INT8: {b_int8}s, FP16: {b_fp16}s, FP32: {b_fp32}s", f"Pytorch: {b_torch}s")


# rep = Profiling(engine_file="Output/INT8.engine", steps=100)
# with open("Output/INT8.profiling.json", "w") as f:
#     json.dump(rep, f)
# rep = Profiling(engine_file="Output/FP16.engine", steps=1000)
# with open("Output/FP16.profiling.json", "w") as f:
#     json.dump(rep, f)
