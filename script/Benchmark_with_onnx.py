# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import numpy as np
import torch
import torchvision
import torchvision.models

from segment_anything.build_sam import sam_model_registry

import os
import ppq.lib as PFL
import random
import tensorrt as trt
import time
import trt_infer
from ppq import *
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
from ppq.api import *
from ppq.api import ENABLE_CUDA_KERNEL, export_ppq_graph, load_onnx_graph
from ppq.core import convert_any_to_numpy
from ppq.quantization.optim import *
from tqdm import tqdm
from utils.datautils import get_loaders

# Nvidia Nsight Performance Profile
QUANT_PLATFROM = TargetPlatform.TRT_INT8
BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 32
DEVICE = "cuda"
CFG_VALID_RESULT = False

output_names = ["image_embedding"]

model_type = {
    # 'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
    # 'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(DEVICE)
model = model.image_encoder

calibration_dataloader = []
for trainloader, testloader in zip(
    *get_loaders("/data/SA-1B/sa_000000/", CALIBRATION_SIZE)
):
    calibration_dataloader.append(trainloader["pixel_values"])


def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """Run a tensorrt model with given samples"""
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    if CFG_VALID_RESULT:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(
                context.engine
            )
            for sample in tqdm(samples, desc="TensorRT is running..."):
                inputs[0].host = convert_any_to_numpy(sample)
                [output] = trt_infer.do_inference(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    batch_size=1,
                )[0]
                results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
    else:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(
                context.engine
            )
            inputs[0].host = convert_any_to_numpy(samples[0])
            for sample in tqdm(samples, desc="TensorRT is running..."):
                trt_infer.do_inference(
                    context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    batch_size=1,
                )
    return results


with torch.no_grad():
    torch.onnx.export(
        model,
        torch.rand(size=[1] + INPUT_SHAPE).to(DEVICE),
        "out/onnx.model",
        export_params=True,
        verbose=False,
        opset_version=11,
        do_constant_folding=True,
        # input_names=list(dummy_inputs.keys()),
        output_names=output_names,
    )
    with ENABLE_CUDA_KERNEL():
        # export non-quantized model to tensorRT for benchmark
        graph = load_onnx_graph(onnx_import_file="out/onnx.model")

        # 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
        # ------------------------------------------------------------
        quantizer = PFL.Quantizer(
            platform=QUANT_PLATFROM, graph=graph
        )  # 取得 TRT_INT8 所对应的量化器
        dispatching = PFL.Dispatcher(graph=graph).dispatch(  # 生成调度表
            quant_types=quantizer.quant_operation_types
        )

        # 为算子初始化量化信息
        for op in graph.operations.values():
            quantizer.quantize_operation(op_name=op.name, platform=dispatching[op.name])

        # 初始化执行器
        collate_fn = lambda x: x.to(DEVICE)
        executor = TorchExecutor(graph=graph, device=DEVICE)
        executor.tracing_operation_meta(
            inputs=torch.rand(size=[1] + INPUT_SHAPE).to(DEVICE)
        )
        executor.load_graph(graph=graph)

        export_ppq_graph(
            graph=graph,
            platform=TargetPlatform.ONNX,
            graph_save_to="./Output/fp32/model_fp32.onnx",
            save_as_external_data=True,
        )
        # graph = load_torch_model(
        #     model=model, sample=torch.zeros(size=[1] + INPUT_SHAPE).to(DEVICE)
        # )
        #

        # ------------------------------------------------------------
        # 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
        # ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
        # ------------------------------------------------------------
        pipeline = PFL.Pipeline(
            [
                QuantizeSimplifyPass(),
                QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
                ParameterQuantizePass(),
                RuntimeCalibrationPass(),
                PassiveParameterQuantizePass(),
                QuantAlignmentPass(force_overlap=True),
                ParameterBakingPass(),
            ]
        )

    with ENABLE_CUDA_KERNEL():
        # 调用管线完成量化
        pipeline.optimize(
            graph=graph,
            dataloader=calibration_dataloader,
            verbose=True,
            calib_steps=32,
            collate_fn=collate_fn,
            executor=executor,
        )

        graphwise_error_analyse(
            graph=graph,
            running_device="cuda",
            dataloader=calibration_dataloader,
            collate_fn=lambda x: x.to(DEVICE),
        )

    builder = trt_infer.EngineBuilder()
    builder.create_network("./Output/fp32/model_fp32.onnx", external_data=True)
    builder.create_engine(engine_path="model_fp32.engine", precision="fp32")

    if CFG_VALID_RESULT:
        executor = TorchExecutor(graph=graph)
        ref_results = []
        for sample in tqdm(
            calibration_dataloader,
            desc="PPQ GENERATEING REFERENCES",
            total=len(calibration_dataloader),
        ):
            result = executor.forward(inputs=sample.to(DEVICE))[0]
            result = result.cpu().reshape([-1, 1000])
            ref_results.append(result)

    # export model to disk.
    export_ppq_graph(
        graph=graph,
        platform=TargetPlatform.TRT_INT8,
        graph_save_to="./Output/int8/model_int8.onnx",
        save_as_external_data=True,
    )
    builder = trt_infer.EngineBuilder()
    builder.create_network("./Output/int8/model_int8.onnx", external_data=True)
    builder.create_engine(engine_path="model_int8.engine", precision="int8")
    if CFG_VALID_RESULT:
        # compute simulating error
        trt_outputs = infer_trt(
            model_path="model_int8.engine",
            samples=[convert_any_to_numpy(sample) for sample in calibration_dataloader],
        )

        error = []
        for ref, real in zip(ref_results, trt_outputs):
            ref = convert_any_to_torch_tensor(ref).float()
            real = convert_any_to_torch_tensor(real).float()
            error.append(torch_snr_error(ref, real))
        error = sum(error) / len(error) * 100
        print(f"Simulating Error: {error: .4f}%")

    # benchmark with onnxruntime int8
    benchmark_samples = [
        np.zeros(shape=[BATCHSIZE, 3, 224, 224], dtype=np.float32) for _ in range(512)
    ]
    print(f"Start Benchmark with tensorRT (Batchsize = {BATCHSIZE})")
    tick = time.time()
    infer_trt(model_path="model_fp32.engine", samples=benchmark_samples)
    tok = time.time()
    print(f"Time span (FP32 MODE): {tok - tick : .4f} sec")

    tick = time.time()
    infer_trt(model_path="model_int8.engine", samples=benchmark_samples)
    tok = time.time()
    print(f"Time span (INT8 MODE): {tok - tick  : .4f} sec")
