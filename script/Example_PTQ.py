import numpy as np
import torch
import torchvision

from segment_anything.build_sam import sam_model_registry

import os
import ppq.lib as PFL
import random
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse, layerwise_error_analyse
from ppq.api import ENABLE_CUDA_KERNEL, export_ppq_graph, load_onnx_graph
from ppq.core import convert_any_to_numpy
from ppq.quantization.optim import *
from utils.datautils import get_loaders


def set_seed(seed):
    # def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 32
DEVICE = "cuda"
PLATFORM = TargetPlatform.TRT_INT8

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
for trainloader, testloader in zip(*get_loaders("/data/SA-1B/sa_000000/", CALIBRATION_SIZE)):
    calibration_dataloader.append(trainloader['pixel_values'])

with torch.no_grad():
    torch.onnx.export(
            model,
            torch.rand(size=[1]+INPUT_SHAPE).to(DEVICE),
            "out/onnx.model",
            export_params=True,
            verbose=False,
            opset_version=11,
            do_constant_folding=True,
            # input_names=list(dummy_inputs.keys()),
            output_names=output_names,
        )
    with ENABLE_CUDA_KERNEL():
        graph = load_onnx_graph(onnx_import_file="out/onnx.model")
        # graph = load_torch_model(
        #     model=model, sample=torch.zeros(size=[1] + INPUT_SHAPE).to(DEVICE)
        # )
        # ------------------------------------------------------------
        # 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
        # ------------------------------------------------------------
        quantizer = PFL.Quantizer(
            platform=PLATFORM, graph=graph
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
        executor.tracing_operation_meta(inputs=torch.rand(size=[1]+INPUT_SHAPE).to(DEVICE))
        executor.load_graph(graph=graph)

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
        reports = layerwise_error_analyse(
            graph=graph,
            running_device=DEVICE,
            collate_fn=collate_fn,
            dataloader=calibration_dataloader,
        )
        os.exit(0)


        graphwise_error_analyse(
            graph=graph,
            running_device="cuda",
            dataloader=calibration_dataloader,
            collate_fn=lambda x: x.to(DEVICE),
        )

        export_ppq_graph(
            graph=graph,
            platform=PLATFORM,
            graph_save_to="Output/quantized.onnx", # large file must be directory
            config_save_to="Output/quantized.json",
            save_as_external_data=True,
        )

        results, executor = [], TorchExecutor(graph=graph)
        # for idx, data in enumerate(calibration_dataloader):
        #     arr = convert_any_to_numpy(executor(data.to(DEVICE))[0])
        #     arr.tofile(f"Output/Result/{idx}.bin")

        from ppq.utils.TensorRTUtil import build_engine

        build_engine(
            onnx_file="Output/quantized.onnx",
            int8_scale_file="Output/quantized.json",
            engine_file="Output/INT8.engine",
            int8=True,
            fp16=True,
            external_data=True
        )
        build_engine(
            onnx_file="Output/quantized.onnx",
            engine_file="Output/FP16.engine",
            int8=False,
            fp16=True,
            external_data=True
        )
        build_engine(
            onnx_file="Output/quantized.onnx",
            engine_file="Output/FP32.engine",
            int8=False,
            fp16=False,
            external_data=True
        )
