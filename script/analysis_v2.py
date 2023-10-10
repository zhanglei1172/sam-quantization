"""这个脚本将教会你如何使用 PPQ 量化自定义算子"""

import numpy as np
import cv2
import onnx
import torch
import torch.nn.functional as F

from segment_anything.build_sam import sam_model_registry

import glob
import ppq.core.common
import random
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from ppq import *
from ppq.api import *
from ppq.executor.op.torch.base import ASSERT_NUM_OF_INPUT, VALUE_TO_EXECUTING_DEVICE
from ppq.IR import BaseGraph, GraphCommandProcessor
from ppq.quantization.optim import *
from ppq.quantization.quantizer import OnnxruntimeQuantizer, TensorRTQuantizer
from typing import Union
from utils.datautils import get_loaders

DEVICE = "cuda"
PLATFORM = TargetPlatform.EXTENSION
BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 5000
output_names = ["image_embedding"]
import ppq.lib as PFL

# ppq.core.common.COMPUTING_OP.update({"LayerNormalization"})
from albumentations import *
from data.transforms import UniformRandomResize
from onnxsim import simplify


def set_seed(seed):
    # def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(0)
collate_fn = lambda x: x.to(DEVICE)

crop_size = (1024, 1024)
val_augmentator = Compose(
    [
        Resize(1024, 1024),
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
    ],
    p=1.0,
)
MAX_NUM_POINTS = 24
points_sampler = MultiPointSampler(
    max_num_points=MAX_NUM_POINTS,
    prob_gamma=0.80,
    merge_objects_prob=0.15,
    max_num_merged_objects=2,
)
trainset = SBDDataset(
    "/data/seg/sbd/benchmark_RELEASE/dataset/",
    split="train",
    augmentator=val_augmentator,
    min_object_area=80,
    points_sampler=points_sampler,
)
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=0,
)
# Get calibration set.
calibration_dataloader = []
for i, _data in enumerate(train_loader):
    if i == CALIBRATION_SIZE:
        break
    # data = {k: v.to(DEVICE) for k, v in data.items()}
    calibration_dataloader.append(_data["images"])


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(DEVICE)
model = model.image_encoder

# calibration_dataloader = []

# for p in glob.glob("/data/seg/sbd/benchmark_RELEASE/dataset/img/*.jpg")[
#     :CALIBRATION_SIZE
# ]:
#     img = cv2.imread(p)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = val_augmentator(image=img)["image"]
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).unsqueeze(0).float()
#     calibration_dataloader.append(img)


class MyTVMQuantizer(TensorRTQuantizer):
    def __init__(self, graph):
        super().__init__(graph)
        self._num_of_bits = 8
        self._quant_min = -128
        self._quant_max = 127

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        OQC = self.create_default_quant_config(
            policy=self.quantize_policy,
            rounding=self.rounding_policy,
            op=operation,
            num_of_bits=self._num_of_bits,
            exponent_bits=0,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
            observer_algorithm="percentile",
        )

        if operation.type in {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            "PPQBiasFusedMatMul",
        }:
            # base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert (
                operation.num_of_input > 0
            ), "Seems you got a Conv layer with no parameters."

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {"Conv", "ConvTranspose"}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = OQC.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL
                        + QuantizationProperty.LINEAR
                        + QuantizationProperty.PER_CHANNEL
                    )
                    conv_weight_config.channel_axis = (
                        1 if operation.type == "ConvTranspose" else 0
                    )
                    conv_weight_config.observer_algorithm = "minmax"
                    conv_weight_config.quant_max = 127
                    conv_weight_config.quant_min = -128
            # if operation.name.endswith("MatMul_3"):
            #     TQC = OQC.input_quantization_config[0]
            #     TQC.policy = QuantizationPolicy(
            #         QuantizationProperty.SYMMETRICAL
            #         + QuantizationProperty.LINEAR
            #         + QuantizationProperty.POWER_OF_2
            #         + QuantizationProperty.PER_TENSOR
            #     )
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {"Gemm", "MatMul", "PPQBiasFusedMatMul"}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = OQC.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL
                        + QuantizationProperty.LINEAR
                        + QuantizationProperty.PER_CHANNEL
                    )
                    gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = "minmax"
                    gemm_weight_config.quant_max = 127
                    gemm_weight_config.quant_min = -128
                for inp, TQC in zip(operation.inputs, OQC.input_quantization_config):
                    if inp.name in exclude_inps:
                        TQC.state = QuantizationStates.FP32

            if operation.num_of_input > 2:
                bias_config = OQC.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type == "LayerNormalization":
            # Layernorm - gamma and beta need to be FP32
            for TQC in OQC.input_quantization_config[1:]:
                # for TQC in OQC.input_quantization_config:
                TQC.state = QuantizationStates.FP32
            TQC = OQC.output_quantization_config[0]
            TQC.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.LINEAR
                + QuantizationProperty.PER_TENSOR
            )
            # TQC.channel_axis = operation.attributes.get("axis", -1)
            # TQC.observer_algorithm = "minmax"

        elif operation.type == "Attention":
            # Attention - Only input and weight need to be quantized.
            for TQC in OQC.input_quantization_config[2:]:
                TQC.state = QuantizationStates.FP32

        for TQC in OQC.input_quantization_config:
            assert not TQC.policy.has_property(QuantizationProperty.ASYMMETRICAL)

        return OQC

    # @property
    # def quantize_policy(self) -> QuantizationPolicy:
    #     return QuantizationPolicy(
    #         QuantizationProperty.ASYMMETRICAL
    #         + QuantizationProperty.LINEAR
    #         + QuantizationProperty.PER_TENSOR
    #     )


def layernorm_forward(op: Operation, values: List[torch.Tensor], ctx: None, **kwargs):
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    x, weight = values[0], values[1]
    if len(values) == 3:
        bias = values[-1]
    else:
        bias = None

    eps = op.attributes.get("epsilon", 1e-5)
    axis = op.attributes.get("axis", -1)

    if axis != -1 and axis != x.ndim - 1:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + eps)
        output = weight * x + bias
    else:
        normalized_shape = weight.shape
        output = F.layer_norm(x, normalized_shape, weight, bias, eps)

    return output


register_operation_handler(
    handler=layernorm_forward, operation_type="LayerNormalization", platform=PLATFORM
)
register_operation_handler(
    handler=layernorm_forward,
    operation_type="LayerNormalization",
    platform=TargetPlatform.INT8,
)
register_operation_handler(
    handler=layernorm_forward,
    operation_type="LayerNormalization",
    platform=TargetPlatform.FP32,
)

register_operation_handler(
    handler=layernorm_forward,
    operation_type="LayerNormalization",
    platform=TargetPlatform.UNSPECIFIED,
)
register_operation_handler(
    handler=layernorm_forward,
    operation_type="LayerNormalization",
    platform=TargetPlatform.TRT_INT8,
)


register_network_quantizer(quantizer=MyTVMQuantizer, platform=TargetPlatform.EXTENSION)
QS = QuantizationSettingFactory.default_setting()
QS.lsq_optimization = False

with torch.no_grad():
    with ENABLE_CUDA_KERNEL():
        # QS = QuantizationSettingFactory.default_setting()
        ir = load_onnx_graph(onnx_import_file="out/onnx.model")
        from ppq.IR import GraphMerger

        processor = GraphMerger(ir)
        processor.fuse_matmul_add()
        # processor.fuse_gemm()
        processor.fuse_layernorm()
        # processor.fuse_gelu()

        quantizer = MyTVMQuantizer(ir)

        # exclued_ops = {"/patch_embed/proj/Conv"}
        exclued_ops = set()
        search_engine = SearchableGraph(ir)
        # for op in search_engine.opset_matching(
        #     sp_expr=lambda x: x.type == "Softmax",
        #     rp_expr=lambda x, y: y.type != "Split",
        #     ep_expr=lambda x: x.type == "Split",
        #     direction="up",
        # ):
        #     exclued_ops.add(op.name)

        # for op in search_engine.opset_matching(
        #     sp_expr=lambda x: x.type == "LayerNormalization",
        #     rp_expr=lambda x, y: not y.is_computing_op,
        #     ep_expr=lambda x: x.is_computing_op,
        #     direction="down",
        # ):
        #     exclued_ops.add(op.name)
        # convert op to quantable-op

        exclude_inps = set()
        for op in search_engine.opset_matching(
            sp_expr=lambda x: x.name.endswith("attn/Split"),
            rp_expr=lambda x, y: not y.is_computing_op,
            ep_expr=lambda x: x.is_computing_op,
            direction="down",
        ):
            exclued_ops.add(op.name)
            for inp in op.inputs:
                if not inp.is_parameter:
                    exclude_inps.add(inp.name)

        for name, op in ir.operations.items():
            if op.type in {
                "Conv",
                "ConvTranspose",
                "MatMul",
                "Gemm",
                "PPQBiasFusedMatMul",
                "LayerNormalization",
                "Softmax",
            }:
                # if name not in exclued_ops and "mlp" in name and "lin2" in name:
                # if "attn/MatMul_1" not in name and "attn/MatMul_2" not in name:
                quantizer.quantize_operation(name, platform=TargetPlatform.INT8)
        # for name, op in ir.operations.items():
        #     if op.type in quantizer.quant_operation_types:
        #         quantizer.quantize_operation(name, platform=TargetPlatform.INT8)

        QS = QuantizationSettingFactory.default_setting()
        lsq_setting = QS.lsq_optimization_setting
        blockwise_reconstruction_setting = QS.blockwise_reconstruction_setting
        # build quant pipeline.
        pipeline = PFL.Pipeline(
            [
                # LayerwiseEqualizationPass(iteration=10),
                QuantizeSimplifyPass(),
                ParameterQuantizePass(),
                RuntimeCalibrationPass(),  # TODO
                PassiveParameterQuantizePass(),
                # LearnedStepSizePass(
                #     interested_layers=lsq_setting.interested_layers,
                #     lr=lsq_setting.lr,
                #     collecting_device=lsq_setting.collecting_device,
                #     steps=lsq_setting.steps,
                #     gamma=lsq_setting.gamma,
                #     is_scale_trainable=lsq_setting.is_scale_trainable,
                #     block_size=lsq_setting.block_size,
                # ),
                # PassiveParameterQuantizePass(),
                # AdaroundPass(
                #     interested_layers=blockwise_reconstruction_setting.interested_layers,
                #     lr=blockwise_reconstruction_setting.lr,
                #     collecting_device=blockwise_reconstruction_setting.collecting_device,
                #     steps=blockwise_reconstruction_setting.steps,
                #     gamma=blockwise_reconstruction_setting.gamma,
                #     is_scale_trainable=blockwise_reconstruction_setting.is_scale_trainable,
                #     block_size=blockwise_reconstruction_setting.block_size,
                # ),
                # PassiveParameterQuantizePass(),
                # LearnedStepSizePass(steps=500, collecting_device='cuda', block_size=5)
                ParameterBakingPass(),
            ]
        )

        # call pipeline.
        executor = TorchExecutor(graph=ir)
        executor.tracing_operation_meta(torch.zeros(size=BATCH_SHAPE).cuda())

        pipeline.optimize(
            graph=ir,
            dataloader=calibration_dataloader,
            verbose=True,
            calib_steps=10,
            collate_fn=collate_fn,
            executor=executor,
        )

        # reports = layerwise_error_analyse(
        #     graph=ir,
        #     running_device=DEVICE,
        #     interested_outputs=["/blocks.1/attn/MatMul_3_output_0"],
        #     dataloader=calibration_dataloader,
        #     collate_fn=collate_fn,
        #     flatten_start_dim=0,
        # )

        # reports = graphwise_error_analyse_v2(
        #     graph=ir,
        #     running_device=DEVICE,
        #     collate_fn=collate_fn,
        #     dataloader=calibration_dataloader,
        #     interested_op=[
        #         "/blocks.0/attn/MatMul_3",
        #         "/blocks.0/attn/MatMul_2",
        #         "/blocks.0/attn/MatMul_1",
        #         "/blocks.0/attn/MatMul",
        #         "/blocks.0/attn/proj/MatMul",
        #         "/blocks.0/mlp/lin1/MatMul",
        #         "/blocks.0/mlp/lin2/MatMul",
        #         "/blocks.1/attn/qkv/MatMul",
        #         "/neck/neck.2/Conv",
        #     ],
        # )
        reports = graphwise_error_analyse(
            graph=ir,
            running_device=DEVICE,
            collate_fn=collate_fn,
            dataloader=calibration_dataloader,
            flatten_start_dim=0,
        )

        ana_vars = ["/blocks.0/attn/Reshape_6_output_0"]
        for op in ir.operations.values():
            if "PPQ_Operation_1" in op.name:
                for inp in op.inputs:
                    if not inp.is_parameter:
                        ana_vars.append(inp.name)

        # vars = variable_analyse_get(
        #     ir,
        #     interested_outputs=ana_vars,
        #     running_device=DEVICE,
        #     collate_fn=collate_fn,
        #     dataloader=calibration_dataloader,
        #     dequantize=False,
        #     steps=0,
        # )
        ana_vars = ["/blocks.1/attn/proj/MatMul_output_0"]
        y_pred = torch.from_numpy(
            variable_analyse_get(
                ir,
                interested_outputs=ana_vars,
                running_device=DEVICE,
                collate_fn=collate_fn,
                dataloader=calibration_dataloader,
                dequantize=False,
                steps=0,
            )[0]
        )
        y_true = torch.from_numpy(
            variable_analyse_get(
                ir,
                interested_outputs=ana_vars,
                running_device=DEVICE,
                collate_fn=collate_fn,
                dataloader=calibration_dataloader,
                dequantize=True,
                steps=0,
            )[0]
        )
        print(torch_snr_error(y_pred, y_true, flatten_start_dim=0))
        torch.save((y_true), "t")

        y_true = torch.load("t")
        y_pred = x
        from ppq import torch_snr_error

        print(torch_snr_error(y_pred.cpu(), y_true, flatten_start_dim=0))

        import matplotlib.pyplot as plt

        import seaborn as sns

        for v, name in zip(vars, ana_vars):
            C = v.shape[-1]
            plt.clf()
            fig, ax = plt.subplots()
            # ax.boxplot(v.reshape(-1, C), positions=range(1, C + 1))
            ax.vlines(range(1, C + 1), 0, np.abs(v).reshape(-1, C).max(axis=0))
            # sns.boxplot(np.abs(v).reshape(-1, C).max(axis=0))
            # sns.boxplot((v).flatten())

            ax.set_xlabel("Channel")
            ax.set_ylabel("Value")
            ax.set_title("Box Plot of range")
            plt.savefig(f"vis/{name.replace('/','_')}.png")

        export_ppq_graph(
            graph=ir,
            platform=TargetPlatform.TRT_INT8,
            # platform=TargetPlatform.ONNXRUNTIME,
            graph_save_to="Output/quantized_3.onnx",
            config_save_to="Output/quantized.json",
            save_as_external_data=True,
            # size_threshold=1024,
        )
        from ppq.utils.TensorRTUtil import build_engine

        build_engine(
            onnx_file="Output/quantized_3.onnx",
            int8_scale_file="Output/quantized.json",
            engine_file="Output/INT8_3.engine",
            int8=True,
            fp16=True,
            external_data=True,
        )
        # build_engine(
        #     onnx_file="Output/quantized.onnx",
        #     engine_file="Output/FP16.engine",
        #     int8=False,
        #     fp16=True,
        #     external_data=True,
        # )
        # build_engine(
        #     onnx_file="Output/quantized.onnx",
        #     engine_file="Output/FP32.engine",
        #     int8=False,
        #     fp16=False,
        #     external_data=True,
        # )
