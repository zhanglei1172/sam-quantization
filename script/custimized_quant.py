"""这个脚本将教会你如何使用 PPQ 量化自定义算子"""

import numpy as np
import torch
import torch.nn.functional as F

from segment_anything.build_sam import sam_model_registry

import ppq.core.common
import random
from ppq import *
from ppq.api import *
from ppq.executor.op.torch.base import ASSERT_NUM_OF_INPUT, VALUE_TO_EXECUTING_DEVICE
from ppq.quantization.quantizer import OnnxruntimeQuantizer
from utils.datautils import get_loaders

DEVICE = "cuda"
PLATFORM = TargetPlatform.EXTENSION
BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 32
output_names = ["image_embedding"]

# ppq.core.common.COMPUTING_OP.update({"LayerNormalization"})


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
model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_b"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(DEVICE)
model = model.image_encoder

calibration_dataloader = []
for trainloader, testloader in zip(
    *get_loaders("/data/SA-1B/sa_000000/", CALIBRATION_SIZE)
):
    calibration_dataloader.append(trainloader["pixel_values"])


class MyTVMQuantizer(OnnxruntimeQuantizer):
    @property
    def quant_operation_types(self) -> set:
        """覆盖 quant_operation_types  自定义需要量化的算子"""
        return {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "Relu",
            "PRelu",
            "Clip",
            "Pad",
            "Resize",
            "MaxPool",
            "AveragePool",
            "GlobalMaxPool",
            "GlobalAveragePool",
            "Mul",
            "Add",
            "Max",
            "Sub",
            "Div",
            "LeakyRelu",
            "Concat",
            "Sigmoid",
            "Slice",
            # "LayerNormalization",
            # "Gelu",
            # "Softmax",
        }

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation=operation)
        if operation.type == "LayerNormalization":
            # LayerNorm 输入按 power of 2 量化
            inp_config = config.input_quantization_config[0]
            inp_config.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.LINEAR
                + QuantizationProperty.PER_TENSOR
            )

            # config.input_quantization_config[0].observer_algorithm = "Minmax"

            # layerNorm weight 和 bias 都不量化
            wconfig = config.input_quantization_config[1]
            bconfig = config.input_quantization_config[2]
            wconfig.state = QuantizationStates.FP32
            bconfig.state = QuantizationStates.FP32

            # 输出量化
            output_policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.LINEAR
                + QuantizationProperty.PER_TENSOR
            )
            config.output_quantization_config[0].policy = output_policy
            # config.output_quantization_config[0].observer_algorithm = "Minmax"
        elif operation.type == "Softmax":
            inp_config = config.input_quantization_config[0]

            # config.input_quantization_config[0].observer_algorithm = "Minmax"

            # 输出量化
            output_policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.LINEAR
                + QuantizationProperty.PER_TENSOR
            )
            config.output_quantization_config[0].policy = output_policy
            # config.output_quantization_config[0].observer_algorithm = "Minmax"
        elif operation.type == "Gelu":
            inp_config = config.input_quantization_config[0]
            inp_config.num_of_bits = 16
            inp_config.quant_max = 2**15 - 1
            inp_config.quant_min = -(2**15)

            # config.input_quantization_config[0].observer_algorithm = "Minmax"

            # 输出量化
            output_policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL
                + QuantizationProperty.LINEAR
                + QuantizationProperty.PER_TENSOR
            )
            config.output_quantization_config[0].policy = output_policy
            # config.output_quantization_config[0].observer_algorithm = "Minmax"
        return config


from torch.autograd import Function


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def int_exp_shift(x_int, scaling_factor, n):
    x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

    with torch.no_grad():
        x0_int = torch.floor(-1.0 / scaling_factor)
    x_int = torch.max(x_int, n * x0_int)

    q = floor_ste.apply(x_int / x0_int)
    r = x_int - x0_int * q
    exp_int = r / 2 - x0_int
    exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (n - q)), min=0)
    scaling_factor = scaling_factor / 2**n
    return exp_int, scaling_factor


def int_layernorm_forward(
    op: Operation, values: List[torch.Tensor], ctx=None, **kwargs
):
    weight = values[1]
    bias = values[2]
    x = values[0]
    config = op.config.input_quantization_config[0]
    scaling_factor = config.scale

    n = torch.tensor(x.shape[2], dtype=torch.float, device=x.device)
    dim_sqrt = torch.sqrt(n)

    # Normalization: computes mean and variance(std)
    x_int = x / scaling_factor
    mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
    y_int = x_int - mean_int
    y_sq_int = y_int**2
    var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

    # Integer Iteration
    k = 2**16
    for _ in range(10):
        k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
        k = k_1
    std_int = k

    factor = floor_ste.apply((2**31 - 1) / std_int)
    y_int = floor_ste.apply(y_int * factor / 2)
    scaling_factor = dim_sqrt / 2**30

    # scaling and shifting
    bias = bias.detach() / (weight.detach())
    bias_int = floor_ste.apply(bias / scaling_factor)

    bias_integer = bias_int

    y_int = y_int + bias_int
    scaling_factor = scaling_factor * weight
    x = y_int * scaling_factor
    return x


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


def int_gelu_forward(op: Operation, values: List[torch.Tensor], ctx=None, **kwargs):
    output_bit = 8
    n = 23
    config = op.config.input_quantization_config[0]
    scaling_factor = config.scale
    pre_x_int = values[0] / scaling_factor
    scaling_factor_sig = scaling_factor * 1.702

    x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
    x_int = pre_x_int - x_int_max

    exp_int, _ = int_exp_shift(x_int, scaling_factor_sig, n)  # e^(x-x_max)

    exp_int_max, _ = int_exp_shift(-x_int_max, scaling_factor_sig, n)  # e^(-x_max)
    exp_int_sum = exp_int + exp_int_max

    exp_int_sum.clamp_max_(2**31 - 1)
    factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
    sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31 - output_bit + 1))
    sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (output_bit - 1)]).cuda()

    x_int = pre_x_int * sigmoid_int
    scaling_factor = scaling_factor * sigmoid_scaling_factor
    return x_int * scaling_factor


def int_softmax_forward(op: Operation, values: List[torch.Tensor], ctx=None, **kwargs):
    n = 15
    output_bit = 8
    config = op.config.input_quantization_config[0]
    scaling_factor = config.scale
    x_int = values[0] / scaling_factor
    x_int_max, _ = x_int.max(dim=-1, keepdim=True)
    x_int = x_int - x_int_max

    exp_int, _ = int_exp_shift(x_int, scaling_factor, n)
    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

    exp_int_sum.clamp_max_(2**31 - 1)
    factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
    exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - output_bit + 1))
    scaling_factor = torch.Tensor([1 / 2 ** (output_bit - 1)]).cuda()

    return exp_int * scaling_factor


# register_operation_handler(
#     handler=gelu_forward,
#     operation_type="GeLU",
#     platform=TargetPlatform.TRT_INT8,
# )
register_network_quantizer(quantizer=MyTVMQuantizer, platform=TargetPlatform.EXTENSION)
QS = QuantizationSettingFactory.default_setting()
QS.lsq_optimization = False

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
        # QS = QuantizationSettingFactory.default_setting()
        ir = load_onnx_graph(onnx_import_file="out/onnx.model")
        quantizer = MyTVMQuantizer(ir)
        # 默认调度失效，直接手动调度所有 LayerNormPlugin 送上量化区

        from ppq.IR import GraphMerger

        # processor = GraphMerger(ir)
        # processor.fuse_layernorm()
        # processor.fuse_gelu()
        # search_engine = SearchableGraph(ir)
        # 为算子初始化量化信息
        # for name, op in ir.operations.items():
        #     if op.type in {
        #         "Conv",
        #         "ConvTranspose",
        #         "MatMul",
        #         "Gemm",
        #         "PPQBiasFusedMatMul",
        #         "LayerNormalization",
        #         "Softmax"
        #     }:
        #         QS.dispatching_table.append(
        #             operation=op.name, platform=TargetPlatform.INT8
        #         )
        # quantizer.quantize_operation(name, platform=TargetPlatform.INT8)
        # for op in ir.operations.values():
        # if op.type == "Softmax":
        #     QS.dispatching_table.append(
        #         operation=op.name, platform=TargetPlatform.TRT_INT8
        #     )
        # elif op.type == "GeLU":
        #     QS.dispatching_table.append(
        #         operation=op.name, platform=TargetPlatform.TRT_INT8
        #     )
        # elif op.type == "LayerNormalization":
        #     QS.dispatching_table.append(
        #         operation=op.name, platform=TargetPlatform.TRT_INT8
        #     )
        # 初始化执行器
        collate_fn = lambda x: x.to(DEVICE)
        # executor = TorchExecutor(graph=ir, device=DEVICE)
        # executor.tracing_operation_meta(inputs=torch.zeros(size=BATCH_SHAPE).cuda())
        # executor.load_graph(graph=ir)
        # QS.dispatching_table.append(
        #     operation="/patch_embed/proj/Conv", platform=TargetPlatform.FP32
        # )

        quantized = quantize_native_model(
            model=ir,
            calib_dataloader=calibration_dataloader,
            calib_steps=32,
            input_shape=BATCH_SHAPE,
            collate_fn=collate_fn,
            platform=PLATFORM,
            device=DEVICE,
            verbose=0,
            setting=QS,
        )

        # layerwise_error_analyse(
        #     graph=quantized,
        #     running_device=DEVICE,
        #     interested_outputs=None,
        #     dataloader=calibration_dataloader,
        #     collate_fn=collate_fn,
        # )

        reports = graphwise_error_analyse(
            graph=quantized,
            running_device=DEVICE,
            collate_fn=collate_fn,
            dataloader=calibration_dataloader,
        )

        export_ppq_graph(
            graph=quantized,
            platform=TargetPlatform.ONNXRUNTIME,
            # platform=TargetPlatform.ONNX,
            graph_save_to="Output/quantized.onnx",
            config_save_to="Output/quantized.json",
            save_as_external_data=True,
        )
