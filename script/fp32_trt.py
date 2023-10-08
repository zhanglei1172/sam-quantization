import numpy as np
import onnx
import torch
import torch.nn.functional as F

from segment_anything.build_sam import sam_model_registry

import ppq.core.common
import random
from ppq import *
from ppq.api import *
from ppq.executor.op.torch.base import ASSERT_NUM_OF_INPUT, VALUE_TO_EXECUTING_DEVICE
from ppq.quantization.quantizer import OnnxruntimeQuantizer, TensorRTQuantizer
from utils.datautils import get_loaders

DEVICE = "cuda"
PLATFORM = TargetPlatform.EXTENSION
BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 32
output_names = ["image_embedding"]
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
model_type = {
    # 'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
    # 'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(DEVICE)
model = model.image_encoder

with torch.no_grad():
    torch.onnx.export(
        model,
        torch.rand(size=[1] + INPUT_SHAPE).to(DEVICE),
        "out/onnx_fp32.model",
        export_params=True,
        verbose=False,
        opset_version=11,
        do_constant_folding=True,
        # input_names=list(dummy_inputs.keys()),
        output_names=output_names,
    )
    onnx_model = onnx.load("out/onnx_fp32.model")

    model_simp, check = simplify(onnx_model)  # 对onnx模型进行简化，消除冗余算子
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "out/onnx_fp32.model", save_as_external_data=True)

from ppq.utils.TensorRTUtil import build_engine

build_engine(
    onnx_file="out/onnx_fp32.model",
    engine_file="Output/FP32.engine",
    int8=False,
    fp16=False,
)
