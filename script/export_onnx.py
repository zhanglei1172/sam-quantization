"""这个脚本将教会你如何使用 PPQ 量化自定义算子"""

import numpy as np
import onnx
import torch
import torch.nn.functional as F

from segment_anything.build_sam import sam_model_registry

import random

DEVICE = "cuda"
BATCHSIZE = 1
INPUT_SHAPE = [3, 1024, 1024]
BATCH_SHAPE = [BATCHSIZE] + INPUT_SHAPE
CALIBRATION_SIZE = 10
output_names = ["image_embedding"]
# ppq.core.common.COMPUTING_OP.update({"LayerNormalization"})
from albumentations import *
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
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(DEVICE)
model = model.image_encoder


with torch.no_grad():
    torch.onnx.export(
        model,
        torch.rand(size=[1] + INPUT_SHAPE).to(DEVICE),
        "Output/onnx.model",
        export_params=True,
        verbose=False,
        opset_version=11,
        do_constant_folding=True,
        # input_names=list(dummy_inputs.keys()),
        output_names=output_names,
    )
    onnx_model = onnx.load("Output/onnx.model")
    model_simp, check = simplify(onnx_model)  # 对onnx模型进行简化，消除冗余算子
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "Output/onnx.model", save_as_external_data=True)
