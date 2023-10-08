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

import torch.profiler


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    # on_trace_ready=trace_handler
    on_trace_ready=torch.profiler.tensorboard_trace_handler("log")
    # used when outputting for tensorboard
) as p:
    with torch.no_grad():
        for iter in range(10):
            model(torch.rand(size=[1] + INPUT_SHAPE).to(DEVICE))
            p.step()
