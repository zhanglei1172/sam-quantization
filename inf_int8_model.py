import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segment_anything.build_sam import sam_model_registry

import argparse
import functools
import os
from albumentations import *
from collections import defaultdict
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from functools import partial
from int8sam.build_sam import ImageEncoderViT
from int8sam.build_sam import sam_model_registry as int8_sam_model_registry
from script.evaluation2 import main
from tqdm import tqdm

MAX_NUM_POINTS = 24

points_sampler = MultiPointSampler(
    max_num_points=MAX_NUM_POINTS,
    prob_gamma=0.80,
    merge_objects_prob=0.15,
    max_num_merged_objects=2,
)
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

valset = SBDDataset(
    "/data/seg/sbd",
    split="val",
    augmentator=val_augmentator,
    min_object_area=80,
    points_sampler=points_sampler,
    epoch_len=500,
)
val_data = DataLoader(
    valset,
    1,
    # sampler=get_sampler(valset, shuffle=False, distributed=False),
    drop_last=True,
    pin_memory=True,
    num_workers=0,
)


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
dtype = torch.bfloat16
model_int8 = int8_sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda").to(dtype)


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path",
    type=str,
    help="LlaMa model to load; pass location of hugginface converted checkpoint.",
)
parser.add_argument(
    "dataset_dir", type=str, help="Where to extract calibration data from."
)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for sampling the calibration data."
)
parser.add_argument(
    "--nsamples", type=int, default=128, help="Number of calibration data samples."
)
parser.add_argument(
    "--percdamp",
    type=float,
    default=0.01,
    help="Percent of the average Hessian diagonal to use for dampening.",
)
parser.add_argument(
    "--nearest", action="store_true", help="Whether to run the RTN baseline."
)
parser.add_argument(
    "--wbits",
    type=int,
    default=16,
    choices=[2, 3, 4, 8, 16],
    help="#bits to use for quantization; use 16 for evaluating base model.",
)
parser.add_argument(
    "--groupsize",
    type=int,
    default=-1,
    help="Groupsize to use for quantization; default uses full row.",
)
parser.add_argument(
    "--sym", action="store_true", help="Whether to perform symmetric quantization."
)
parser.add_argument(
    "--new-eval",
    action="store_true",
    help="Whether to use the new PTB and C4 eval.",
)
parser.add_argument(
    "--act-order",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic",
)
parser.add_argument(
    "--true-sequential",
    action="store_true",
    help="Whether to run in true sequential model.",
)
parser.add_argument(
    "--save",
)
parser.add_argument(
    "--num_workers",
    action="store_true",
)

args = parser.parse_args()

model_int8.load_state_dict(torch.load("./out/int8sam.pt"))
model_int8.eval()

main(model_int8.to(torch.bfloat16).to("cuda"), val_data, args, "cuda")
# os.makedirs(os.path.dirname("./out/act_scales.pt"), exist_ok=True)

# torch.save(act_scales, args.output_path)
