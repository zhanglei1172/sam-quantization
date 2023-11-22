import numpy as np
import torch

import random
from pathlib import Path
from tqdm import tqdm

seed = 0
# def set_seed(seed):
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from segment_anything.build_sam import sam_model_registry

# import monai
import requests
import time
from albumentations import *
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from gptq import *
from gptq_triton import QuantLinear, load_quant, quant_linear
from PIL import Image

# from quant import *
from script.evaluation2 import get_next_click_torch, main
from transformers import SamModel, SamProcessor
from typing import Optional
from utils.modelutils import *
from utils.utils import *

# from datasets import DownloadConfig


# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open("car.png").convert("RGB")


# torch.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def get_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


@torch.no_grad()
def bench_speed(model, inp_shape, dtype, device, num_iters=100, warmup_iters=25):
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    inp = torch.randn(inp_shape, dtype=dtype).to(device)
    print("Warm up...")
    for _ in range(warmup_iters):
        model(inp)
    print("Speed test...")
    torch.cuda.synchronize()
    tik = time.time()
    for _ in tqdm(range(num_iters)):
        model(inp)
    torch.cuda.synchronize()
    tok = time.time()
    del inp
    torch.cuda.empty_cache()
    model.to("cpu")
    print(f"Average time per iteration: {(tok - tik) / num_iters}")
    return (tok - tik) / num_iters


if __name__ == "__main__":
    import argparse
    from utils.datautils import *

    parser = argparse.ArgumentParser()
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
        "--num_workers",
        action="store_true",
    )
    parser.add_argument(
        "--save",
    )

    args = parser.parse_args()
    assert args.batch_size == 1, "Batch size must be 1 for calibration."
    # set_seed(args.seed)
    model_type = {
        # 'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
        # 'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
        "vit_h": args.model_path,
    }
    device = "cuda"
    mt = "vit_h"
    model = sam_model_registry[mt](checkpoint=None)

    # model = get_llama(args.model)
    model.eval()

    trainset = SBDDataset(
        args.dataset_dir,
        split="train",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=args.nsamples,
    )
    train_data = DataLoader(
        trainset,
        args.batch_size,
        sampler=get_sampler(trainset, shuffle=False, distributed=False),
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    valset = SBDDataset(
        args.dataset_dir,
        split="val",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500,
    )
    val_data = DataLoader(
        valset,
        args.batch_size,
        # sampler=get_sampler(valset, shuffle=False, distributed=False),
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # bench_speed(model.image_encoder, (1, 3, 1024, 1024), torch.float32, device)
    model.to(torch.bfloat16)
    # bench_speed(model.image_encoder, (1, 3, 1024, 1024), torch.float16, device)

    model = load_quant(model, args.save, sub_module="image_encoder", fuse_mlp=False)
    model.eval()
    bench_speed(model.image_encoder, (1, 3, 1024, 1024), torch.bfloat16, device)

    main(model.to(DEV), val_data, args, device)
