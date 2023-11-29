import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segment_anything.build_sam import sam_model_registry

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
    split="train",
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


@torch.no_grad()
def get_static_layer_scales(
    model,
    dataloader,
    num_samples=512,
):
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))

    iter_data = iter(dataloader)
    for i in pbar:
        data = next(iter_data)["images"].to(device).to(dtype)
        model(data)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    layer_scales = []
    for idx in range(len(model.blocks)):
        scale_dict = {}
        scale_dict["qkv_input_scale"] = (
            act_dict[f"blocks.{idx}.attn.qkv"]["input"] / 127
        )
        scale_dict["qkv_output_scale"] = (
            act_dict[f"blocks.{idx}.attn.qkv"]["output"] / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[f"blocks.{idx}.attn.proj"]["input"] / 127
        )
        scale_dict["fc1_input_scale"] = act_dict[f"blocks.{idx}.mlp.lin1"]["input"] / 127
        scale_dict["fc2_input_scale"] = act_dict[f"blocks.{idx}.mlp.lin2"]["input"] / 127
        layer_scales.append(scale_dict)

    return layer_scales, act_dict


model_type = {
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
mt = "vit_h"
dtype = torch.bfloat16
model_fp = sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda").to(dtype)
model_int8 = int8_sam_model_registry[mt](checkpoint=model_type[mt]).to("cuda").to(dtype)


layer_scales, raw_scales = get_static_layer_scales(
    model_fp.image_encoder,
    val_data,
    num_samples=16,
)

int8_encoder = ImageEncoderViT.from_float(model_fp.image_encoder, layer_scales)

model_int8.image_encoder = int8_encoder



torch.save(model_int8.cpu().state_dict(), "./out/int8sam.pt")

# os.makedirs(os.path.dirname("./out/act_scales.pt"), exist_ok=True)

# torch.save(act_scales, args.output_path)
