import numpy as np
import torch

import random

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

import monai
import requests
import time
from albumentations import *
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from gptq import *
from PIL import Image
from quant import *
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


def quantize_layer(layer, inps, nsamples, dev):
    outs = []
    if args.nearest:
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)
    for j in range(nsamples):
        outs.append(layer(inps[j : j + 1]))
    # layers[i] = layer.cpu()
    # del layer
    # outs = torch.cat(outs, dim=0)
    torch.cuda.empty_cache()
    return layer, inps, outs


@torch.no_grad()
def quantize_image_encoder(model, images, dev):
    nsamples = len(images)

    torch.cuda.empty_cache()
    # model.patch_embed = model.patch_embed.to(dev)
    layer = model.patch_embed.to(dev)
    layer, _, inps = quantize_layer(
        layer,
        images,
        nsamples,
        dev,
    )
    inps = torch.cat(inps, dim=0)
    model.patch_embed = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    if model.pos_embed is not None:
        inps += model.pos_embed.to(dev)

    # outs = torch.zeros_like(inps)
    layers = model.blocks
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        layer, inps, outs = quantize_layer(layer, inps, nsamples, dev)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = torch.cat(outs, dim=0), inps

    layer = model.neck.to(dev)
    layer, inps, outs = quantize_layer(layer, inps.permute(0, 3, 1, 2), nsamples, dev)
    model.neck = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)


@torch.no_grad()
def quantize_prompt_encoder(model, dev, points=None, boxes=None, masks=None):
    layer = model.to(dev)
    torch.cuda.empty_cache()
    sparse_embeddings = None
    bs = model._get_batch_size(points, boxes, masks)
    sparse_embeddings = torch.empty(
        (bs, 0, model.embed_dim), device=model._get_device()
    )
    if points is not None:
        coords, labels = points
        point_embeddings = model._embed_points(coords, labels, pad=(boxes is None))
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
    if boxes is not None:
        box_embeddings = model._embed_boxes(boxes)
        sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

    if masks is not None:
        layer, inps, dense_embeddings = quantize_layer(layer._embed_masks, masks, len(masks), dev)
        dense_embeddings = torch.cat(dense_embeddings, dim=0)
    else:
        dense_embeddings = model.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, model.image_embedding_size[0], model.image_embedding_size[1]
        )

    model = layer.cpu()
    del layer
    torch.cuda.empty_cache()
    return sparse_embeddings, dense_embeddings


@torch.no_grad()
def quantize_mask_decoder(
    model,
    dev,
    image_embeddings,
    image_pe,
    sparse_prompt_embeddings,
    dense_prompt_embeddings,
    multimask_output=False,
):
    layer = model.to(dev)
    torch.cuda.empty_cache()

    outs = []

    subset = find_layers(layer)
    for name in subset:
        quantizer = Quantizer()
        quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
        W = subset[name].weight.data
        quantizer.find_params(W, weight=True)
        subset[name].weight.data = quantize(
            W, quantizer.scale, quantizer.zero, quantizer.maxq
        ).to(next(iter(layer.parameters())).dtype)
    for j in range(len(image_embeddings)):
        outs.append(
            layer.predict_masks(
                image_embeddings=image_embeddings[j : j + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings[j : j + 1],
                dense_prompt_embeddings=dense_prompt_embeddings[j : j + 1],
            )
        )
    masks, iou_pred = zip(*outs)
    torch.cuda.empty_cache()

    model = layer.cpu()
    del layer
    torch.cuda.empty_cache()

    if multimask_output:
        mask_slice = slice(1, None)
    else:
        mask_slice = slice(0, 1)
    masks = torch.cat(masks, dim=0)[:, mask_slice, :, :]
    iou_pred = torch.cat(iou_pred, dim=0)[:, mask_slice]

    # Prepare output
    return masks, iou_pred


@torch.no_grad()
def quantize_model(model, data, dev):
    print("Evaluating ...")

    nsamples = len(data)

    # for i in range(nsamples):
    #     batch = testloader[i:i+1].to(dev)
    #     try:
    #         model(batch)
    #     except ValueError:
    #         pass
    images, gt_masks, points = [], [], []
    for batch_data in data:
        images.append(batch_data["images"])
        gt_masks.append(batch_data["instances"])
        points.append(batch_data["points"])

    # dtype = next(iter(model.parameters())).dtype

    images, gt_masks, points = (
        torch.cat(images, dim=0).to(dev),
        torch.cat(gt_masks, dim=0).to(dev),
        torch.cat(points, dim=0).to(dev),
    )
    torch.cuda.empty_cache()
    image_embedding = quantize_image_encoder(model.image_encoder, images, dev)
    prev_masks = torch.zeros_like(gt_masks)
    # sparse_embeddings, dense_embeddings = model.prompt_encoder(
    #     input_points=input_points,
    #     input_labels=input_labels,
    #     input_boxes=input_boxes,
    #     input_masks=input_masks,
    # )
    click_points = []
    click_labels = []
    batch_points, batch_labels = get_next_click_torch(prev_masks, gt_masks)

    points_co = torch.cat(batch_points, dim=0).to(dev)
    points_la = torch.cat(batch_labels, dim=0).to(dev)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_multi = torch.cat(click_points, dim=1).to(dev)
    labels_multi = torch.cat(click_labels, dim=1).to(dev)

    points_input = points_multi
    labels_input = labels_multi

    prev_masks = torch.zeros_like(gt_masks)
    low_res_masks = F.interpolate(
        prev_masks.float(), size=(crop_size[0] // 4, crop_size[1] // 4)
    )
    sparse_embeddings, dense_embeddings = quantize_prompt_encoder(
        model.prompt_encoder,
        dev,
        points=[points_input, labels_input],
        boxes=None,
        masks=low_res_masks,  # TODO
    )

    low_res_masks, iou_predictions = quantize_mask_decoder(
        model.mask_decoder,
        dev,
        image_embedding,
        model.prompt_encoder.get_dense_pe().to(dev),
        sparse_embeddings,
        dense_embeddings,
        multimask_output=False,
    )

    torch.save(model.state_dict(), f"./checkpoints/sam-{args.wbits}.pt")


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
    model = sam_model_registry[mt](checkpoint=model_type[mt])
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

    # eval_origin(model, testloader, DEV)
    quantize_model(model, train_data, DEV)
    # model.load_state_dict(torch.load(f"./checkpoints/sam-{args.wbits}.pt"))
    # main(model.to(DEV), val_data, args, device)
