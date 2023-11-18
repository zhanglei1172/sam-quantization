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
from gptq_triton import QuantLinear, quant_linear
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
        layer, inps, dense_embeddings = quantize_layer(
            layer.mask_downscaling, masks, len(masks), dev
        )
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


@torch.no_grad()
def sam_sequential(
    model,
    dataloader,
    device,
    wbits: int,
    nsamples: int,
    true_sequential: bool,
    sym: bool,
    percdamp: float,
    groupsize: int,
    act_order: bool,
    name_prefix=""
):
    # Prepare
    layers = model.blocks
    dtype = next(iter(model.parameters())).dtype
    inps = [None] * nsamples
    outs = [None] * nsamples
    # outs = torch.zeros_like(inps)

    # Move the first layer to GPU
    model.patch_embed = model.patch_embed.to(device)
    if model.pos_embed is not None:
        # parameta asign
        model.pos_embed.data = model.pos_embed.data.to(device)
    layers[0] = layers[0].to(device)

    # Create a dummy layer that catches the input and attention mask, and then bails
    # This allows us to capture all the inputs to the first layer for the calibration data
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        if i == nsamples:
            break
        try:
            model(batch["images"].to(dtype).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Move things back to the CPU (but not the first layer, since we'll just move it back to GPU immediately below)
    model.patch_embed = model.patch_embed.cpu()
    if model.pos_embed is not None:
        model.pos_embed.data = model.pos_embed.data.cpu()
    torch.cuda.empty_cache()

    quantizers = {}

    # Layers are quantized in order, and only one layer lives on the GPU at a time to save memory
    # Otherwise quantizing large models would be impossible (NOTE for future readers: are you enjoying your 1TB VRAM?)
    for i, layer in tqdm(enumerate(layers), total=len(layers)):
        layer = layer.to(device)
        full = {
            name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)
        }

        if true_sequential:
            sequential = [
                ["attn.qkv"],
                ["attn.proj"],
                ["mlp.lin1", "mlp.lin2"],
            ]
        else:
            sequential = [list(full.keys())]

        # For each subset of linear layers
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}

            # Prepare a quantizer for each linear layer
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    wbits, perchannel=True, sym=sym, mse=False
                )

            # Feed data to the quantizer, and save outs
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j],
                )
            for h in handles:
                h.remove()

            # With the data collected, quantize the layers
            for name in subset:
                print(i, name)
                scale, zero = gptq[name].fasterquant(
                    percdamp=percdamp, groupsize=groupsize, actorder=act_order
                )
                quantizers[name_prefix+"blocks.%d.%s" % (i, name)] = (
                    gptq[name].quantizer,
                    scale,
                    zero,
                )
                gptq[name].free()

        # Save outputs of the layer after quantization, so we can feed them into the next layer
        for j in range(nsamples):
            outs[j] = layer(
                inps[j],
            )

        # Move the layer back to the CPU, and free up memory
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        # Swap buffers
        inps, outs = outs, inps

    return quantizers


def sam_pack(model, quantizers, wbits: int, groupsize: int):
    # Find all the quantized layers
    layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    layers = {n: layers[n] for n in quantizers}

    # Replace all applicable instances of Linear with QuantLinear in the model
    quant_linear.make_quant(model, wbits, groupsize)

    for name, m in tqdm(model.named_modules(), total=len(list(model.named_modules()))):
        if not isinstance(m, QuantLinear):
            continue

        quantizer, scale, zero = quantizers[name]
        quantizer, scale, zero = quantizer.cpu(), scale.cpu(), zero.cpu()
        pack_linear(m, layers[name].weight.data, scale, zero, layers[name].bias.data)


def pack_linear(
    quant,
    weights: torch.FloatTensor,
    scales: torch.FloatTensor,
    zeros,
    bias: Optional[torch.FloatTensor],
):
    """
    Packs the quantized weights, scales, and zero points into a QuantLinear layer
    """
    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales

    quant.scales = scales.clone().to(torch.float16)

    if quant.bias is not None:
        quant.bias = bias.clone().to(torch.float16)

    # Round weights to nearest integer based on scale and zero point
    # Each weight will be one int, but should not exceed quant.bits
    intweight = []
    for idx in range(quant.infeatures):
        g_idx = idx // quant.groupsize
        # TODO: This is oddly complex.  The `gptq.quantize` function does `return scale * (q - zero)`, so shouldn't
        # this just be `q = torch.round((weights[:,idx] / scales[g_idx]) + zero[g_idx])`?
        q = torch.round((weights[:, idx] + scale_zeros[g_idx]) / scales[g_idx]).to(
            torch.int32
        )
        intweight.append(q[:, None])
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()

    # Now pack the weights into uint32's
    # qweight = torch.zeros((intweight.shape[0] // 32 * quant.bits, intweight.shape[1]), dtype=torch.int32)
    quant.qweight.zero_()
    i = 0
    row = 0
    while row < quant.qweight.shape[0]:
        if quant.bits in [2, 4, 8]:
            for j in range(i, i + (32 // quant.bits)):
                quant.qweight[row] |= intweight[j] << (quant.bits * (j - i))
            i += 32 // quant.bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    # Subtract 1 from the zero point
    zeros = zeros - 1

    # Pack the zero points into uint32's
    zeros = zeros.to(torch.int32)
    # qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
    quant.qzeros.zero_()
    i = 0
    col = 0
    while col < quant.qzeros.shape[1]:
        if quant.bits in [2, 4, 8]:
            for j in range(i, i + (32 // quant.bits)):
                quant.qzeros[:, col] |= zeros[:, j] << (quant.bits * (j - i))
            i += 32 // quant.bits
            col += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")


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

    model.half()
    # eval_origin(model, testloader, DEV)
    # quantize_model(model, train_data, DEV)
    quantizers = sam_sequential(
        model.image_encoder,
        train_data,
        DEV,
        args.wbits,
        args.nsamples,
        args.true_sequential,
        args.sym,
        args.percdamp,
        args.groupsize,
        args.act_order,
        name_prefix=""
    )
    args.save = Path(args.save)
    args.save.mkdir(parents=True, exist_ok=True)
    sam_pack(model.image_encoder, quantizers, args.wbits, args.groupsize)
    torch.save(model.state_dict(), args.save / "model.pt")
    with open(args.save / "quant_config.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "wbits": args.wbits,
                    "groupsize": args.groupsize,
                }
            )
        )
    # model.load_state_dict(torch.load(f"./checkpoints/sam-{args.wbits}.pt"))
    main(model.to(DEV), val_data, args, device)
