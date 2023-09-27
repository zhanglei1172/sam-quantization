import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

import argparse
import math
import os
import pandas as pd
import time
from albumentations import *
from config import Config
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from fq_vit.models.sam.build_sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from models import *
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="FQ-ViT")

parser.add_argument(
    "model", choices=["default", "vit_h", "vit_l", "vit_b"], help="model"
)
parser.add_argument("checkpoint", help="model checkpoint path")
parser.add_argument("dataset_dir", metavar="DIR", help="path to dataset")
parser.add_argument("--quant", default=False, action="store_true")
parser.add_argument("--ptf", default=False, action="store_true")
parser.add_argument("--lis", default=False, action="store_true")
parser.add_argument(
    "--quant-method", default="minmax", choices=["minmax", "ema", "omse", "percentile"]
)
parser.add_argument(
    "--calib-batchsize", default=2, type=int, help="batchsize of calibration set"
)
parser.add_argument("--calib-iter", default=10, type=int)
parser.add_argument(
    "--val-batchsize", default=1, type=int, help="batchsize of validation set"
)
parser.add_argument(
    "--num-workers",
    default=0,
    type=int,
    help="number of data loading workers (default: 0)",
)
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--visulize", default=False, action="store_true", help="visulize")


def str2model(name):
    d = {
        "deit_tiny": deit_tiny_patch16_224,
        "deit_small": deit_small_patch16_224,
        "deit_base": deit_base_patch16_224,
        "vit_base": vit_base_patch16_224,
        "vit_large": vit_large_patch16_224,
        "swin_tiny": swin_tiny_patch4_window7_224,
        "swin_small": swin_small_patch4_window7_224,
        "swin_base": swin_base_patch4_window7_224,
        "default": build_sam_vit_h,
        "vit_h": build_sam_vit_h,
        "vit_l": build_sam_vit_l,
        "vit_b": build_sam_vit_b,
    }
    print("Model: %s" % d[name].__name__)
    return d[name]


def seed(seed=0):
    import numpy as np
    import torch

    import os
    import random
    import sys

    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_next_click_torch(prev_seg, gt_semantic_seg):
    mask_threshold = 0

    batch_points = []
    batch_labels = []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        if fn_masks[i, 0, point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False
        # bp = point[1:3:-1].clone().detach().reshape(1,1,2)
        bp = torch.tensor([point[2], point[1]]).reshape(1, 1, 2)
        bl = torch.tensor(
            [
                int(is_positive),
            ]
        ).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = torch.logical_and(
        torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = torch.logical_and(
        torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return intersection / union


def calibrate_forward(sam_model, batch_data):
    images, gt_masks, points = (
        batch_data["images"],
        batch_data["instances"],
        batch_data["points"],
    )
    device = gt_masks.device
    torch.cuda.empty_cache()
    prev_masks = torch.zeros_like(gt_masks)

    low_res_masks = F.interpolate(
        prev_masks.float(),
        size=(prev_masks.shape[-2] // 4, prev_masks.shape[-1] // 4),
        mode="bilinear",
        align_corners=False,
    )
    image_embedding = sam_model.image_encoder(images)  # (B, 256, 64, 64

    click_points = []
    click_labels = []

    for num_click in range(1):
        batch_points, batch_labels = get_next_click_torch(prev_masks, gt_masks)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        click_points.append(points_co)
        click_labels.append(points_la)

        points_multi = torch.cat(click_points, dim=1).to(device)
        labels_multi = torch.cat(click_labels, dim=1).to(device)

        points_input = points_multi
        labels_input = labels_multi

        coords_torch = points_input
        labels_torch = labels_input

        (
            sparse_embeddings,
            dense_embeddings,
        ) = sam_model.prompt_encoder(  # TODO quantize emb
            # points=[coords_torch, labels_torch],
            points=[points_input, labels_input],
            boxes=None,
            # masks=None if num_click == 0 else low_res_masks,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # print(low_res_masks.shape)
        # low_res_masks = torch.sigmoid(low_res_masks)
        prev_masks = F.interpolate(
            low_res_masks,
            size=batch_data["instances"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def main():
    args = parser.parse_args()
    seed(args.seed)
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
    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)
    model = str2model(args.model)(checkpoint=args.checkpoint, cfg=cfg)
    model = model.to(device)

    # Note: Different models have different strategies of data preprocessing.
    # model_type = args.model

    valset = SBDDataset(
        args.dataset_dir,
        split="val",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        # epoch_len=500,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.val_batchsize,
        shuffle=False,
        # drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    # switch to evaluate mode
    model.eval()

    # define loss function (criterion)
    # criterion = nn.CrossEntropyLoss().to(device)

    if args.quant:
        trainset = SBDDataset(
            args.dataset_dir,
            split="train",
            augmentator=val_augmentator,
            min_object_area=80,
            points_sampler=points_sampler,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.calib_batchsize,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        # Get calibration set.
        data_list = []
        for i, data in enumerate(train_loader):
            if i == args.calib_iter:
                break
            data = {k: v.to(device) for k, v in data.items()}
            data_list.append(data)

        print("Calibrating...")
        model.model_open_calibrate()
        with torch.no_grad():
            for i, data in enumerate(data_list):
                if i == len(data_list) - 1:
                    # This is used for OMSE method to
                    # calculate minimum quantization error
                    model.model_open_last_calibrate()
                _ = calibrate_forward(model, data)
        del data_list
        model.model_close_calibrate()
        model.model_quant()

    print("Validating...")
    val_miou1, val_miou5 = validate(args, val_loader, model, device)


@torch.no_grad()
def validate(args, val_loader, model, device):
    batch_time = AverageMeter()
    # losses = AverageMeter()
    click1 = AverageMeter()
    click5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, batch_data in enumerate(val_loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        images, gt_masks, points = (
            batch_data["images"],
            batch_data["instances"],
            batch_data["points"],
        )
        prev_masks = torch.zeros_like(gt_masks).to(device)

        image_embedding = model.image_encoder(images)  # (B, 256, 64, 64

        click_points = []
        click_labels = []

        for num_click in range(5):
            batch_points, batch_labels = get_next_click_torch(prev_masks, gt_masks)

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_multi = torch.cat(click_points, dim=1).to(device)
            labels_multi = torch.cat(click_labels, dim=1).to(device)

            points_input = points_multi
            labels_input = labels_multi

            # points_input = get_points(batch_data['points'].to(device))

            # # points_input = [points_input[0][0][0].unsqueeze(0).unsqueeze(0), points_input[1][0][0].unsqueeze(0).unsqueeze(0)]
            # points_input = [points_input[0][0][0:1].unsqueeze(0), points_input[1][0][0:1].unsqueeze(0)]
            # # print(points_input)
            # coords_torch = torch.as_tensor(points_input[0], dtype=torch.float, device=device)
            # labels_torch = torch.as_tensor(points_input[1], dtype=torch.int, device=device)
            coords_torch = points_input
            labels_torch = labels_input

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                # points=[coords_torch, labels_torch],
                points=[points_input, labels_input],
                boxes=None,
                masks=None if num_click == 0 else low_res_masks,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            # print(low_res_masks.shape)
            # low_res_masks = torch.sigmoid(low_res_masks)
            prev_masks = F.interpolate(
                low_res_masks,
                size=batch_data["instances"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # pred_masks = (prev_masks > 0).dtype(torch.int)
            # print(prev_masks)
            pred_masks = torch.as_tensor(
                (prev_masks > 0), dtype=torch.float, device=device
            )
            # print(pred_masks.unique())
            cur_iou = get_iou(
                batch_data["instances"].to(device), pred_masks, ignore_label=-1
            )
            if num_click == 0:
                first_iou = cur_iou

        # losses.update(loss.data.item(), data.size(0))
        click1.update(first_iou.data.item(), images.size(0))
        click5.update(cur_iou.data.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                # "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Click@1 {click1.val:.3f} ({click1.avg:.3f})\t"
                "Click@5 {click5.val:.3f} ({click5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    click1=click1,
                    click5=click5,
                )
            )
            if args.visulize:
                img = ToPILImage()(batch_data["images"][0])
                mask = ToPILImage()(batch_data["instances"][0])
                img.save(f"saved_imgs/{i}_img.png")
                mask.save(f"saved_imgs/{i}_mask.png")
                pred_mask = ToPILImage()(pred_masks[0])
                pred_mask.save(f"saved_imgs/{i}_pred_mask.png")

                imagedraw = ImageDraw.Draw(img)

                for index in range(len(coords_torch[0])):
                    # print(coords_torch)
                    # print(coords_torch[0][0])
                    if labels_torch[0][index] == 0:
                        imagedraw.point(
                            (coords_torch[0][index][0], coords_torch[0][index][1]),
                            (255, 0, 0),
                        )
                        imagedraw.ellipse(
                            (
                                coords_torch[0][index][0] - 3,
                                coords_torch[0][index][1] - 3,
                                coords_torch[0][index][0] + 3,
                                coords_torch[0][index][1] + 3,
                            ),
                            fill=(255, 0, 0),
                        )
                    else:
                        imagedraw.ellipse(
                            (
                                coords_torch[0][index][0] - 3,
                                coords_torch[0][index][1] - 3,
                                coords_torch[0][index][0] + 3,
                                coords_torch[0][index][1] + 3,
                            ),
                            fill=(0, 0, 255),
                        )
                        imagedraw.point(
                            (coords_torch[0][index][0], coords_torch[0][index][1]),
                            (0, 255, 0),
                        )
                img.save(f"saved_imgs/{i}_points.png")
    val_end_time = time.time()
    print(
        " * Click@1 {click1.avg:.3f} Click@5 {click5.avg:.3f} Time {time:.3f}".format(
            click1=click1, click5=click5, time=val_end_time - val_start_time
        )
    )

    return click1.avg, click5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
