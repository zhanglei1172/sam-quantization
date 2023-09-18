import numpy as np
import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from albumentations import *
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from transformers import SamModel, SamProcessor

MAX_NUM_POINTS = 24
points_sampler = MultiPointSampler(
    max_num_points=MAX_NUM_POINTS,
    prob_gamma=0.80,
    merge_objects_prob=0.15,
    max_num_merged_objects=2,
)


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


def get_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def get_points(points):
    for batch_num in range(len(points)):
        for point_num in range(len(points[batch_num])):
            if (
                point_num < MAX_NUM_POINTS // 2
                and points[batch_num][point_num][2] != -1
            ):
                points[batch_num][point_num][2] = 1
            elif (
                point_num >= MAX_NUM_POINTS // 2
                and points[batch_num][point_num][2] != -1
            ):
                points[batch_num][point_num][2] = 0

    return [
        torch.concat([points[:, :, 1:2], points[:, :, 0:1]], dim=2),
        points[:, :, 2],
    ]


def batch_forward(
    sam_model,
    image_embedding,
    gt_masks,
    low_res_masks,
    points=None,
    boxes=None,
    masks=None,
):
    # masks = low_res_masks
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        input_points=points[0].unsqueeze(1),
        input_labels=points[1].unsqueeze(1),
        input_boxes=boxes,
        input_masks=masks,
    )
    low_res_masks, iou_predictions, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
        image_positional_embeddings=sam_model.get_image_wide_positional_embeddings().to(device).repeat(image_embedding.shape[0], 1, 1, 1),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(
        low_res_masks.squeeze(1), size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
    )
    return low_res_masks.squeeze(1), prev_masks

@torch.no_grad()
def interaction(sam_model, batch_data, num_clicks):
    image, gt_mask, points = (
        batch_data["images"],
        batch_data["instances"],
        batch_data["points"],
    )
    # inputs = processor(image, input_points=points, return_tensors="pt")
    orig_image, orig_gt_mask, orig_points = (
        image.clone(),
        gt_mask.clone(),
        points.clone(),
    )
    # image_embedding = sam_model.image_encoder(image)
    image_embedding = sam_model.vision_encoder(
        image
    ).last_hidden_state

    # prev_masks = torch.zeros_like(orig_gt_mask).to(orig_gt_mask.device)
    # low_res_masks = F.interpolate(
    #     prev_masks.float(), size=(args.img_size // 4, args.img_size // 4)
    # )
    low_res_masks = None
    # random_num_clicks = np.random.randint(0, 5)

    for num_click in range(num_clicks):
        last_click_indx = num_click

        points_input = get_points(points)
        low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, orig_gt_mask, low_res_masks, points=points_input)
        # low_res_masks, prev_masks = batch_forward(
            # sam_model, image_embedding, orig_gt_mask, low_res_masks, points=None
        # )
        if visualize:
            ###################################################################
            from torchvision.transforms import ToPILImage

            from PIL import ImageDraw

            img = ToPILImage()(orig_image[0])
            mask = ToPILImage()(orig_gt_mask[0])
            img.save("img.png")
            mask.save("mask.png")

            # print(batch_data['points'])
            # print(batch_data['instances'].shape)
            print(orig_points)

            imagedraw = ImageDraw.Draw(img)
            for point in orig_points[0]: # make point more visible
                if orig_gt_mask[0][0][int(point[0])][int(point[1])] == 0:
                    imagedraw.point((point[1], point[0]), (255, 0, 0))
                    for i in range(1, 5):
                        imagedraw.point((point[1] + i, point[0]), (255, 0, 0))
                        imagedraw.point((point[1] - i, point[0]), (255, 0, 0))
                        imagedraw.point((point[1], point[0] + i), (255, 0, 0))
                        imagedraw.point((point[1], point[0] - i), (255, 0, 0))
                else:
                    imagedraw.point((point[1], point[0]), (0, 255, 0))
                    for i in range(1, 5):
                        imagedraw.point((point[1] + i, point[0]), (0, 255, 0))
                        imagedraw.point((point[1] - i, point[0]), (0, 255, 0))
                        imagedraw.point((point[1], point[0] + i), (0, 255, 0))
                        imagedraw.point((point[1], point[0] - i), (0, 255, 0))

                
            img.save("points.png")
        print(
            f"num_click: {num_click}, iou: {get_iou(orig_gt_mask, (prev_masks>0))}"
        )
        ###################################################################
        points = get_next_points(prev_masks, orig_gt_mask, points, num_click + 1)

    batch_data["points"] = points

    points_input = get_points(points)

    low_res_masks, prev_masks = batch_forward(
        sam_model, image_embedding, orig_gt_mask, low_res_masks, points=points_input
    )

    return prev_masks


def eval_val():
    # return 0
    model.eval()
    sam_model = model

    # dice_list = []
    val_data = DataLoader(
        valset,
        args.batch_size,
        sampler=get_sampler(valset, shuffle=False, distributed=False),
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    all_iou = 0
    for step, (batch_data) in enumerate(val_data):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        images, gt_masks, points = (
            batch_data["images"],
            batch_data["instances"],
            batch_data["points"],
        )

        prev_masks = interaction(sam_model, batch_data, num_clicks=5)

        iou = get_iou(gt_masks, prev_masks>0)

        all_iou += iou
        # print('cur_iou: ', iou)
        print('mean iou: ', all_iou/(step+1))

    return all_iou


if __name__ == "__main__":
    device = "cuda"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
    )
    parser.add_argument("dataset_dir", type=str, help="Where to evaluate.")
    args = parser.parse_args()
    model = SamModel.from_pretrained(
        args.model_path,
        proxies={
            "http": "socks5://127.0.0.1:7890",
            "https": "socks5://127.0.0.1:7890",
            "scoks5": "socks5://127.0.0.1:7890",
        },
    ).to(device)
    processor = SamProcessor.from_pretrained(
        args.model_path,
        proxies={
            "http": "socks5://127.0.0.1:7890",
            "https": "socks5://127.0.0.1:7890",
            "scoks5": "socks5://127.0.0.1:7890",
        },
    )

    crop_size = (1024, 1024)
    val_augmentator = Compose(
        [
            Resize(1024, 1024),
            # UniformRandomResize(scale_range=(0.75, 1.25)),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
        ],
        p=1.0,
    )

    valset = SBDDataset(
        args.dataset_dir,
        split="val",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        # epoch_len=500
    )

    model.eval()
    visualize = False
    eval_val()
