import numpy as np
import cv2
import torch

# import albumentations as F
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from segment_anything.build_sam import sam_model_registry

import pandas as pd

try:
    import tensorrt as trt
    import trt_infer
except:
    pass

from albumentations import *
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from PIL import ImageDraw

try:
    from ppq import convert_any_to_numpy, convert_any_to_torch_tensor
    from ppq.utils.TensorRTUtil import trt
except:
    pass
from typing import List

# crop_size = (1024, 1024)
# crop_size = (256, 16)
# point_x_y = [232,11]是对应的
crop_size = (1024, 1024)

train_augmentator = Compose(
    [
        Resize(1024, 1024),
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0, rotate_limit=(-3, 3), border_mode=0, p=0.75
        ),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
        ),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
    ],
    p=1.0,
)

val_augmentator = Compose(
    [
        Resize(1024, 1024),
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
    ],
    p=1.0,
)

MAX_NUM_POINTS = 24

points_sampler = MultiPointSampler(
    max_num_points=MAX_NUM_POINTS,
    prob_gamma=0.80,
    merge_objects_prob=0.15,
    max_num_merged_objects=2,
)


def get_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def get_next_points(pred, gt, points, click_indx, pred_thresh=0):
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
    valid_points_batch = []
    for batch_num in range(len(points)):
        valid_points = []
        for point_num in range(len(points[batch_num])):
            if (
                point_num < MAX_NUM_POINTS // 2
                and points[batch_num][point_num][2] != -1
            ):
                points[batch_num][point_num][2] = 1
                x = points[batch_num][point_num][1]
                y = points[batch_num][point_num][0]
                ind = points[batch_num][point_num][2]
                valid_points.append([x, y, ind])
            elif (
                point_num >= MAX_NUM_POINTS // 2
                and points[batch_num][point_num][2] != -1
            ):
                points[batch_num][point_num][2] = 0
                # valid_points.append(points[batch_num][point_num])
                x = points[batch_num][point_num][1]
                y = points[batch_num][point_num][0]
                ind = points[batch_num][point_num][2]
                valid_points.append([x, y, ind])
        valid_points_batch.append(valid_points)
    all_valid_points = torch.tensor(valid_points_batch).to(points.device)

    # return [torch.concat([points[:,:,1:2], points[:,:,0:1]], dim=2), points[:,:,2]]
    return [all_valid_points[:, :, :2], all_valid_points[:, :, 2]]


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


def infer_trt(engine, samples: List[np.ndarray]) -> List[np.ndarray]:
    """Run a tensorrt model with given samples"""

    results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        for sample in samples:
            inputs[0].host = convert_any_to_numpy(sample)
            output = trt_infer.do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
                batch_size=1,
            )[0]
            results.append(
                convert_any_to_torch_tensor(output).reshape([-1, 256, 64, 64])
            )
    return results


@torch.no_grad()
def main(sam_model, val_data, args, device):
    dtype = next(sam_model.parameters()).dtype
    all_iou = 0
    iou_list = []
    backend = None
    if hasattr(args, "trt_engine"):
        backend = "ORT"
        if args.trt_engine.endswith(".onnx"):
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                args.trt_engine, providers=["CUDAExecutionProvider"]
            )
            output_name = sess.get_outputs()[0].name
        else:
            backend = "TRT"
            logger = trt.Logger(trt.Logger.ERROR)
            with open(args.trt_engine, "rb") as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
    for step, batch_data in enumerate(val_data):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        images, gt_masks, points = (
            batch_data["images"].to(dtype),
            batch_data["instances"].to(dtype),
            batch_data["points"].to(dtype),
        )
        prev_masks = torch.zeros_like(gt_masks).to(device)

        if backend == "ORT":
            ort_outs = sess.run(
                output_names=[output_name],
                input_feed={"input.1": images.cpu().numpy()},
            )

            image_embedding = (
                convert_any_to_torch_tensor(ort_outs).squeeze().unsqueeze(0).to(device)
            )
            # print(image_embedding.shape)
        elif backend == "TRT":
            trt_outputs = infer_trt(
                engine,
                samples=[convert_any_to_numpy(sample) for sample in images.cpu()],
            )
            image_embedding = torch.cat(trt_outputs).to(device)
        else:
            image_embedding = sam_model.image_encoder(images)  # (B, 256, 64, 64

        click_points = []
        click_labels = []

        for num_click in range(5):
            batch_points, batch_labels = get_next_click_torch(prev_masks, gt_masks)

            points_co = torch.cat(batch_points, dim=0).to(device).to(dtype)
            points_la = torch.cat(batch_labels, dim=0).to(device).to(dtype)

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

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                # points=[coords_torch, labels_torch],
                points=[points_input, labels_input],
                boxes=None,
                masks=None if num_click == 0 else low_res_masks,
            )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe().to(dtype),  # (1, 256, 64, 64)
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
            print("cur_iou: ", cur_iou)
        all_iou += cur_iou * args.batch_size
        iou_list.append(cur_iou.item())
        print("mean iou: ", all_iou / ((step + 1) * args.batch_size))
        # print(iou_predictions)

        # print(batch_data['instances'].unique())
        # print(batch_data)
    #     img = ToPILImage()(batch_data["images"][0])
    #     mask = ToPILImage()(batch_data["instances"][0])
    #     img.save(f"saved_imgs/{step}_img.png")
    #     mask.save(f"saved_imgs/{step}_mask.png")
    #     pred_mask = ToPILImage()(pred_masks[0])
    #     pred_mask.save(f"saved_imgs/{step}_pred_mask.png")

    #     imagedraw = ImageDraw.Draw(img)

    #     for index in range(len(coords_torch[0])):
    #         # print(coords_torch)
    #         # print(coords_torch[0][0])
    #         if labels_torch[0][index] == 0:
    #             imagedraw.point(
    #                 (coords_torch[0][index][0], coords_torch[0][index][1]), (255, 0, 0)
    #             )
    #             imagedraw.ellipse(
    #                 (
    #                     coords_torch[0][index][0] - 3,
    #                     coords_torch[0][index][1] - 3,
    #                     coords_torch[0][index][0] + 3,
    #                     coords_torch[0][index][1] + 3,
    #                 ),
    #                 fill=(255, 0, 0),
    #             )
    #         else:
    #             imagedraw.ellipse(
    #                 (
    #                     coords_torch[0][index][0] - 3,
    #                     coords_torch[0][index][1] - 3,
    #                     coords_torch[0][index][0] + 3,
    #                     coords_torch[0][index][1] + 3,
    #                 ),
    #                 fill=(0, 0, 255),
    #             )
    #             imagedraw.point(
    #                 (coords_torch[0][index][0], coords_torch[0][index][1]), (0, 255, 0)
    #             )
    #     img.save(f"saved_imgs/{step}_points.png")
    # pd.DataFrame(iou_list).to_csv(
    #     f"saved_csvs/{args.model_path.split('/')[-1].split('.')[0]}.csv"
    # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
    )
    parser.add_argument("dataset_dir", type=str, help="Where to evaluate.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    # parser.add_argument(
    #     "--img_size",
    #     type=int,
    #     default=1024,
    # )
    args = parser.parse_args()
    model_type = {
        # 'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
        # 'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
        "vit_h": args.model_path,
    }
    # trainset = SBDDataset(
    #     args.dataset_dir,
    #     split='train',
    #     augmentator=train_augmentator,
    #     min_object_area=80,
    #     keep_background_prob=0.01,
    #     points_sampler=points_sampler,
    #     samples_scores_path='./assets/sbd_samples_weights.pkl',
    #     samples_scores_gamma=1.25
    # )

    valset = SBDDataset(
        args.dataset_dir,
        split="val",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500,
    )

    # train_data = DataLoader(
    #     trainset, args.batch_size,
    #     sampler=get_sampler(trainset, shuffle=True, distributed=False),
    #     drop_last=True, pin_memory=True,
    #     num_workers=args.num_workers
    # )

    val_data = DataLoader(
        valset,
        args.batch_size,
        # sampler=get_sampler(valset, shuffle=False, distributed=False),
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    device = "cuda"
    # device='cpu'
    mt = "vit_h"
    from segment_anything.utils.transforms import ResizeLongestSide

    sam_model = sam_model_registry[mt](checkpoint=model_type[mt]).to(device)
    sam_model.eval()
    # transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    main(sam_model, val_data, args)
