import torch.profiler
from torch.utils.data import DataLoader, Dataset

from segment_anything.build_sam import sam_model_registry

from albumentations import *
from data.datasets.sbd import SBDDataset
from data.points_sampler import MultiPointSampler
from data.transforms import UniformRandomResize
from script.evaluation2 import get_next_click_torch

model_type = {
    # 'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
    # 'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
}
device = "cuda"
mt = "vit_h"
model = sam_model_registry[mt](checkpoint=model_type[mt]).to(device)

# model = get_llama(args.model)
model.eval()

MAX_NUM_POINTS = 24
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
points_sampler = MultiPointSampler(
    max_num_points=MAX_NUM_POINTS,
    prob_gamma=0.80,
    merge_objects_prob=0.15,
    max_num_merged_objects=2,
)
valset = SBDDataset(
    "/data/seg/sbd/benchmark_RELEASE/dataset",
    split="val",
    augmentator=val_augmentator,
    min_object_area=80,
    points_sampler=points_sampler,
    epoch_len=10,
)
val_data = DataLoader(
    valset,
    1,
    # sampler=get_sampler(valset, shuffle=False, distributed=False),
    drop_last=True,
    pin_memory=True,
    num_workers=0,
)


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        dir_name="runs/",
    ),
    with_stack=True,
) as profiler:
    with torch.no_grad():
        for step, batch_data in enumerate(val_data):
            all_iou = 0
            iou_list = []

            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            images, gt_masks, points = (
                batch_data["images"].to(device),
                batch_data["instances"].to(device),
                batch_data["points"].to(device),
            )
            prev_masks = torch.zeros_like(gt_masks).to(device)

            image_embedding = model.image_encoder(images)  # (B, 256, 64, 64

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
            profiler.step()
