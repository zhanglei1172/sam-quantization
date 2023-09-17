from data.datasets.sbd import SBDDataset
from dataloaders import *
from albumentations import *
# import albumentations as F
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

from PIL import ImageDraw
from segment_anything.build_sam import sam_model_registry

model_type = {
    'vit_b': './checkpoint/sam_vit_b_01ec64.pth',
    'vit_l': './checkpoint/sam_vit_l_0b3195.pth',
    'vit_h': './checkpoint/sam_vit_h_4b8939.pth',
}


# crop_size = (1024, 1024)
# crop_size = (256, 16)
# point_x_y = [232,11]是对应的
crop_size = (1024, 1024)

train_augmentator = Compose([
    Resize(1024, 1024),
    UniformRandomResize(scale_range=(0.75, 1.25)),
    Flip(),
    RandomRotate90(),
    ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                        rotate_limit=(-3, 3), border_mode=0, p=0.75),
    PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
    RandomCrop(*crop_size),
    RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
    RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
], p=1.0)

val_augmentator = Compose([
    Resize(1024, 1024),
    UniformRandomResize(scale_range=(0.75, 1.25)),
    PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
    RandomCrop(*crop_size)
], p=1.0)

MAX_NUM_POINTS=12

points_sampler = MultiPointSampler(max_num_points=MAX_NUM_POINTS, prob_gamma=0.80,
                                    merge_objects_prob=0.15,
                                    max_num_merged_objects=2)

trainset = SBDDataset(
    '/data/gsz/mmiseg/datasets/SBD_ORI/dataset',
    split='train',
    augmentator=train_augmentator,
    min_object_area=80,
    keep_background_prob=0.01,
    points_sampler=points_sampler,
    samples_scores_path='./assets/sbd_samples_weights.pkl',
    samples_scores_gamma=1.25
)

valset = SBDDataset(
    '/data/gsz/mmiseg/datasets/SBD_ORI/dataset',
    split='val',
    augmentator=val_augmentator,
    min_object_area=80,
    points_sampler=points_sampler,
    epoch_len=500
)

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

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
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
            if point_num < MAX_NUM_POINTS//2 and points[batch_num][point_num][2] != -1:
                points[batch_num][point_num][2] = 1
            elif point_num >= MAX_NUM_POINTS//2 and points[batch_num][point_num][2] != -1:
                points[batch_num][point_num][2] = 0
    
    return [torch.concat([points[:,:,1:2], points[:,:,0:1]], dim=2), points[:,:,2]]

def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union

if __name__ == '__main__':
    
    train_data = DataLoader(
        trainset, 1,
        sampler=get_sampler(trainset, shuffle=True, distributed=False),
        drop_last=True, pin_memory=True,
        num_workers=2
    )

    val_data = DataLoader(
        valset, 1,
        sampler=get_sampler(valset, shuffle=False, distributed=False),
        drop_last=True, pin_memory=True,
        num_workers=2
    )

    device = 'cuda'
    mt = 'vit_b'
    from segment_anything.utils.transforms import ResizeLongestSide
    sam_model = sam_model_registry[mt](checkpoint=model_type[mt]).to(device)
    # transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    all_iou = 0
    for i, batch_data in enumerate(train_data):

        image_embedding = sam_model.image_encoder(batch_data['images'].to(device)) # (B, 256, 64, 64

        points_input = get_points(batch_data['points'].to(device))

        # points_input = [points_input[0][0][0].unsqueeze(0).unsqueeze(0), points_input[1][0][0].unsqueeze(0).unsqueeze(0)]
        # points_input = [points_input[0][0][0:4].unsqueeze(0), points_input[1][0][0:4].unsqueeze(0)]

        coords_torch = torch.as_tensor(points_input[0], dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(points_input[1], dtype=torch.int, device=device)

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=[coords_torch, labels_torch],
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        # print(low_res_masks.shape)
        # low_res_masks = torch.sigmoid(low_res_masks)
        prev_masks = F.interpolate(low_res_masks, size=batch_data['instances'].shape[-2:], mode='bilinear', align_corners=False)
        # pred_masks = (prev_masks > 0).dtype(torch.int)
        # print(prev_masks)
        pred_masks = torch.as_tensor((prev_masks > 0), dtype=torch.float, device=device)
        # print(pred_masks.unique())
        cur_iou = get_iou(batch_data['instances'].to(device), pred_masks, ignore_label=-1)
        all_iou += cur_iou
        
        print(all_iou/(i+1))
        # print(iou_predictions)

        
        # print(batch_data['instances'].unique())
        # # print(batch_data)
        # img = ToPILImage()(batch_data['images'][0])
        # mask = ToPILImage()(batch_data['instances'][0])
        # img.save('img.png')
        # mask.save('mask.png')
        # pred_mask = ToPILImage()(pred_masks[0])
        # pred_mask.save('pred_mask.png')

        # imagedraw = ImageDraw.Draw(img)

        # for index in range(len(coords_torch[0])):
        #     # print(coords_torch)
        #     # print(coords_torch[0][0])
        #     if labels_torch[0][index] == 0:
        #         imagedraw.point((coords_torch[0][index][0],coords_torch[0][index][1]),(255,0,0))
        #     else:
        #         imagedraw.point((coords_torch[0][index][0],coords_torch[0][index][1]),(0,255,0))
        # img.save('points.png')

        # break
        # print(batch_data['points'])
        # print(batch_data['instances'].shape)

        # imagedraw = ImageDraw.Draw(img)
        # for point in batch_data['points'][0]:
        #     if batch_data['instances'][0][0][int(point[0])][int(point[1])] == 0:
        #         imagedraw.point((point[1],point[0]),(255,0,0))
        #     else:
        #         imagedraw.point((point[1],point[0]),(0,255,0))
        # img.save('points.png')
        # break
        ##########################################################
        # device = 'cpu'
        # batch_data = {k: v.to(device) for k, v in batch_data.items()}
        # image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
        # orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

        # prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

        # last_click_indx = None

        # with torch.no_grad():
        #     self.max_num_next_clicks=3
        #     num_iters = random.randint(0, self.max_num_next_clicks)

        #     for click_indx in range(num_iters):
        #         last_click_indx = click_indx

        #         if not validation:
        #             self.net.eval()

        #         if self.click_models is None or click_indx >= len(self.click_models):
        #             eval_model = self.net
        #         else:
        #             eval_model = self.click_models[click_indx]

        #         net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
        #         prev_output = torch.sigmoid(eval_model(net_input, points)['instances'])

        #         points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)

        #         if not validation:
        #             self.net.train()

        #     if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
        #         zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
        #         prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

        # batch_data['points'] = points

        # net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
        # output = self.net(net_input, points)

        # loss = 0.0
        # loss = self.add_loss('instance_loss', loss, losses_logging, validation,
        #                         lambda: (output['instances'], batch_data['instances']))
        # loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
        #                         lambda: (output['instances_aux'], batch_data['instances']))

