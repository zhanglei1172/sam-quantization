# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import cv2
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import monai
# import SimpleITK as sitk
# import torchio as tio
from Losses import NormalizedFocalLossSigmoid, MaskedNormalizedFocalLossSigmoid, SigmoidBinaryCrossEntropyLoss, CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler


from contextlib import nullcontext


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='duiqi')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# parser.add_argument('--data_dir', type=str, default='/cpfs01/user/guosizheng/SAM3D/dataset/MSD01_BrainTumor_flair')
parser.add_argument('--log_out_dir', type=str, default='./work_dir/log.log')
# parser.add_argument('--load_weights_only')

# train
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[2,0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)

parser.add_argument('--resume', action='store_true', default=False)

# lr_scheduler
# parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
# parser.add_argument('--warmup_epochs', type=int, default=0)
# parser.add_argument('--warmup_factor', type=float, default=1e-6)
# parser.add_argument('--warmup_method', type=str, default='linear')
# parser.add_argument('--final_lr', type=float, default=1e-6)
# parser.add_argument('--power', type=float, default=1.0)
# parser.add_argument('--max_iters', type=int, default=1000)
# parser.add_argument('--min_lr', type=float, default=1e-6)
# parser.add_argument('--step_size', type=int, default=5)
# parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 160, 190])
# parser.add_argument('--step_size', type=list, default=[20, 40])
parser.add_argument('--gamma', type=float, default=0.1)


parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.1)

# parser.add_argument('-lwl', '--layer_wize_lr', action='store_true', default=False)
# parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()


# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
if args.multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %% set up logger

###################################### Logging ######################################
import datetime
import logging
logger = logging.getLogger(__name__)

LOG_OUT_DIR = join(args.work_dir, args.task_name)

###################################### Logging ######################################

# args.multi_gpu = True

# %% set up click methods
###################################### Click type ######################################
# from utils import load_sam_2d_weight, get_next_click3D_torch_2
# click_methods = {
#     'random': get_next_click3D_torch_2,
#     'choose': None,
# }
###################################### Click type ######################################


# %% set up model for fine-tuning 
device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    # sam_model.train()
    return sam_model

# %% set up dataloader
################################################## Data ##################################################
from utils.dataloader import trainset, valset, get_next_points

def get_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)


def get_dataloaders(args):
    train_data = DataLoader(
        trainset, args.batch_size,
        sampler=get_sampler(trainset, shuffle=True, distributed=args.multi_gpu),
        drop_last=True, pin_memory=True,
        num_workers=args.num_workers
    )

    # train_data = DataLoader(
    #     dataset=trainset,
    #     sampler=DistributedSampler(trainset),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )

    val_data = DataLoader(
        valset, args.batch_size,
        sampler=get_sampler(valset, shuffle=False, distributed=args.multi_gpu),
        drop_last=True, pin_memory=True,
        num_workers=args.num_workers
    )

    return train_data # , val_data
################################################## Data ##################################################


# %% set up trainer
########################################## Trainer ##########################################

class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        
    def set_loss_fn(self):
        # self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.nfl = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2.0)
        self.sbce = SigmoidBinaryCrossEntropyLoss()

    
    def set_optimizer(self):
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            # {'params': sam_model.image_encoder.parameters()}, 
            {'params': sam_model.prompt_encoder.parameters()},
            # {'params': sam_model.mask_decoder.parameters()},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            # TODO: add args for coswarm
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            # TODO: add other lr_scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))
    
    def batch_forward(self, sam_model, image_embedding, gt_masks, low_res_masks, points=None, boxes=None, masks=None):
        # masks = low_res_masks
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, points):
        for batch_num in range(len(points)):
            for point_num in range(len(points[batch_num])):
                if point_num < 24 and points[batch_num][point_num][2] != -1:
                    points[batch_num][point_num][2] = 1
                elif point_num >= 24 and points[batch_num][point_num][2] != -1:
                    points[batch_num][point_num][2] = 0
        
        return [points[:,:,:2], points[:,:,2]]

    def interaction(self, sam_model, batch_data, num_clicks):
        image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
        orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()
        image_embedding = sam_model.image_encoder(image)

        prev_masks = torch.zeros_like(orig_gt_mask).to(orig_gt_mask.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4))
        random_num_clicks = np.random.randint(0, 5)
        
        with torch.no_grad():
            for num_click in range(random_num_clicks):
                last_click_indx = num_click

                points_input = self.get_points(points)
                # low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, orig_gt_mask, low_res_masks, points=points_input)
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, orig_gt_mask, low_res_masks, points=None)

###################################################################
                from torchvision.transforms import ToPILImage
                from PIL import ImageDraw

                img = ToPILImage()(orig_image[0])
                mask = ToPILImage()(orig_gt_mask[0])
                img.save('img.png')
                mask.save('mask.png')

                # print(batch_data['points'])
                # print(batch_data['instances'].shape)
                print(orig_points)

                imagedraw = ImageDraw.Draw(img)
                for point in orig_points[0]:
                    if orig_gt_mask[0][0][int(point[0])][int(point[1])] == 0:
                        imagedraw.point((point[1],point[0]),(255,0,0))
                    else:
                        imagedraw.point((point[1],point[0]),(0,255,0))
                img.save('points.png')
                print(f'num_click: {num_click}, iou: {self.get_iou(orig_gt_mask, prev_masks)}')
###################################################################
                points = get_next_points(prev_masks, orig_gt_mask,points, num_click+1)

        batch_data['points'] = points

        points_input = self.get_points(points)

        low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, orig_gt_mask, low_res_masks, points=points_input)

        return prev_masks

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        for n, value in sam_model.named_parameters():
            if "prompt_encoder" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()

        step_loss = 0
        for step, (batch_data) in enumerate(tbar):

            # loss = self.bf(batch_data)
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            images, gt_masks, points = batch_data['images'], batch_data['instances'], batch_data['points']

            with amp.autocast():
                # image_embedding = sam_model.image_encoder(images)

                prev_masks = self.interaction(sam_model, batch_data, num_clicks=11)                
                loss = 20 * self.sbce(prev_masks, gt_masks) + self.nfl(prev_masks, gt_masks)  #  + 20 * self.focal_loss(prev_masks, gt3D)
                loss = torch.mean(loss)

            # import IPython
            # IPython.embed()
            cur_loss = loss.item()
            epoch_loss += cur_loss

            self.scaler.scale(loss).backward()    
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print_loss = step_loss / self.args.accumulation_steps
                    step_loss = 0
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, IoU: {self.get_iou(gt_masks, prev_masks)}')
                else:
                    step_loss += cur_loss
        epoch_loss /= step 
        return epoch_loss

    # TODO: add evaluation function
    def eval_epoch(self, epoch, num_clicks):
        # return 0
        self.model.eval()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        # dice_list = []
        val_data = DataLoader(
            valset, args.batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=args.multi_gpu),
            drop_last=True, pin_memory=True,
            num_workers=args.num_workers
        )
        all_iou = 0
        for step, (batch_data) in enumerate(val_data):
            
            
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            images, gt_masks, points = batch_data['images'], batch_data['instances'], batch_data['points']
 
            prev_masks = self.interaction(sam_model, batch_data, num_clicks=11) 

            iou = self.get_iou(gt_masks, prev_masks)

            print(iou)
            all_iou += iou

        all_iou /= step    

        return all_iou

    def get_iou(self, gt_mask, pred_mask, ignore_label=-1):
        ignore_gt_mask_inv = gt_mask != ignore_label
        obj_gt_mask = gt_mask == 1

        intersection = torch.logical_and(torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
        union = torch.logical_and(torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

        return intersection / union

    def plot_result(self, plot_data, description, save_name):

        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        
        self.scaler = amp.GradScaler()

        ############################## 一个epoch的训练过程开始 #############################################
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

            num_clicks = np.random.randint(1, 21)
            epoch_loss = self.train_epoch(epoch, num_clicks)
            eval_iou = self.eval_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            ##################################### 保存权重和loss #########################################
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                # self.dices.append(epoch_dice)
                self.ious.append(eval_iou)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                # print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                print(f'CURRENT LR: {self.lr_scheduler.get_last_lr()}')
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}, IoU: {eval_iou}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save epoch checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(
                        epoch, 
                        state_dict, 
                        describe=f'epoch_{epoch+1}'
                    )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                # if epoch_dice > self.best_dice: 
                #     self.best_dice = epoch_dice
                #     self.save_checkpoint(
                #         epoch,
                #         state_dict,
                #         describe='dice_best'
                #     )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                # self.plot_result(self.ious, 'IoU', 'iou')
            ##################################### 保存权重和loss #########################################
        ########################### 一个epoch的训练过程结束 #############################################
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        # logger.info(f'model : {self.model}')
        # logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

########################################## Trainer ##########################################
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
 
# def main():
#     rank = torch.distributed.get_rank()
#     # 问题完美解决！
#     init_seeds(1 + rank)

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            # args.multi_gpu = False
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            # args.multi_gpu = True
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
