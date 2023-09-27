import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

import glob
import json
import os
from PIL import Image


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox
    
class SAMDataset(Dataset):
  def __init__(self, dir_path, processor):
    self.dataset = glob.glob(os.path.join(dir_path, "*.jpg"))
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = Image.open(item)
    ground_truth_mask = np.array(mask_utils.decode(json.load(open(f'{item[:-3]}json'))['annotations'][0]['segmentation']))

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    # inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask).unsqueeze(0)
    inputs["original_image"] = torch.tensor(np.array(image)).unsqueeze(0)

    return inputs

class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)
    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        target = []

        for m in masks:
            # decode masks from COCO RLE format
            target.append(mask_utils.decode(m['segmentation'])) 
        target = np.stack(target)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.imgs)

def get_SA1B(path, nsamples, seed):
    from transformers import SamProcessor

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base",proxies={'http': 'socks5://127.0.0.1:7890', 'https': 'socks5://127.0.0.1:7890', "scoks5":"socks5://127.0.0.1:7890"})

    traindata = SAMDataset(dir_path=path, processor=processor)

    import random

    # random.seed(seed)
    trainloader = []
    testloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata))
        trainloader.append(traindata[i])
    for _ in range(nsamples):
        i = random.randint(0, len(traindata))
        testloader.append(traindata[i])
    return trainloader, testloader

def get_loaders(
    path, nsamples=128, seed=0):
    return get_SA1B(path, nsamples, seed)
