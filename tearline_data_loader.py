import os
import pickle 
from PIL import Image
from torch.utils import data 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import cv2
import numpy as np

# Currently trying to create data loader for COWC images

class Cowc(data.Dataset):
    """
    Dataloader for Cars Overhead with Context
    """
    def __init__(self,
                 root,
                 split,
                 image_width,
                 image_height,
                 transform=None,
                 label_file = None):
        # root folder, split 
        self.split = split
        self.image_width = image_width 
        self.image_height = image_height
        self.transform = transform 
        self.all_image_paths = []
        self.label_file = label_file
        self.root_folder = os.path.join(root, "64x64_15cm_24px-exc_v5-marg-32_expanded/")

        # load labels 
        if label_file is None:
            label_file = os.path.join(self.root_folder, "toronto_"+split+"_label.txt")
        if not os.path.exists(label_file):
            raise ValueError(
                "Label file {:s} does not exist!".format(label_file))
        with open(label_file) as f:
            lines = f.readlines()
        
        # store the file list 
        file_label_list = []
        for line in lines:
            tmp = line.rstrip('\n').split('\t')
            filename = tmp[0]
            label_id = int(tmp[-1])
            file_label_list.append((filename, label_id))
        
        # load 
        self.img_label_list = self._load_dataset(file_label_list)

    def _load_dataset(self, file_label_list):
        cached_filename = os.path.join(self.root_folder, "cached_{:s}.pkl".format(self.split))
        if os.path.exists(cached_filename):
            print("=> Loading from cached file {:s} ...".format(cached_filename))
            try:
                img_label_list = pickle.load(open(cached_filename, "rb"))
            except (RuntimeError, TypeError, NameError):
                print("Can't load cached file. Please remove the file and rebuild the cache!")
        else:
            # load dataset into memory
            print("Loading {:s} set into memory. This might take a while...".format(self.split))
            img_label_list = tuple()
            for filename, label_id in tqdm(file_label_list):
                # using RGB instead of greyscale now
                img = cv2.imread(filename)
                # trying 64x64
                # img = img.resize((32, 32), Image.BILINEAR)
                label = label_id
                img_label_list += ((img, label), )
            pickle.dump(img_label_list, open(cached_filename, "wb"))
        return img_label_list

    # These methods are called by pytorch when training
    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, index):
        # load img and label 
        img, label = self.img_label_list[index]
        # NOTE: unsure about this
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (self.image_width, self.image_height))
        
        # image width and height 
        image_width = self.image_width
        image_height = self.image_height

        # TODO: figure out if COWC supports these feature boxes
        # just set box to the entire image for now
        boxes = [[0, 0, image_width, image_height],]
        # 'bounding box' to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of bounding box 
        area = image_width * image_height
        area = torch.as_tensor(area, dtype=torch.float32)
        # no crowd instances (not sure what this means yet)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        label = torch.as_tensor([label,], dtype=torch.int64)

        # prepare final 'target' dictionary
        target = dict()
        target["boxes"] = boxes 
        target["labels"] = label 
        target["area"] = area 
        target["iscrowd"] = iscrowd 
        image_id = torch.tensor([index])
        target["image_id"] = image_id


        # apply data augmentation
        if self.transform is not None:
            sample = self.transform(image = img,
                                    bboxes = target["boxes"],
                                    labels = label)
            img = sample["image"]
            # target["boxes"] = torch.Tensor(sample["bboxes"])

        return img, target

def create_dataset(root, split, image_width, image_height, transform):
    dataset = Cowc(root, 
         split, 
         image_width, 
         image_height, 
         transform,
    )
    return dataset

def create_loader(dataset, batch_size, split):
    shuffle = True if split=="train" else False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False
    )
    return loader

def collate_fn(batch):
    """
    Handles data loading with varying size tensors
    """
    return tuple(zip(*batch))
