import os 
from PIL import Image 
from torch.utils import data

# Currently trying to create data loader for COWC images

class Cowc(data.Dataset):
    """
    Dataloader for Cars Overhead with Context
    """
    def __init__(self,
                 root,
                 label_file=None,
                 num_classes=1,
                 split="train",
                 transform=None)
    assert split in ["train", "val", "test"]

