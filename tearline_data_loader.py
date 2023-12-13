import os
import pickle 
from PIL import Image 
from torch.utils import data
from tqdm import tqdm

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
                 transform=None):
        assert split in ["train", "val", "test"]
        # root folder, split 
        self.root_folder = os.path.join(root, "64x64_15cm_24px-exc_v5-marg-32_expanded")
        self.split = split 
        self.transform = transform 
        self.n_classes = num_classes 

        # load labels 
        if label_file is None:
            label_file = os.path.join(self.root_folder, "toronto_test_label.txt")
        if not os.path.exists(label_file):
            raise ValueError(
                "Label file {:s} does not exist!".format(label_file))
        with open(label_file) as f:
            lines = f.readlines()
        
        # store the file list 
        file_label_list = []
        for line in lines:
            tmp = line.rstrip('\n').split(' ')
            filename = tmp[0]
            label_id = int(tmp[-1])
            file_label_list.append((filename, label_id))
        
        # load 
        self.img_label_list = self._load_dataset(file_label_list)

    def _load_dataset(self, file_label_list):
        # load dataset into memory
        print("Loading {:s} set into memroy. This might take a while...".format(self.split))
        img_label_list = tuple()
        for filename, label_id in tqdm(file_label_list):
            img = Image.open(filename).convert('RGB')
            # Images are currently 64x64 
            label = label_id
            img_label_list += ((img, label), )
        return img_label_list

