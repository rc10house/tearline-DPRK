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
        self.root_folder = os.path.join(root, "64x64_15cm_24px-exc_v5-marg-32_expanded/")
        self.split = split 
        self.transform = transform 
        self.n_classes = num_classes 

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
                # img = Image.open(filename).convert('RGB')
                # trying greyscale
                img = Image.open(filename).convert("L")
                # trying 32x32
                img = img.resize((32, 32), Image.BILINEAR)
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
        
        # apply data augmentation
        if self.transform is not None:
            img = self.transform(img)
        return img, label 
