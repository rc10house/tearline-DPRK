import os
import time 
import argparse 

import torch
from torch.utils.checkpoint import checkpoint 
import torchvision.transforms as transforms
from tearline_data_loader import Cowc
from lenet import LeNet, test_model
from tqdm import tqdm
from PIL import Image, ImageDraw


class Target():
    """Dataloader for target data"""
    def __init__(self,
                 path,
                 transform=None):
        self.root_folder = os.path.join("./", path)
        self.transform = transform
        self.path = path
        # load data
        if not args.path:
            print("Please specify a path to images with --path")
            return

        print("Loading {:s} into memory...".format(args.path))
        files = os.listdir(args.path)
        image_list = tuple() 
        for f in tqdm(files):
            img = Image.open(os.path.join(self.root_folder, f)).convert("L")
            img = img.resize((32, 32), Image.BILINEAR)
            image_list += ((img, -1), )
        
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # load img and label 
        img, label = self.image_list[index]
        
        # apply data augmentation
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stitch(results, path):

    root = os.path.join("./", path)
    files = os.listdir(root)  

    output = os.path.join("./", "results")

    for i, f in enumerate(files):
        img_path = os.path.join(root, f)
        img = Image.open(img_path).convert("L")
        
        new_img = Image.new("RGB", img.size, color=255)
        new_img.paste(im=img, box=(0,0))
        
        draw = ImageDraw.Draw(new_img)
        if (results[i] == 3.):  
            # draw red cross over image
            draw.line((0,0) + img.size, fill="red")
            draw.line((0, img.size[1], img.size[0], 0), fill="red")
            new_img.save(os.path.join(output, f.replace(".jpeg", "_processed_red.jpeg")))
        else:
            new_img.save(os.path.join(output, f.replace(".jpeg", "_processed.jpeg")))

def main(args):
    # set up random seed 
    torch.manual_seed(0)

    model = LeNet()
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.456, std=0.224)
    ])
    target = Target(args.path, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(
        target, batch_size=32, shuffle=False)

    # run model 
    if not args.load:
        args.load = "./outputs/model_best.pth.tar"
    if os.path.isfile(args.load):
        print("=> loading model '{:s}'".format(args.load))
        checkpoint = torch.load(args.load)
        # load model weight
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        print("=> loaded model '{:s}' (epoch {:d})".format(args.load, checkpoint['epoch']))
    else:
        print("=> no model found at '{}'".format(args.load))
        return

    # running and timing
    print("Running the model...\n")
    start = time.time()
    model.eval()
    cars = 0 

    results = torch.empty(1)

    with torch.no_grad():
        for input, _ in data_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1] 
            # print(pred)
            classes = ["not car", "other", "pickup", "sedan"]
            for p in pred:
                results = torch.cat((results, p))
                if classes[p] == "sedan": cars += 1
        print("cars: " + str(cars))

    stitch(results, args.path)

    end = time.time() 
    print("Model took {:0.2f} sec".format(end-start))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Image Classification (cars) using Pytorch")
    parser.add_argument("--path", default="", type=str, metavar="PATH",
                        help="path to directory of 64x64 images to run model on")
    parser.add_argument("--load", default="", type=str, metavar="PATH",
                        help="path to saved model")
    args = parser.parse_args()
    main(args)
