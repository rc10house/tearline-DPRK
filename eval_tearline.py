import os
import time
import argparse 

import torch
from torch._dynamo.skipfiles import check
from torch.utils.checkpoint import checkpoint 
import torchvision.transforms as transforms
from tearline_data_loader import Cowc
from lenet import LeNet, test_model


# main function for testing
def main(args):
    # set up random seed
    torch.manual_seed(0)

    ###############
    # setup model #
    ###############
    model = LeNet()

    # set up tarnsforms to transform the PIL image to tensors
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.456, std=0.224)
    ])

    ################################
    # setup dataset and dataloader #
    ################################
    test_set = Cowc(root="./", split="test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False)

    ######################
    # evaulate the model #
    ######################
    # load from a previous model 
    if not args.load:
        args.load = "./outputs/model_best.pth.tar"
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{:s}'".format(args.load))
        checkpoint = torch.load(args.load)
        # load model weight 
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{:s}' (epoch {:d})".format(args.load, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.load))
        return 

    # evaluation and timing 
    print("Evaluating the model...\n")
    start = time.time()
    # evaluate the loaded model 
    acc = test_model(model, test_loader, epoch-1)
    end = time.time()
    print("Evaluation took {:0.2f} sec".format(end-start))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Image Classification (cars) using Pytorch")
    parser.add_argument("--load", default="", type=str, metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    args = parser.parse_args()
    main(args)
