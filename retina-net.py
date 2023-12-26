# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms
# Pretrained weights for fine-tuning
from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights 
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class RetinaNet(nn.Module):
    def __init__(self, num_classes=4):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        num_anchors = model.head.classification_head.num_anchors
        # These are currently copied from: https://debuggercafe.com/train-pytorch-retinanet-on-custom-dataset/
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        return model



def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
         for input, target in test_loader:
            output, _ = model(input)
            # pred = output.max(1, keepdim=True)[1]
            _, pred = torch.max(output.data, 1)
            classes = ["not car", "other", "pickup", "sedan"]
            # print("Predicted: " + classes[pred[4]])
            # print("pred: " + str(pred) + " target: " + str(target))
            # print(pred.eq(target.view_as(pred)).sum().item())
            # pred.eq
            correct += pred.eq(target.view_as(pred)).sum().item()
            # exit()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
