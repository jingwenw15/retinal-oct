"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Net(nn.Module):
    """
    Define a Net for a smaller version of ResNet (remove certain layers from ResNet).
    http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", device)

        self.resnet = models.resnet18(weights='IMAGENET1K_V1') # TODO: later on, don't use pretrained weights
        in_features = self.resnet.fc.in_features
        
        # NOTE: Citation = https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/18
        # self.resnet.layer1 = nn.Sequential(nn.Identity()) // bad results here 
        self.resnet.layer2 = nn.Sequential(nn.Identity()) # take out layer 2
        self.resnet.layer3 = nn.Sequential(nn.Identity()) # take out layer 3 
        self.resnet.layer4 = nn.Sequential(nn.Identity()) # take out layer 4 
        
        # freeze all layers except last 
        # for param in self.resnet.parameters():
        #     param.requires_grad = False 

        # replace FC layer with our layer 
        self.resnet.fc = nn.Linear(in_features=64, out_features=4, device=device)
        print(self.resnet)
        self.resnet = self.resnet.to(device)

        

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 4 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        NOTE: Citation/Reference = https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        return self.resnet(s)


def ce_loss(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return F.cross_entropy(outputs, labels, reduction='mean')


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 4 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

# NOTE: these functions say "acc" but I actually mean "recall"
def cnv_acc(outputs, labels): 
    outputs = np.argmax(outputs, axis=1) 
    labeled_cnv = (labels == 0)
    return np.sum(outputs[labeled_cnv] == 0)/float(np.sum(labeled_cnv))    

def dme_acc(outputs, labels): 
    outputs = np.argmax(outputs, axis=1) 
    labeled_dme = (labels == 1)
    return np.sum(outputs[labeled_dme] == 1)/float(np.sum(labeled_dme))    

def drusen_acc(outputs, labels): 
    outputs = np.argmax(outputs, axis=1) 
    labeled_drusen = (labels == 2)
    return np.sum(outputs[labeled_drusen] == 2)/float(np.sum(labeled_drusen))    

def normal_acc(outputs, labels): 
    outputs = np.argmax(outputs, axis=1) 
    labeled_normal = (labels == 3)
    return np.sum(outputs[labeled_normal] == 3)/float(np.sum(labeled_normal))    


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'cnv': cnv_acc,
    'dme': dme_acc,
    'drusen': drusen_acc,
    'normal': normal_acc
}

