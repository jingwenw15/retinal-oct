"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Net(nn.Module):
    """
    Define the Net for the VGG16 model that we will perform transfer learning on. 
    https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", device)

        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = self.vgg.classifier[6].in_features

        # freeze layers 
        if params.freeze == "all": 
            for param in self.vgg.parameters():
                param.requires_grad = False 
        elif params.freeze == "some": 
            for name, param in self.vgg.named_parameters():
                if "classifier" not in name: 
                    param.requires_grad = False

        # replace FC layer with our layer 
        self.vgg.classifier[6] = nn.Linear(in_features=in_features, out_features=4, device=device)
        self.vgg = self.vgg.to(device)

        

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
        return self.vgg(s)


def ce_loss(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 4 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return F.cross_entropy(outputs, labels, reduction='mean')


def accuracy(outputs, labels, split=None, images_name=None, fd=None):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 4 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    if split: 
        for o, l, filename in zip(outputs, labels, images_name):
            fd.write(filename + ',' + str(o) + ',' + str(l) + '\n')

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

