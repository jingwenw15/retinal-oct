"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Net(nn.Module):
    """
    Define a Net which will contain the Mobilenet model that we will perform learning on.
    https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.mobilenet_v2    """

    def __init__(self, params, use_pretrained=True):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", device)

        if use_pretrained: 
            self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        else: 
            self.mobilenet = models.mobilenet_v2()
        in_features = self.mobilenet.classifier[1].in_features

        # replace FC layer with our layer 
        self.mobilenet.classifier[1] = nn.Linear(in_features=in_features, out_features=4, device=device)
        self.mobilenet = self.mobilenet.to(device)

        

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 4 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        NOTE: Citation/Reference = https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        return self.mobilenet(s)


def ce_loss(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 4 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    """
    return F.cross_entropy(outputs, labels, reduction='mean')

def mse_loss(outputs, labels): 
    return F.mse_loss(outputs, labels, reduction='mean')

def distill_loss_fn(outputs, labels, t=4): 
    # use softmax with temperature t 
    # reference: https://josehoras.github.io/knowledge-distillation/
    student_weights = F.softmax(outputs / t, dim=1)
    teacher_weights = F.softmax(labels / t, dim=1)
    return F.mse_loss(student_weights, teacher_weights, reduction='mean')
# TODO: match outputs of any layer in general after both layers (after ReLU layer) take middle layers, proportionately 

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
