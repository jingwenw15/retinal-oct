"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Current device:", device)

        self.resnet = models.resnet18(weights='IMAGENET1K_V1') # TODO: later on, don't use pretrained weights
        in_features = self.resnet.fc.in_features
        self.resnet.layer4 = nn.Sequential(nn.Identity())
        
        # freeze all layers except last 
        # for param in self.resnet.parameters():
        #     param.requires_grad = False 

        # replace FC layer with our layer 
        self.resnet.fc = nn.Linear(in_features=256, out_features=4, device=device)
        # print(self.resnet)
        self.resnet = self.resnet.to(device)

        

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        return self.resnet(s)


def loss_fn(outputs, labels):
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
    # num_examples = outputs.size()[0]
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples
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
    # could add more metrics such as accuracy for each token type
}

r = Net({})