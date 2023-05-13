# adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def get_resnet_model(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device:", device)

    resnet_model = models.resnet18(weights='IMAGENET1K_V1')
    in_features = resnet_model.fc.in_features

    # freeze all layers except last 
    for param in resnet_model.parameters():
        param.requires_grad = False 

    # replace FC layer with our layer 
    resnet_model.fc = nn.Linear(in_features=in_features, out_features=4, device=device)
    resnet_model = resnet_model.to(device)

    return resnet_model 

