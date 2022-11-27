import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
 
from torch.utils.data import  DataLoader
from torchvision import models
 
import torchvision.transforms as transforms
import torchvision.datasets as dataset
 
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from model import Hybrid_ViT, MobileNet_AVG, EfficientNet

####image load###
img=cv.imread("/home/image-retrieval/ndir_simulated/test2.jpg")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img=np.array(img)
img=transform(img)
img=img.unsqueeze(0)
#print(img.size())
#######################

#####model load##############
#model = models.resnet18(pretrained=True)
model = MobileNet_AVG()
# study
#for name, param in model.named_parameters():
#    if ((param.ndimension() == 4)): #conv2는 4차원
#        print("conv in name",name)

activations = dict() 
layers = dict()

# hook()
def forward_hook(layer_name:str):
    def hook_fn(module, input, output):
        activations[layer_name] = output.view(output.size(0),-1).detach()
        #print(activations)
    return hook_fn


# forward hook
for name, layer in model.named_modules():
    # "conv.3" refers to output of any inverted residual block,
    # which passed batch norm layer, except 1st, 2nd layer(inverted block).
    print(name)
    #print(layer)
    if "conv.3" in name:
        layers[name] = layer
        print(layer.register_forward_hook(forward_hook(name)))
        
#print("========BEFORE FORWARD(begin)=========")
#print(activations) # (N, C, H, W) -> (N, CHW)
#print("========BEFORE FORWARD(end)=========")

model.eval()
output = model(img)

# print("========AFTER FORWARD(begin)=========")
#print(activations["base.7.conv.3"].shape)
# print("========AFTER FORWARD(end)=========")