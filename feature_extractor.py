import argparse
import os
import time
import faiss
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import natsort

from torch.utils.data import  DataLoader
from torchvision import models
 
import torchvision.transforms as transforms
import torchvision.datasets as dataset
 
import matplotlib.pyplot as plt
import numpy as np
import re
import cv2 as cv
import timm

from dataset import SimulatedDataset
from model import Hybrid_ViT, MobileNet_AVG, EfficientNet
from util import load_checkpoint, negative_embedding_subtraction

@torch.no_grad()
def features_extract(args, model, data_type:str):
    features = []
    img_paths = os.listdir(os.path.join(args.data_path, data_type))
    img_paths = [os.path.join(args.data_path, data_type, img) for img in img_paths]
    img_paths = natsort.natsorted(img_paths)
    #print(f'img_paths is {img_paths}')
    dataset = SimulatedDataset(img_paths, img_size=args.image_size)
    #loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc=data_type, unit='batch')
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        features.append(feat)
    print(f'feature extraction: {time.time() - start:.2f} sec')
    
    start = time.time()
    feature = np.vstack(features)
    print(f'convert to numpy: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)
    print(f'make dir: {time.time() - start:.2f} sec')

    start = time.time()
    torch.save(feature, f'{args.feature_path}/{args.model}_{args.image_size}_{data_type.split("/")[-1]}.pth')
    print(f'save time: {time.time() - start:.2f} sec')
    print(feature.shape)

def main(args):    
    ############# create model & check device #############    
    print(f"========== model name: {args.model}")
    try:
        model = timm.create_model(args.model, pretrained=True)
        model.to(args.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    except:
        print(f"check model name")
    ########################################

    ############ 쿼리와 레퍼런스별 피처뽑기 #######
    data_types = ['00.query', '01.reference']
    for data_type in data_types:
        features_extract(args, model, data_type)
        print(f"===== Done: {args.model} on {data_type}\n\n")
    #############################################

    #################### send to rank.py ###################################################
    query_path = args.feature_path + '/' + args.model + '_' + str(args.image_size) + '_' + data_types[0] + '.pth'
    reference_path = args.feature_path + '/' + args.model + '_' + str(args.image_size) + '_' + data_types[1] + '.pth'
    ########################################################################################
    del model
    
    return query_path, reference_path

if __name__ == '__main__':
    pass