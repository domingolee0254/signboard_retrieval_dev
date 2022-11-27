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
def extract_frame_features(model, data_type:str, img_size, args):
    features = []
    img_paths = os.listdir(os.path.join(args.root_path, data_type))
    img_paths = [os.path.join(args.root_path, data_type, img) for img in img_paths]
    img_paths = natsort.natsorted(img_paths)
    print(f'img_paths is {img_paths}')
    dataset = SimulatedDataset(img_paths, img_size=img_size)
    #loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc=data_type, unit='batch')
#   print(f'\nbar is \n\n\ {bar.dtype}')
    #conv 3 layer activation => layer3.pth feature
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        features.append(feat)
    print(f'feature extraction: {time.time() - start:.2f} sec')
    start = time.time()
    feature = np.vstack(features)

    print(f'convert to numpy: {time.time() - start:.2f} sec')
    
    if args.pca:
        start = time.time()
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        feature = pca.apply_py(feature)
        print(f'apply pca: {time.time() - start:.2f} sec`')

    if args.negative_embedding:
        start = time.time()
        print("negative embedding subtraction")
        train = np.load(f'/home/image-retrieval/ndir_simulated/negative_embbeding/train_feats.npy')
        index_train = faiss.IndexFlatIP(train.shape[1])
        ngpu = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = False
        index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
        index_train.add(train)

        # DBA on training set
        sim, ind = index_train.search(train, k=10)
        k = 10
        alpha = 3.0
        _train = (train[ind[:, :k]] * (sim[:, :k, None] ** alpha)).sum(axis=1)
        _train /= np.linalg.norm(_train, axis=1, keepdims=True)

        index_train = faiss.IndexFlatIP(train.shape[1])
        ngpu = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = False
        index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
        index_train.add(_train)
        feature = negative_embedding_subtraction(feature, _train, index_train, num_iter=1, k=10, beta=0.35)
        print(f'apply neg embedding: {time.time() - start:.2f} sec')
    else:
        start = time.time()
        print("normalizing descriptors")
        faiss.normalize_L2(feature)
        print(f'normalize: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)
    print(f'make dir: {time.time() - start:.2f} sec')

    start = time.time()
    torch.save(feature, f'{args.feature_path}/{model_name}_{img_size}_{data_type}.pth')
    print(f'save time: {time.time() - start:.2f} sec')
    print(feature.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/image-retrieval/ndir_simulated/')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features_1121')
    parser.add_argument('--model', type=str, default='mobilenet_avg')  # hybrid_vit, mobilenet_avg, desc_1st, efficientnet
    parser.add_argument('--checkpoint', type=bool, default=True)

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--pca', type=bool, default=False)   # only hybrid_vit, mobilenet_avg
    parser.add_argument('--pca_file', type=str, default='/home/image-retrieval/ndir_simulated/pca/pca_hybrid_vit_256.vt')

    parser.add_argument('--negative_embedding', type=bool, default=False)   # only desc_1st


    # intermediate layer
    parser.add_argument('--target_layer', type=int, default=3)

    args = parser.parse_args()

    data_types = ['01.db_new', '01.q_origin']
    model_list = ['vit_base_patch8_224_dino']
    img_size = 224
    #model_list = ['vit_huge_patch14_224_in21k', 'twins_pcpvt_large', 'twins_svt_large']
    #model_list = ['swin_large_patch4_window7_224_in22k', 'vit_base_patch16_224_in21k', 'convnext_base_in22ft1k', 'coatnet_0_rw_224', 'coatnet_bn_0_rw_224']
    for model_name in model_list:
        print(f"========== model name: {model_name}")
        try:
            model = timm.create_model(model_name, pretrained=True)

            model.to(args.device)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        except:
            continue


        for data_type in data_types:
            extract_frame_features(model, data_type, img_size, args)
            print(f"\n\n===== Done: {model_name} on {data_type}")
        
        del model
