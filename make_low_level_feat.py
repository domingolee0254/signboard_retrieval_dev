import argparse
import os
import time
import natsort
import faiss
from tqdm import tqdm
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
import timm

from dataset import SimulatedDataset
from model import Hybrid_ViT, MobileNet_AVG, EfficientNet
from util import load_checkpoint, negative_embedding_subtraction


# hook()
# layer.register_forward_hook()
# input -> module -> output
def forward_hook(layer_name:str):
    def hook_fn(module, input, output):
        print('dossdood')
        activations[layer_name] = output.view(output.size(0),-1).detach().cpu()
    return hook_fn

@torch.no_grad()
def extract_frame_features(model, transform, args,target_layer_name ):
    features = []
    inter_activation = []
    img_paths = os.listdir(os.path.join(args.root_path, transform))
    img_paths = [os.path.join(args.root_path, transform, img) for img in img_paths]
    img_paths = natsort.natsorted(img_paths)
    print(f'img_paths is {img_paths}')
    dataset = SimulatedDataset(img_paths, img_size=512)
    #loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.worker)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc=transform, unit='batch')
#   print(f'\nbar is \n\n\ {bar.dtype}')
    #conv 3 layer activation => layer3.pth feature
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        #print(f'\nbatch_idx is \n\n\ {batch_idx}')
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        features.append(feat)

        # activations: dict() -> extract single value only from the data type
        inter_activation.append(list(activations.values())[0])
        assert len(list(activations.values())[0]) == len(feat), f"batch size error: Try not to use nn.DataParallel"
    print(f'feature extraction: {time.time() - start:.2f} sec')
    start = time.time()
    feature = np.vstack(features)
    activation = np.vstack(inter_activation)
    print(f"DEBUGGING: len(activation) = {len(activation)} {activation}")
    print(f"DEBUGGING: size(activation) = {activation.shape}")

    print(f"DEBUGGING: len(feature) = {len(feature)} {feature}")
    print(f"DEBUGGING: size(feature) = {len(feature)} {feature.shape}")
    print(f'convert to numpy: {time.time() - start:.2f} sec')
    
    ####mine##
    #print(f'feature is {features}')
    #print(f'vstacked feature is {feature}')
    ####
    if args.pca:
        start = time.time()
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        feature = pca.apply_py(feature)
        print(f'apply pca: {time.time() - start:.2f} sec')

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
    torch.save(feature, f'{args.feature_path}/{transform}.pth')
    torch.save(activation, f'{args.feature_path}/{transform}_{target_layer_name}.pth')
    print(f'save time: {time.time() - start:.2f} sec')
    print(feature.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/image-retrieval/ndir_simulated/')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features')
    parser.add_argument('--model', type=str, default='mobilenet_avg')  # hybrid_vit, mobilenet_avg, desc_1st, efficientnet
    parser.add_argument('--checkpoint', type=bool, default=True)

    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--pca', type=bool, default=False)   # only hybrid_vit, mobilenet_avg
    parser.add_argument('--pca_file', type=str, default='/home/image-retrieval/ndir_simulated/pca/pca_hybrid_vit_256.vt')

    parser.add_argument('--negative_embedding', type=bool, default=False)   # only desc_1st


    # intermediate layer
    parser.add_argument('--target_layer', type=int, default=3)

    args = parser.parse_args()

    ###########################image load##########################
    # img=cv.imread("/home/image-retrieval/ndir_simulated/test2.jpg")
    # img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomResizedCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # img=np.array(img)
    # img=transform(img)
    # img=img.unsqueeze(0)
    #print(img.size())

    
    # ##############################################

    ################# model load #################
    print("loading model")
    model = None
    checkpoint = None
    if args.model == 'hybrid_vit':
        model = Hybrid_ViT()
        checkpoint = '/workspace/ckpts/res26_vits32_fivr_triplet.pth'
    elif args.model == 'mobilenet_avg':
        model = MobileNet_AVG()
        checkpoint = '/home/image-retrieval/ndir_simulated/ckpts/mobilenet_avg_ep16_ckpt.pth'
    elif args.model == 'desc_1st':
        model = EfficientNet(eval_p=1.0)
        checkpoint = '/workspace/ckpts/efficientnet_v2_disc_contrastive.pth.tar'
    
    model.to(args.device)
    
    activations = dict() 
    layers = dict()
    ###############################################




    ################## Check device #################
    #if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
    
    #print('before load')  # check weight load
    #check_parameters = list(model.named_parameters())[-7]
    #print(check_parameters[0], check_parameters[1][:10])

    #if args.checkpoint:
        #load_checkpoint(args, model, checkpoint)

    #print('after load')  # check weight load
    #check_parameters = list(model.named_parameters())[-7]
    #print(check_parameters[0], check_parameters[1][:10])
    ################################################






    ################## forward hook#################
    print("\n\n")
    print("="*50)
    target_layer_name = None
    for name, layer in model.named_modules():
        # "conv.3" refers to output of any inverted residual block,
        # which passed batch norm layer, except 1st, 2nd layer(inverted block).
        #print(layer)
        if f"base.{args.target_layer}.conv.3" in name:
            #layers[name] = layer
            layer.register_forward_hook(forward_hook(name))
            print(f"hook: {name}")
            target_layer_name = name
        #else:
        #    print(name)

    #print(f"activations:{activations}")
    print("="*50)
    print("\n\n")

    ################## Check device #################
    #if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    
    #print('before load')  # check weight load
    #check_parameters = list(model.named_parameters())[-7]
    #print(check_parameters[0], check_parameters[1][:10])

    #if args.checkpoint:
    #    load_checkpoint(args, model, checkpoint)

    #print('after load')  # check weight load
    #check_parameters = list(model.named_parameters())[-7]
    #print(check_parameters[0], check_parameters[1][:10])
    ################################################






    ################## MAIN #################
    transform = 'db_origin'
    extract_frame_features(model, transform, args, target_layer_name)


    ################## MAIN #################
    print(f"after extract_frame_..() activations: {activations}")
    
    #print("========BEFORE FORWARD(begin)=========")
    #print(activations) # (N, C, H, W) -> (N, CHW)
    #print("========BEFORE FORWARD(end)=========")
    ###################################################
    #model.eval()
    #output = model(img)
        
