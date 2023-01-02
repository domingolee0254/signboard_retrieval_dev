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

from dataset import AugmentedDataset, KeepLatioDataset
from model import vit_base_patch8_224_dino, swin_large_patch4_window7_224_in22k, beitv2_large_patch16_224_in22k
from util import load_checkpoint, negative_embedding_subtraction

@torch.no_grad()
def features_extract(args, model, data_type:str, target_layer:str):
    features = []
    img_paths = os.listdir(os.path.join(args.data_path, data_type))
    img_paths = [os.path.join(args.data_path, data_type, img) for img in img_paths]
    img_paths = natsort.natsorted(img_paths)
    #print(f'img_paths is {img_paths}')
    if args.image_mode == 'original':
        dataset = AugmentedDataset(img_paths, img_size=args.image_size)
    elif args.image_mode == 'keep_latio':
        dataset = KeepLatioDataset(img_paths, img_size=args.image_size)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc=data_type, unit='batch')
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(args.device)
        feat = model(imgs).cpu()
        print(f'feat shape is {feat.shape}')
        features.append(feat)
    print(f'feature extraction: {time.time() - start:.2f} sec')
    
    start = time.time()
    feature = np.vstack(features)
    feature = feature.reshape(feature.shape[0],-1)
    print(f"new-feature shape is {feature.shape}")
    
    print(f'convert to numpy: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)
    print(f'make dir: {time.time() - start:.2f} sec')

    start = time.time()
    features_pth_path = f'{args.feature_path}/{args.model}_{args.image_size}_{args.image_mode}_{data_type.split("/")[-1]}_{target_layer}.pth'
    torch.save(feature, features_pth_path)
    print(f'save time: {time.time() - start:.2f} sec')
    print(feature.shape)


class FeatLayer:
    def __init__(self):
        self.model = model

    def build_model(args, model, target_layer):
        print(f"========== model name: {args.model}")
        ############# device check #############
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(args.device)

        ############ 쿼리와 레퍼런스별 피처뽑기 #######
        data_types = ['00.query', '01.reference']
        for data_type in data_types:
            features_extract(args, model, data_type, target_layer)
            print(f"===== Done: {args.model} on {data_type}\n\n")
        query_path = args.feature_path+'/'+args.model+'_'+str(args.image_size)+'_'+str(args.image_mode)+'_'+data_types[0]+'_'+target_layer+'.pth'
        reference_path = args.feature_path + '/' + args.model + '_' + str(args.image_size) + '_' + str(args.image_mode)+'_'+data_types[1]+'_'+target_layer+'.pth'
        
        return query_path, reference_path

############# create CNN model #############   
def cnn(args):    
    target_layers = ['fc_layer', 'with_pooling_layer', 'without_pooling_layer']
    for target_layer in target_layers:
        if target_layer == 'fc_layer':
            model = timm.create_model(args.model, pretrained=True)
            query_path_fc_layer, reference_path_fc_layer = FeatLayer.build_model(args, model, target_layer)

        elif target_layer == 'with_pooling_layer':
            model = timm.create_model(args.model, num_classes=0, pretrained=True)
            query_path_with_pooling_layer, reference_path_with_pooling_layer = FeatLayer.build_model(args, model, target_layer)

        else:
            model = timm.create_model(args.model, pretrained=True, num_classes=0, global_pool='')
            query_path_without_pooling_layer, reference_path_without_pooling_layer = FeatLayer.build_model(args, model, target_layer)
    del model
    return [(query_path_fc_layer, reference_path_fc_layer), (query_path_with_pooling_layer, reference_path_with_pooling_layer), (query_path_without_pooling_layer, reference_path_without_pooling_layer)]

############# create ATTN model #############
def attn(args):    
    target_layers = ['fc_layer', 'token_cls']
    for target_layer in target_layers:
        if args.model == 'vit_base_patch8_224_dino':
            if target_layer == 'fc_layer':
                model = timm.create_model(args.model, pretrained=True)
                query_path_fc_layer, reference_path_fc_layer = FeatLayer.build_model(args, model, target_layer)
            elif target_layer == 'token_cls':
                model = vit_base_patch8_224_dino()
                query_path_token_cls, reference_path_token_cls = FeatLayer.build_model(args, model, target_layer)

        if args.model == 'swin_large_patch4_window7_224_in22k':
            if target_layer == 'fc_layer':
                model = timm.create_model(args.model, pretrained=True)
                query_path_fc_layer, reference_path_fc_layer = FeatLayer.build_model(args, model, target_layer)
            elif target_layer == 'token_cls':
                model = swin_large_patch4_window7_224_in22k()
                query_path_token_cls, reference_path_token_cls = FeatLayer.build_model(args, model, target_layer)  

        if args.model == 'beitv2_large_patch16_224_in22k':
            if target_layer == 'fc_layer':
                model = timm.create_model(args.model, pretrained=True)
                query_path_fc_layer, reference_path_fc_layer = FeatLayer.build_model(args, model, target_layer)
            elif target_layer == 'token_cls':
                model = beitv2_large_patch16_224_in22k()
                query_path_token_cls, reference_path_token_cls = FeatLayer.build_model(args, model, target_layer)
    del model
    return [(query_path_fc_layer, reference_path_fc_layer), (query_path_token_cls, reference_path_token_cls)]

def main(args):
    res = None
    if args.model_mode == 'CNN':
        # [(query_path_fc_layer, reference_path_fc_layer), \
        # (query_path_with_pooling_layer, reference_path_with_pooling_layer), \
        # (query_path_without_pooling_layer, reference_path_without_pooling_layer)] = cnn(args)
        res = cnn(args)
    
    else:
        # [(query_path_fc_layer, reference_path_fc_layer), \
        # (query_path_token_cls, reference_path_token_cls)] = attn(args)
        res = attn(args)

    return res
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/image-retrieval/ndir_simulated/')
    parser.add_argument('--data_path', type=str, default='/home/image-retrieval/ndir_simulated/dataset')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features')
    parser.add_argument('--result_path', type=str, default='/home/image-retrieval/ndir_simulated/result')
    parser.add_argument('--image_mode', type=str, default='original', help='original | keep_latio')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_mode', type=str, default='CNN', help='CNN | ATTN')  
    parser.add_argument('--model', type=str, default=None)  
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')    
    args = parser.parse_args()

    main(args)
    # dummy = torch.randn(2, 3, 224, 224)
    # model = timm.create_model(args.model, pretrained=True)
    # output_og = model(dummy)
    # model.reset_classifier(0, '')
    # output_new = model(dummy)
    # # print(model)
    # print(f"output_og is {output_og.shape}")
    # print(f"output_new is {output_new.shape}")