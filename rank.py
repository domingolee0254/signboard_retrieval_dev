import argparse
import time

import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt
import natsort
import os 
import itertools
import cv2
import csv

from utils import metric

def search(args, query_path, reference_path):
    ### query feature extracting ### 
    start= time.time()
    query_feat = torch.load(query_path).numpy()
    faiss.normalize_L2(query_feat) # feature normalize_L2 for compare
    
    ### reference feature extracting ###
    reference_feat = torch.load(reference_path).numpy()
    faiss.normalize_L2(reference_feat) # feature normalize_L2 for compare
    print(f'[Load] {time.time() - start:.2f} sec, Query: {query_feat.shape}, Reference: {reference_feat.shape}') #print(query_feat.shape) #q.shape (100, 768) // db.shape (3377, 768)
    
    ### image feature similarity comparing ###  
    start = time.time()
    index = faiss.IndexFlatIP(reference_feat.shape[1])
    index.add(reference_feat)
    D, I = index.search(query_feat, reference_feat.shape[0])
    print(f'[Search Time] {time.time() - start:.2f} sec')
    return D, I, reference_feat

def get_rank(D, I, args, query_path):
    f = open(f'{args.result_path}/rank.txt', 'a')
    
    # get mAP and rank 
    start = time.time()
    mAP, rank_sum = metric.calculate_mAP(args.top_k, I)
    avg_rank = rank_sum / I.shape[0]
    print(f'[Total] mAP: {mAP/(I.shape[0])}')

    # save rank.txt
    rank_per_model = args.model + '_' + str(args.image_size) + '_' + str(args.image_mode) + '_' + query_path.split('.')[-2][6:] + '_top_' + str(args.top_k)
    mAP_per_model = args.model + '_' + str(args.image_size) + '_' + str(args.image_mode) + '_' + query_path.split('.')[-2][6:] + '_top_' + str(args.top_k)
    print(f'{rank_per_model},\trank: {rank_sum / I.shape[0]: .6f},\tmAP: {mAP/(I.shape[0])}', file=f)
    f.close()
    print(f'{rank_per_model}: {rank_sum / I.shape[0]: .6f}')
    
    return mAP, avg_rank, query_path

def make_csv(args, mAP, avg_rank, query_path, reference_feat):
    #row_header = img_size,latio,model,layer,top_k,mAP,avg_rank,feat_dims
    f = open(f'{args.result_path}/result.csv', 'a', encoding='utf-8')
    f.write(str(args.image_size)+','+args.image_mode+','+args.model+','+query_path.split('.')[-2][6:]+','+'top@'+str(args.top_k)+','+str(mAP)+','+str(avg_rank)+','+str(reference_feat.shape[1])+'\n')
    f.close()

def main(args, query_path, reference_path):
    D, I, reference_feat = search(args, query_path, reference_path)
    mAP, avg_rank, query_path = get_rank(D, I, args, query_path)
    make_csv(args, mAP, avg_rank, query_path, reference_feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get ranking of feature')
    parser.add_argument('-k', '--top_k', type=int, default=10)
    args = parser.parse_args()

    #query_path = '/home/image-retrieval/ndir_simulated/features/resnet50_224_00.query.pth'
    #reference_path ='/home/image-retrieval/ndir_simulated/features/resnet50_224_01.reference.pth'