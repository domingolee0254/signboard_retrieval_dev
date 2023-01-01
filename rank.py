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

from utils import metric

def search(args, query_path, reference_path):
    ### query feature extracting ### 
    start= time.time()
    query_feat = torch.load(query_path).numpy()
    faiss.normalize_L2(query_feat) # feature normalize_L2 for compare
    
    ### reference feature extracting ###
    reference_feat = torch.load(reference_path).numpy()
    faiss.normalize_L2(reference_feat) # feature normalize_L2 for compare
    print(f'[Load] {time.time() - start:.2f} sec, Query: {query_feat.shape}, DB: {reference_feat.shape}') #print(query_feat.shape) #q.shape (100, 768) // db.shape (3377, 768)
    
    ### image feature similarity comparing ###  
    start = time.time()
    index = faiss.IndexFlatIP(reference_feat.shape[1])
    index.add(reference_feat)
    D, I = index.search(query_feat, reference_feat.shape[0])
    print(f'[Search Time] {time.time() - start:.2f} sec')

    return D,I

def get_rank(D, I, args):
    f = open(f'{args.result_path}/rank.txt', 'a')
    
    # get mAP and rank 
    start = time.time()
    mAP, rank_sum = metric.calculate_mAP(args, I)
    print(f'[Total] mAP: {mAP/(I.shape[0])}')

    # save rank.txt 
    rank_per_model = args.model + '_' + str(args.image_size)
    print(f'{rank_per_model}: {rank_sum / I.shape[0]: .6f}', file=f)
    f.close()
    print(f'{rank_per_model}: {rank_sum / I.shape[0]: .6f}')

def main(args, query_path, reference_path):
    D,I = search(args, query_path, reference_path)
    get_rank(D, I, args)

if __name__ == '__main__':
    pass