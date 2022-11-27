import argparse
import time

import faiss
import numpy as np
import torch


def get_rank(args):
    f = open(f'{args.feature_path}/rank.txt', 'a')

    #db_path = f'{args.feature_path}/{transform}.pth'
    # db_path = '/home/image-retrieval/ndir_simulated/features/db_origin_base.3.conv.3.pth'
    db_path = '/home/image-retrieval/ndir_simulated/features/db_origin.pth'
    db_feat = torch.load(db_path).numpy()
    faiss.normalize_L2(db_feat) 

    start = time.time()
    # index = faiss.IndexFlatL2(query_feat.shape[1])
    index = faiss.IndexFlatIP(query_feat.shape[1])
    index.add(db_feat)
    D, I = index.search(query_feat, query_feat.shape[0])
    #print(D.shape, I.shape)
    print(f'{time.time() - start:.2f} sec')

    start = time.time()
    rank_sum = 0
    for i in range(I.shape[1]):
        rank = np.where(I[i] == i)
        print(I[i])
        rank_sum += rank[0][0]

    print(f'{time.time() - start:.2f} sec')

    print(f'layer_3: {rank_sum / I.shape[0]: .6f}', file=f)
    print(f'[layer_3] finish')
    f.close() 

    print(f'layer_17: {rank_sum / I.shape[0]: .6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get average rank with simulated dataset')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features/')
    args = parser.parse_args()

    #query_path = f'{args.feature_path}/q_origin.pth'
    # query_path = '/home/image-retrieval/ndir_simulated/features/q_origin_base.3.conv.3.pth'
    query_path = '/home/image-retrieval/ndir_simulated/features/q_origin.pth'
    query_feat = torch.load(query_path).numpy()
    faiss.normalize_L2(query_feat)
    
    #transforms = ['original']
    # transforms = ['BlackBorder_01', 'BlackBorder_02', 'Brightness_01', 'Brightness_02', 'Brightness_03',
    #               'Crop_01', 'Crop_02', 'Flip_H', 'Flip_V', 'GrayScale',
    #               'Logo_01', 'Logo_02', 'Logo_03', 'PIP',
    #               'Rotation_01', 'Rotation_02', 'Rotation_03',
    #               'multi_BrC', 'multi_BrL', 'multi_BrP',
    #               'multi_CL', 'multi_CP', 'multi_LP', 'multi_BrCP']

    #or transform in transforms:
    get_rank(args)

