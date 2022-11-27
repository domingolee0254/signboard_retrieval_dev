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


def get_rank(args):
    f = open(f'{args.feature_path}/rank.txt', 'a')

    #db_path = f'{args.feature_path}/{transform}.pth'
    # db_path = '/home/image-retrieval/ndir_simulated/features/db_origin_base.3.conv.3.pth'
    db_path = '/home/image-retrieval/ndir_simulated/features_1121/mobilenetv2_050_224_01.db_new.pth'
    db_feat = torch.load(db_path).numpy()
    faiss.normalize_L2(db_feat) 

    start = time.time()
    # index = faiss.IndexFlatL2(query_feat.shape[1])
    index = faiss.IndexFlatIP(query_feat.shape[1])
    index.add(db_feat)
    D, I = index.search(query_feat, query_feat.shape[0])
    #print(D, I)
    print(f'{time.time() - start:.2f} sec')

    # #######og#########
    # start = time.time()
    # rank_sum = 0
    # for i in range(I.shape[1]):
    #     rank = np.where(I[i] == i)
    #     print(I[i])
    #     rank_sum += rank[0][0]
    # ##################
    start = time.time()
    rank_sum = 0
    for i in range(I.shape[1]):
        try:
            rank = np.where(I[i] == i)
            #print(I[i])
            rank_sum += rank[0][0]
        except:
            continue
    print(f'{time.time() - start:.2f} sec')

    print(f'mobilenetv2_050_224: {rank_sum / I.shape[0]: .6f}', file=f)
    print(f'[mobilenetv2_050_224] finish')
    f.close() 

    print(f'mobilenetv2_050_224: {rank_sum / I.shape[0]: .6f}')
    ############### quality analysis #####################

    #root_path = '/home/image-retrieval/ndir_simulated/'
    root_path = os.getcwd()
    q_img_path_list = []
    db_img_path_list = []
    res_save_dir =  '/home/image-retrieval/ndir_simulated/res_save_dir'
    # #q_img_path = '/home/image-retrieval/ndir_simulated/01.q_origin/'
    # #db_img_path = '/home/image-retrieval/ndir_simulated/01.db_new/'
    data_types = ['01.q_origin', '01.db_new']
    for data_type in data_types:
        if data_type == '01.q_origin':
            q_img_paths = os.listdir(os.path.join(root_path, data_type))
            q_img_paths = [os.path.join(root_path, data_type, img) for img in q_img_paths]
            q_img_path_list = natsort.natsorted(q_img_paths)
        else:     
            db_img_paths = os.listdir(os.path.join(root_path, data_type))
            db_img_paths = [os.path.join(root_path, data_type, img) for img in db_img_paths]
            db_img_path_list = natsort.natsorted(db_img_paths)

    #print(f'q_img_path_list is {q_img_path_list}')
    #print(f'db is {db_img_path_list}')
    
    #query_idx = 0 ##rank1 #677 #42 #41 #221(-) #1205 #222 #83 #19 #7 #1224 ##rank3 #2(-) #11(-) ##etc #341(같은 폰트)
    topk = 5

    fig = plt.figure(figsize=(10,10)) # rows*cols 행렬의 i번째 subplot 생성
    rows = topk
    cols = 2

    # 하얀 바탕으로
    fig.patch.set_facecolor('xkcd:white')
    # 지정한 query 1개의 topk개의 db images
    for q_idx, each_q_path in enumerate(q_img_path_list):
        query_img = cv2.imread(each_q_path)
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"Query:{os.path.basename(each_q_path[:-4])}", fontdict={'fontsize': 10})
        for tmp_idx, db_idx in enumerate(I[q_idx][:topk]): # [  99   66 3290]
            print(tmp_idx, db_idx)
            print(f"matched db_img_path is {db_img_path_list[db_idx]}")
            matched_db_img = cv2.imread(db_img_path_list[db_idx]) #쿼리 이미지 한번 돌때 각각의 top 3 이미지 가가각 
            ax = fig.add_subplot(rows, cols, 2*(tmp_idx+1))
            ax.imshow(cv2.cvtColor(matched_db_img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"DB_top{tmp_idx+1}", fontdict={'fontsize': 10})
            fig.savefig(res_save_dir + '/' +str(os.path.basename(each_q_path[:-4])) +'.jpg')
        fig.clf()
        print("==="*100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get average rank with simulated dataset')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features_1121/')
    args = parser.parse_args()

    #query_path = f'{args.feature_path}/q_origin.pth'
    # query_path = '/home/image-retrieval/ndir_simulated/features/q_origin_base.3.conv.3.pth'
    query_path = '/home/image-retrieval/ndir_simulated/features_1121/mobilenetv2_050_224_01.q_origin.pth'
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



