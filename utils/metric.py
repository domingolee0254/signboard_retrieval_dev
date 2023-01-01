import time
import numpy as np

def calculate_ap(idx, gt):    
    ap,rank=0.,None    
    for n,i in enumerate(idx,start=1):
        if i==gt:
            ap=1/n
            rank=n
            break
    return ap,rank

def calculate_mAP(args, I):
    rank_sum = 0
    mAP=0.    

    start = time.time()
    for i in range(I.shape[0]):        
        rank = np.where(I[i] == i)
        rank_sum += rank[0][0]
        idx=I[i]
        if args.topk:
            idx=I[i,:args.topk]
        ap,r=calculate_ap(idx,i)
        mAP+=ap
        print(f'[Query {i+1}] Rank: {r}, AP: {ap:.4f}, mAP: {mAP/(i+1)}')
    print(f'{time.time() - start:.2f} sec')

    return mAP, rank_sum