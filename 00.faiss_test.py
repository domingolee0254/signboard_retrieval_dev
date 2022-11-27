import faiss
import numpy as np
import random

import torch
from torch import nn

class CustomModel(nn.Module):
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.load('/root/mnt/ndir_simulated/features/origin.pth', map_location=device)
# print(f"model.type is {model.type()}")
# print(f"model.shape is {model.shape}")
# print(f"model is {model}")

# with torch.no_grad():
#     model.eval()
#     inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
#     outputs = model(inputs)
#     print(outputs)


# 코사인 유사도 (Cosine Similarity) 를 이용해서 가장 가까운 벡터를 찾으려면 몇가지를 바꿔줘야 한다.
# 코사인 유사도 (Cosine Similarity) 를 사용하려면 벡터 내적으로 색인하는 index를 만들면 된다.
# 코사인 유사도를 계산하라면 벡터 내적을 필연적으로 계산해야 하기 때문이다.
 
# 랜덤으로 10차원 벡터를 10개 생성
vectors = torch.load('/root/mnt/ndir_simulated/features/origin.pth', map_location=device)
# 10차원짜리 벡터를 검색하기 위한 Faiss index를 생성
# 생성할 때 Inner Product을 검색할 수 있는 index를 생성한다.
index = faiss.IndexFlatIP(1280)
# 아래는 위와 동일하다.
# index = faiss.index_factory(300, "Flat", faiss.METRIC_INNER_PRODUCT)
 
# Vector를 numpy array로 바꾸기
vectors = vectors.cpu().numpy().astype(np.float32)
#print(f"vectors is {vectors}")
#print(f".type is {vectors.type()}")

# vectors를 노말라이즈 해준다.
faiss.normalize_L2(vectors)
# 아까 만든 10x10 벡터를 Faiss index에 넣기
index.add(vectors)
# query vector를 하나 만들기
query_vector = vectors
print("query vector: {}".format(query_vector))
# 가장 가까운 것 10개 찾기
distances, indices = index.search(query_vector, 1)
# 결과룰 출력하자.
idx = 0
for i in indices:
    print("v{}: {}, distance={}".format(idx+1, vectors[i], distances[idx]))
    idx += 1