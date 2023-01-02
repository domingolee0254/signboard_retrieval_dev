#!/bin/bash
FEATURE_PATH='/home/image-retrieval/ndir_simulated/features' # layer_wise
DATA_PATH='/home/image-retrieval/ndir_simulated/dataset' # orginal, keep_ratio
IMAGE_MODE=$1
IMAGE_SIZE=224 # resolution_wise
MODEL_MODE=$2
MODEL=$3 # model_wise
BATCH_SIZE=64
NUM_WORKERS=0
TOP_K=$4

python main.py \
    --data_path $DATA_PATH \
    --feature_path $FEATURE_PATH \
    --image_mode $IMAGE_MODE \
    --image_size $IMAGE_SIZE \
    --model_mode $MODEL_MODE \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --top_k $TOP_K \
    