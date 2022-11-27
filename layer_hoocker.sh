#!/bin/bash


TARGET=make_low_level_feat.py
MODEL=mobilenet_avg
CHECKPOINT=/home/image-retrieval/ndir_simulated/ckpts/mobilenet_avg_ep16_ckpt.pth


for target_layer in {2..17}
do
python ${TARGET}\
    --model ${MODEL}\
    --checkpoint ${CHECKPOINT}\
    --target_layer ${target_layer}
done