# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 250 && echo "eth Launched." &
P0=$!

wait $P0 $P1 $P2 $P3 $P4