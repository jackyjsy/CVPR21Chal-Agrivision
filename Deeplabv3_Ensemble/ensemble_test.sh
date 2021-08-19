#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PYTHON=${PYTHON:-"python"}

export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29500
CONFIG="config/agri-test_res101.yaml"

arr=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
NUM_GPU=${#arr[*]}

export NCCL_SOCKET_IFNAME=$NCCL_SOCKET

echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
echo "NUM_GPU:" $NUM_GPU
echo "NCCL_SOCKET_IFNAME:" $NCCL_SOCKET_IFNAME

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=$PORT \
    ensemble_test.py --cfg $CONFIG
