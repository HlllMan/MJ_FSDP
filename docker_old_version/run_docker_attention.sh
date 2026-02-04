#!/bin/bash
# 使用 Attention 目录的代码（不需要 device_mesh）
# 用法: bash run_docker_attention.sh <node_rank> <master_addr> [master_port]

NODE_RANK=${1}
MASTER_ADDR=${2}
MASTER_PORT=${3:-29600}

if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ]; then
    echo "用法: bash run_docker_attention.sh <node_rank> <master_addr> [master_port]"
    echo ""
    echo "推荐使用 InfiniBand IP："
    echo "  例如: bash run_docker_attention.sh 0 192.168.1.11 29600"
    exit 1
fi

echo "==================== 启动配置 ===================="
echo "Task: Attention Training with TP+DP"
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "================================================="

docker run --gpus all \
    --runtime=nvidia \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    --rm \
    --privileged \
    --pids-limit=-1 \
    --security-opt seccomp=unconfined \
    -e NCCL_IB_TIMEOUT=22 \
    -e NCCL_IB_RETRY_CNT=13 \
    -e NCCL_IB_AR_THRESHOLD=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -v /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP:/workspace \
    nvcr.io/nvidia/pytorch:23.10-py3 \
    bash -c "cd /workspace/Attention && bash launch_prod.sh $NODE_RANK $MASTER_ADDR $MASTER_PORT"


