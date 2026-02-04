#!/bin/bash
# Docker 启动脚本 - MLP 任务（使用新版 PyTorch，支持 device_mesh）
# PyTorch 2.3+ (NGC 24.04 或更新)
# 用法: bash run_docker_new.sh <node_rank> <master_addr> [master_port]

NODE_RANK=${1}
MASTER_ADDR=${2}
MASTER_PORT=${3:-29600}

if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ]; then
    echo "用法: bash run_docker_new.sh <node_rank> <master_addr> [master_port]"
    echo ""
    echo "示例："
    echo "  节点 0 (Master): bash run_docker_new.sh 0 192.168.1.11 29600"
    echo "  节点 1 (Worker): bash run_docker_new.sh 1 192.168.1.11 29600"
    echo ""
    echo "配置："
    echo "  - 2 个节点，每个节点 8 张卡"
    echo "  - TP_SIZE = 2 (Tensor Parallelism)"
    echo "  - DP_SIZE = 8 (Data Parallelism)"
    echo "  - PyTorch 镜像: nvcr.io/nvidia/pytorch:24.07-py3"
    echo "  - PyTorch 版本: 2.4+ (支持 device_mesh)"
    exit 1
fi

echo "==================== 启动配置 ===================="
echo "Task: MLP Training with TP+DP (New PyTorch)"
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "GPUs per Node: 8"
echo "Total Nodes: 2"
echo "TP Size: 2"
echo "DP Size: 8"
echo "Total GPUs: 16"
echo "PyTorch Image: nvcr.io/nvidia/pytorch:24.07-py3"
echo "================================================="

# 检查镜像是否存在
if ! docker images | grep -q "nvcr.io/nvidia/pytorch.*24.07-py3"; then
    echo ""
    echo "警告: 未找到 PyTorch 24.07-py3 镜像"
    echo "正在拉取镜像，这可能需要几分钟..."
    echo ""
    docker pull nvcr.io/nvidia/pytorch:24.07-py3
fi

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
    nvcr.io/nvidia/pytorch:24.07-py3 \
    bash -c "cd /workspace/final && bash docker/launch_mlp.sh $NODE_RANK $MASTER_ADDR $MASTER_PORT"


