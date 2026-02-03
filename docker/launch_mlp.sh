#!/bin/bash
# 容器内部启动脚本 - MLP 任务
# TP=2, DP=8, 2节点×8卡

# NCCL 配置（商汤平台优化参数）
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_AR_THRESHOLD=0
export NCCL_DEBUG=WARN  # 生产环境：只显示警告和错误
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5

# 启动参数（优先从命令行参数，如果没有则从环境变量）
NODE_RANK=${1:-${NODE_RANK}}
MASTER_ADDR=${2:-${MASTER_ADDR}}
MASTER_PORT=${3:-${MASTER_PORT:-29600}}

echo "==================== 训练配置 ===================="
echo "Task: MLP with TP+DP"
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo ""
echo "Parallelism Configuration:"
echo "  - TP_SIZE: 2 (每 2 张卡一组)"
echo "  - DP_SIZE: 8 (8 个数据副本)"
echo "  - Total GPUs: 16 (2 nodes × 8 GPUs)"
echo ""
echo "GPU Topology:"
if [ "$NODE_RANK" == "0" ]; then
    echo "  Node 0 (Master):"
    echo "    - DP Group 0: GPU 0,1 (TP)"
    echo "    - DP Group 1: GPU 2,3 (TP)"
    echo "    - DP Group 2: GPU 4,5 (TP)"
    echo "    - DP Group 3: GPU 6,7 (TP)"
else
    echo "  Node 1 (Worker):"
    echo "    - DP Group 4: GPU 0,1 (TP)"
    echo "    - DP Group 5: GPU 2,3 (TP)"
    echo "    - DP Group 6: GPU 4,5 (TP)"
    echo "    - DP Group 7: GPU 6,7 (TP)"
fi
echo "================================================="

# 显示网络接口（调试用）
echo ""
echo "==================== 网络接口 ===================="
if command -v ip &> /dev/null; then
    ip addr show | grep "inet " | grep -v 127.0.0.1
elif command -v ifconfig &> /dev/null; then
    ifconfig | grep "inet " | grep -v 127.0.0.1
else
    echo "警告: 无法检测网络接口"
fi
echo "================================================="

# 显示 GPU 信息
echo ""
echo "==================== GPU 信息 ===================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================="

echo ""
echo "启动训练..."
echo ""

# 启动 torchrun
torchrun \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=8 \
    TP.py


