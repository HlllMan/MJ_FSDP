#!/bin/bash
# Docker 启动脚本 - MLP 任务（TP=2, DP=8, 2节点×8卡）
# 用法: bash run_docker_mlp.sh <node_rank> <master_addr> [master_port] [--use-custom-image]
#  DEBUG_STEP=0 VALIDATE_BASELINE=0 \
#  bash docker/run_docker_mlp.sh 0 172.17.226.11 29610 --use-custom-image
#  DEBUG_STEP=0 VALIDATE_BASELINE=0 \
#  bash docker/run_docker_mlp.sh 1 172.17.226.11 29610 --use-custom-image


NODE_RANK=${1}
MASTER_ADDR=${2}
MASTER_PORT=${3:-29600}
USE_CUSTOM_IMAGE=${4}

# 检查是否使用自定义镜像
if [ "${USE_CUSTOM_IMAGE}" = "--use-custom-image" ]; then
    DOCKER_IMAGE="pytorch-tp-dp:latest"
    SKIP_UPGRADE=true
else
    DOCKER_IMAGE="nvcr.io/nvidia/pytorch:24.01-py3"
    SKIP_UPGRADE=false
fi

if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ]; then
    echo "用法: bash run_docker_mlp.sh <node_rank> <master_addr> [master_port] [--use-custom-image]"
    echo ""
    echo "示例："
    echo "  使用官方镜像（自动升级 PyTorch）:"
    echo "    节点 0 (Master): bash run_docker_mlp.sh 0 192.168.1.100 29600"
    echo "    节点 1 (Worker): bash run_docker_mlp.sh 1 192.168.1.100 29600"
    echo ""
    echo "  使用自定义镜像（已包含最新 PyTorch，启动更快）:"
    echo "    节点 0 (Master): bash run_docker_mlp.sh 0 192.168.1.100 29600 --use-custom-image"
    echo "    节点 1 (Worker): bash run_docker_mlp.sh 1 192.168.1.100 29600 --use-custom-image"
    echo ""
    echo "配置："
    echo "  - 2 个节点，每个节点 8 张卡"
    echo "  - TP_SIZE = 2 (Tensor Parallelism)"
    echo "  - DP_SIZE = 8 (Data Parallelism)"
    echo "  - Total GPUs = 16 (2 nodes × 8 GPUs)"
    echo ""
    echo "构建自定义镜像:"
    echo "  bash docker/build_docker.sh"
    exit 1
fi

echo "==================== 启动配置 ===================="
echo "Task: MLP Training with TP+DP"
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "GPUs per Node: 8"
echo "Total Nodes: 2"
echo "TP Size: 2"
echo "DP Size: 8"
echo "Total GPUs: 16"
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
    -e NODE_RANK=${NODE_RANK} \
    -e MASTER_ADDR=${MASTER_ADDR} \
    -e MASTER_PORT=${MASTER_PORT} \
    -v /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP:/workspace \
    -v /tmp/pip_cache:/root/.cache/pip \
    ${DOCKER_IMAGE} \
    bash -c "
        set -e
        echo '==================== 环境信息 ===================='
        echo 'PyTorch 版本:'
        python -c 'import torch; print(torch.__version__)'
        echo ''
        echo 'CUDA 信息:'
        python << PYEOF
import torch
print('CUDA available:', torch.cuda.is_available())
cuda_ver = torch.version.cuda if torch.cuda.is_available() else 'N/A'
print('CUDA version:', cuda_ver)
print('GPU count:', torch.cuda.device_count())
PYEOF
        echo '================================================='
        echo ''
        if [ \"${SKIP_UPGRADE}\" = \"true\" ]; then
            echo '==================== PyTorch 版本信息 ===================='
            echo '使用自定义镜像，已包含最新 PyTorch，跳过升级'
        else
            echo '==================== 检查 PyTorch 版本 ===================='
            CURRENT_VERSION=\$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '0.0.0')
            echo '当前 PyTorch 版本: '\${CURRENT_VERSION}
            echo ''
            # 检查版本是否 >= 2.2.0
            NEED_UPGRADE=\$(python << PYEOF
import sys
try:
    import torch
    version = torch.__version__.split('+')[0]  # 移除开发版本后缀
    parts = [int(x) for x in version.split('.')[:3]]
    if parts[0] > 2 or (parts[0] == 2 and parts[1] >= 2):
        print('no')
    else:
        print('yes')
except:
    print('yes')
PYEOF
)
            if [ \"\${NEED_UPGRADE}\" = \"yes\" ]; then
                echo 'PyTorch 版本 < 2.2.0，需要升级...'
                echo '检查 CUDA 版本...'
                nvcc --version | grep release || echo '无法检测 CUDA 版本'
                echo ''
                echo '升级 PyTorch 到最新版本 (支持 device_mesh)...'
                pip install --upgrade pip >/dev/null 2>&1
                pip install --upgrade torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | grep -E '(Collecting|Installing|Successfully|ERROR)' || true
                echo ''
            else
                echo 'PyTorch 版本 >= 2.2.0，跳过升级 ✓'
            fi
        fi
        echo ''
        echo '验证安装:'
        python -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); cuda_ver = torch.version.cuda if torch.cuda.is_available() else \"N/A\"; print(\"CUDA version:\", cuda_ver)'
        echo '================================================='
        echo ''
        cd /workspace/final && bash docker/launch_mlp.sh ${NODE_RANK} ${MASTER_ADDR} ${MASTER_PORT}
    "


