#!/bin/bash
# 快速启动脚本 - Worker 节点
# 用法: bash quick_start_worker.sh <master_ip>

MASTER_IP=${1}

if [ -z "$MASTER_IP" ]; then
    echo "用法: bash quick_start_worker.sh <master_ip>"
    echo ""
    echo "示例: bash quick_start_worker.sh 192.168.1.100"
    echo ""
    echo "提示: Master IP 可以在 Master 节点的启动输出中找到"
    exit 1
fi

cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final

echo "==================== Worker 节点启动 ===================="
echo "Master IP: $MASTER_IP"
echo "Master Port: 29600"
echo "Node Rank: 1 (Worker)"
echo "========================================================"
echo ""
read -p "按 Enter 继续启动 Worker 节点..."

# 启动 Worker
bash docker/run_docker_mlp.sh 1 $MASTER_IP 29600


