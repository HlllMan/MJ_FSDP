#!/bin/bash
# 快速启动脚本 - Master 节点
# 自动检测 IP 并启动

cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final

# 自动获取本机 IP（过滤掉 127.0.0.1 和 docker0）
MASTER_IP=$(hostname -I | awk '{print $1}')

if [ -z "$MASTER_IP" ]; then
    echo "错误: 无法自动检测 IP 地址"
    echo "请手动指定: bash docker/run_docker_mlp.sh 0 <your_ip> 29600"
    exit 1
fi

echo "==================== Master 节点启动 ===================="
echo "自动检测到的 IP: $MASTER_IP"
echo ""
echo "请在 Worker 节点 (DGX-092) 上运行："
echo "  bash docker/run_docker_mlp.sh 1 $MASTER_IP 29600"
echo ""
echo "或者使用快速启动脚本："
echo "  bash docker/quick_start_worker.sh $MASTER_IP"
echo "========================================================"
echo ""
read -p "按 Enter 继续启动 Master 节点..."

# 启动 Master
bash docker/run_docker_mlp.sh 0 $MASTER_IP 29600


