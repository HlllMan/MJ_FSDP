#!/bin/bash
# 快速启动脚本 - Master 节点（新版 PyTorch）
# 智能选择最佳 IP（优先 InfiniBand）

cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final/docker

# 智能选择 IP：优先使用 InfiniBand 网卡
echo "检测网络接口..."
echo ""

# 获取所有 IP
ALL_IPS=$(hostname -I)

# 尝试查找 InfiniBand IP（通常是 192.168.x.x 且来自 ibp* 网卡）
IB_IP=$(ip addr show | grep -E 'inet.*ibp' | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

# 如果没有 InfiniBand，尝试找以太网 IP（排除 Docker 和 lo）
if [ -z "$IB_IP" ]; then
    ETH_IP=$(ip addr show | grep -E 'inet.*enp' | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | grep -v '193.0' | head -1)
    MASTER_IP=$ETH_IP
    NETWORK_TYPE="Ethernet"
else
    MASTER_IP=$IB_IP
    NETWORK_TYPE="InfiniBand"
fi

# 如果还是找不到，用第一个非 Docker IP
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(hostname -I | awk '{print $1}' | grep -v '193.0' | head -1)
    NETWORK_TYPE="Unknown"
fi

if [ -z "$MASTER_IP" ]; then
    echo "错误: 无法自动检测 IP 地址"
    echo ""
    echo "请手动运行："
    echo "  bash run_docker_new.sh 0 <your_ip> 29600"
    echo ""
    echo "可用的 IP 地址："
    hostname -I
    exit 1
fi

echo "==================== Master 节点启动 (新版 PyTorch) ===================="
echo "网络类型: $NETWORK_TYPE"
echo "自动检测到的 IP: $MASTER_IP"
echo ""
echo "所有可用 IP:"
echo "$ALL_IPS"
echo ""
echo "请在 Worker 节点 (DGX-092) 上运行："
echo "  bash run_docker_new.sh 1 $MASTER_IP 29600"
echo ""
echo "或者使用快速启动脚本："
echo "  bash quick_start_worker_new.sh $MASTER_IP"
echo "========================================================================"
echo ""
read -p "按 Enter 继续启动 Master 节点（或 Ctrl+C 取消）..."

# 启动 Master
bash run_docker_new.sh 0 $MASTER_IP 29600


