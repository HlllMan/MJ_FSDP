# MLP 训练 Docker 启动指南

## 🎯 配置概览

- **节点数量**: 2 个节点（DGX-011, DGX-092）
- **每节点 GPU**: 8 张卡
- **总 GPU**: 16 张卡
- **TP_SIZE**: 2（Tensor Parallelism）
- **DP_SIZE**: 8（Data Parallelism）
- **任务**: MLP 训练（TP.py）
- **Docker 镜像**: `nvcr.io/nvidia/pytorch:24.01-py3`（PyTorch 2.2+ 预装，无需升级）

## 📊 GPU 拓扑结构

```
总共 16 张 GPU，分组如下：

DP Group 0: Node0-GPU0, Node0-GPU1 (TP)
DP Group 1: Node0-GPU2, Node0-GPU3 (TP)
DP Group 2: Node0-GPU4, Node0-GPU5 (TP)
DP Group 3: Node0-GPU6, Node0-GPU7 (TP)
DP Group 4: Node1-GPU0, Node1-GPU1 (TP)
DP Group 5: Node1-GPU2, Node1-GPU3 (TP)
DP Group 6: Node1-GPU4, Node1-GPU5 (TP)
DP Group 7: Node1-GPU6, Node1-GPU7 (TP)

每个 DP Group 内的 2 张卡通过 TP 切分模型
8 个 DP Group 处理不同的数据批次
```

## 🚀 启动步骤

### **1. 确定 Master 节点的 IP 地址**

在 DGX-011 (Master 节点) 上：

```bash
# 查看 IP 地址
hostname -I
# 或
ip addr show | grep "inet " | grep -v 127.0.0.1
```

假设 Master 的 IP 是: `192.168.1.100`

### **2. 启动 Master 节点 (DGX-011)**

```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final

# 启动 Node 0（Master）
bash run_docker_mlp.sh 0 192.168.1.100 29600
```

参数说明：
- `0`: node_rank（Master 节点）
- `192.168.1.100`: master_addr（Master 的 IP）
- `29600`: master_port（通信端口）

### **3. 启动 Worker 节点 (DGX-092)**

在 **DGX-092** 上执行：

```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final

# 启动 Node 1（Worker）
bash run_docker_mlp.sh 1 192.168.1.100 29600
```

参数说明：
- `1`: node_rank（Worker 节点）
- `192.168.1.100`: master_addr（**和 Master 一样，用 Master 的 IP**）
- `29600`: master_port（**和 Master 一样**）

## ✅ 验证启动成功

### **正常输出应该包含**：

1. **初始化信息**：
```
[Rank 0] DP rank: 0, TP mesh: DeviceMesh(...), DP mesh: DeviceMesh(...)
[Rank 1] DP rank: 0, TP mesh: DeviceMesh(...), DP mesh: DeviceMesh(...)
...
[Rank 15] DP rank: 7, TP mesh: DeviceMesh(...), DP mesh: DeviceMesh(...)
```

2. **所有 16 个 rank 都出现**：
- Rank 0-7: Node 0 的 8 张卡
- Rank 8-15: Node 1 的 8 张卡

3. **训练开始**：
```
[Rank 0] Starting epoch 0
[Rank 0] Processing first batch of epoch 0
[Rank 0] Starting forward pass...
[Rank 0] Loss computed: 1.234567
```

## 🔧 常见问题排查

### **问题 1: Worker 节点连接超时**

```
Error: Timed out initializing process group in store based barrier
```

**解决方案**：
1. 检查 Master IP 是否正确
2. 检查端口 29600 是否被占用：`netstat -tulpn | grep 29600`
3. 检查防火墙设置
4. 确保 Master 节点先启动

### **问题 2: 端口被占用**

```
Address already in use
```

**解决方案**：
```bash
# 停止所有 pytorch Docker 容器
docker stop $(docker ps -q --filter ancestor=nvcr.io/nvidia/pytorch:24.01-py3)

# 或者换一个端口（如 29700）
bash run_docker_mlp.sh 0 192.168.1.100 29700
```

### **问题 3: CUDA 初始化失败**

```
CUDA failed to initialize
```

**解决方案**：
- 脚本已经包含 `--privileged` 等必要参数
- 检查 `nvidia-smi` 是否正常
- 重启 Docker daemon

### **问题 4: NCCL 通信错误**

```
NCCL error: unhandled system error
```

**解决方案**：
- 检查 InfiniBand 连接（如果使用）
- 脚本已经设置了 NCCL 优化参数
- 查看详细日志：修改 `launch_mlp.sh` 中 `NCCL_DEBUG=INFO`

## 📝 文件说明

- **run_docker_mlp.sh**: Docker 启动脚本（在宿主机运行）
- **launch_mlp.sh**: 容器内启动脚本（设置环境变量，启动 torchrun）
- **TP.py**: 主训练脚本（已配置 TP=2）
- **TP_parallel.py**: TP 并行实现
- **DDP.py**: 自定义 DDP 实现
- **MyTrainDataset.py**: 数据集和模型定义

## 🎓 如何修改配置

### **修改 TP_SIZE**

在 `TP.py` 中修改：

```python
tp_size = 2  # 改成其他值，如 4
```

注意：`world_size` 必须能被 `tp_size` 整除

### **修改 batch_size 或 epochs**

在 `TP.py` 的最后：

```python
if __name__ == "__main__":
    main(total_epochs=10, batch_size=32)  # 修改这里
```

### **修改 NCCL 日志级别**

在 `launch_mlp.sh` 中修改：

```bash
export NCCL_DEBUG=WARN  # 改成 INFO 查看详细日志
```

## 🛑 停止训练

```bash
# 停止所有 pytorch Docker 容器
docker stop $(docker ps -q --filter ancestor=nvcr.io/nvidia/pytorch:24.01-py3)

# 或者在运行的终端按 Ctrl+C
```

## 🆚 镜像更新说明

**旧版本（23.10-py3）**：
- 每次启动需要升级 PyTorch（耗时 2-5 分钟）
- 需要网络连接下载包

**新版本（24.01-py3）**：
- PyTorch 2.2+ 已预装，启动速度快
- 无需网络升级，离线可用
- 与单卡基准版本使用相同镜像，便于验证

## 📞 快速启动命令（复制粘贴）

### DGX-011 (Master):
```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final
bash run_docker_mlp.sh 0 $(hostname -I | awk '{print $1}') 29600
```

### DGX-092 (Worker):
```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final
# 替换下面的 MASTER_IP 为实际的 Master IP
bash run_docker_mlp.sh 1 MASTER_IP 29600
```

## 🎉 成功标志

看到以下输出说明启动成功：
- ✅ 所有 16 个 rank 都打印了初始化信息
- ✅ DP rank 从 0 到 7 都存在
- ✅ 开始打印 "Starting epoch 0"
- ✅ 开始打印 Loss 值


