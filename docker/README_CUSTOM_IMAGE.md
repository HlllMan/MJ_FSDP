# 自定义 Docker 镜像使用指南

## 概述

自定义 Docker 镜像 `pytorch-tp-dp:latest` 基于 `nvcr.io/nvidia/pytorch:24.01-py3`，并预装了最新版本的 PyTorch（>=2.2.0，支持 `device_mesh` API）。

## 优势

- ✅ **无需每次升级**：镜像已包含最新 PyTorch，启动更快
- ✅ **版本稳定**：构建时固定版本，避免运行时版本变化
- ✅ **离线可用**：镜像包含所有依赖，不依赖网络

## 构建镜像

### 方法 1：使用构建脚本（推荐）

```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final/docker
bash build_docker.sh
```

构建时间：约 5-10 分钟（取决于网络速度）

### 方法 2：手动构建

```bash
cd /public/data0/HOME/jdnlp1004/miaoji.norman/FSDP/final/docker
docker build -f Dockerfile -t pytorch-tp-dp:latest .
```

## 使用自定义镜像

### 启动训练（使用自定义镜像）

```bash
# 节点 0 (Master)
bash docker/run_docker_mlp.sh 0 172.17.226.11 29600 --use-custom-image

# 节点 1 (Worker)
bash docker/run_docker_mlp.sh 1 172.17.226.11 29600 --use-custom-image
```

### 启动训练（使用官方镜像，自动升级）

```bash
# 节点 0 (Master)
bash docker/run_docker_mlp.sh 0 172.17.226.11 29600

# 节点 1 (Worker)
bash docker/run_docker_mlp.sh 1 172.17.226.11 29600
```

## 验证镜像

```bash
# 检查镜像是否存在
docker images | grep pytorch-tp-dp

# 验证 PyTorch 版本
docker run --rm --gpus all pytorch-tp-dp:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 更新镜像

如果需要更新到更新的 PyTorch 版本：

1. 修改 `Dockerfile` 中的安装命令（如果需要特定版本）
2. 重新构建镜像：
   ```bash
   bash build_docker.sh
   ```

## 镜像大小

- 基础镜像：约 10-15 GB
- 自定义镜像：约 12-18 GB（包含 PyTorch）

## 注意事项

1. **首次构建**：需要下载基础镜像和 PyTorch，可能需要较长时间
2. **存储空间**：确保有足够的磁盘空间（至少 20GB）
3. **网络连接**：构建时需要网络连接（使用清华镜像加速）

## 故障排除

### 构建失败

- 检查网络连接
- 检查磁盘空间：`df -h`
- 查看详细错误：`docker build --progress=plain -f Dockerfile -t pytorch-tp-dp:latest .`

### 镜像不存在

- 确认构建成功：`docker images | grep pytorch-tp-dp`
- 如果不存在，重新构建：`bash build_docker.sh`






