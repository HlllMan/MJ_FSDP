#!/bin/bash
# 构建自定义 PyTorch Docker 镜像
# 用法: bash build_docker.sh

IMAGE_NAME="pytorch-tp-dp:latest"
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile"

echo "==================== 构建 Docker 镜像 ===================="
echo "镜像名称: ${IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "================================================="
echo ""

# 检查 Dockerfile 是否存在
if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "错误: Dockerfile 不存在: ${DOCKERFILE_PATH}"
    exit 1
fi

# 构建镜像
echo "开始构建镜像（这可能需要 5-10 分钟）..."
docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "==================== 构建成功 ===================="
    echo "镜像名称: ${IMAGE_NAME}"
    echo ""
    echo "验证镜像:"
    docker run --rm --gpus all ${IMAGE_NAME} python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    echo ""
    echo "使用方法:"
    echo "  修改 docker/run_docker_mlp.sh 中的镜像名称为: ${IMAGE_NAME}"
    echo "================================================="
else
    echo ""
    echo "==================== 构建失败 ===================="
    exit 1
fi

