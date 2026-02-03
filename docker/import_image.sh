#!/bin/bash
# 导入 Docker 镜像
# 用法: bash import_image.sh [image_file]

IMAGE_FILE=${1:-"pytorch-tp-dp-latest.tar"}

echo "==================== 导入 Docker 镜像 ===================="
echo "镜像文件: ${IMAGE_FILE}"
echo "================================================="
echo ""

# 检查文件是否存在
if [ ! -f "${IMAGE_FILE}" ]; then
    echo "错误: 镜像文件不存在: ${IMAGE_FILE}"
    echo ""
    echo "请先从另一个节点传输镜像文件，或使用:"
    echo "  scp user@node0:/path/to/${IMAGE_FILE} ."
    exit 1
fi

FILE_SIZE=$(du -h "${IMAGE_FILE}" | cut -f1)
echo "文件大小: ${FILE_SIZE}"
echo ""

# 导入镜像
echo "正在导入镜像（这可能需要几分钟）..."
docker load -i "${IMAGE_FILE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "==================== 导入成功 ===================="
    echo "验证镜像:"
    docker images | grep pytorch-tp-dp
    echo ""
    echo "测试镜像:"
    docker run --rm --gpus all pytorch-tp-dp:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    echo "================================================="
else
    echo ""
    echo "==================== 导入失败 ===================="
    exit 1
fi






