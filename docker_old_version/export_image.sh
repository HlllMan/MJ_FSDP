#!/bin/bash
# 导出 Docker 镜像
# 用法: bash export_image.sh [output_file]

IMAGE_NAME="pytorch-tp-dp:latest"
OUTPUT_FILE=${1:-"pytorch-tp-dp-latest.tar"}

echo "==================== 导出 Docker 镜像 ===================="
echo "镜像名称: ${IMAGE_NAME}"
echo "输出文件: ${OUTPUT_FILE}"
echo "================================================="
echo ""

# 检查镜像是否存在
if ! docker images | grep -q "pytorch-tp-dp"; then
    echo "错误: 镜像 ${IMAGE_NAME} 不存在"
    echo "请先运行: bash build_docker.sh"
    exit 1
fi

# 导出镜像
echo "正在导出镜像（这可能需要几分钟）..."
docker save -o "${OUTPUT_FILE}" "${IMAGE_NAME}"

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo ""
    echo "==================== 导出成功 ===================="
    echo "镜像文件: ${OUTPUT_FILE}"
    echo "文件大小: ${FILE_SIZE}"
    echo ""
    echo "传输到另一个节点:"
    echo "  scp ${OUTPUT_FILE} user@node1:/path/to/destination/"
    echo ""
    echo "在另一个节点导入:"
    echo "  docker load -i ${OUTPUT_FILE}"
    echo "================================================="
else
    echo ""
    echo "==================== 导出失败 ===================="
    exit 1
fi






