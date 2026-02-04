# 自定义 PyTorch 镜像 - 包含最新版本的 PyTorch（支持 device_mesh）
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_NO_CACHE_DIR=0
# 限制 OpenBLAS 线程数，避免线程创建失败
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# 升级 pip 并安装最新版本的 PyTorch
# 禁用进度条和交互式输出，避免线程问题
RUN pip install --upgrade pip --no-cache-dir --quiet && \
    pip install --upgrade torch torchvision torchaudio \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-cache-dir \
    --quiet \
    --progress-bar off || true

# 验证安装（构建时没有 GPU，只检查版本）
RUN python -c "import torch; \
version_str = torch.__version__.split('+')[0]; \
version_parts = [int(x) for x in version_str.split('.')[:3]]; \
print('PyTorch version:', torch.__version__); \
print('Version parts:', version_parts); \
assert version_parts[0] > 2 or (version_parts[0] == 2 and version_parts[1] >= 2), \
    f'PyTorch version {version_parts} < 2.2.0'; \
print('PyTorch version >= 2.2.0')"

# 设置默认命令
CMD ["/bin/bash"]

