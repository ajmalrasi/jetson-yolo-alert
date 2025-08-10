# Jetson-friendly base with CUDA/cuDNN/TRT + PyTorch
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
# Avoid interactive installs
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY app /workspace/app

# Default shell
CMD ["bash"]
