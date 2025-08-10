# Jetson-friendly base with CUDA/cuDNN/TRT + PyTorch
FROM dustynv/l4t-pytorch:r36.4.0
# Avoid interactive installs
ENV DEBIAN_FRONTEND=noninteractive

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY app /workspace/app

# Default shell
CMD ["bash"]
