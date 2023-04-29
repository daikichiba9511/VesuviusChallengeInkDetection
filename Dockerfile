FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV LC_ALL="C.UTF-8" LESSCHARSET="utf-8"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspace/working

WORKDIR /workspace/working

# nvidia-dockerのGPGキーが更新されたから
# Ref: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && useradd -m builder



RUN apt update && apt upgrade -y \
    && DEBIAN_FRONTEND=nointeractivetzdata \
    TZ=Asia/Tokyo \
    apt install -y \
    make \
    cmake \
    unzip \
    git \
    curl \
    wget \
    tzdata \
    locales \
    sudo \
    tar \
    gcc \
    g++ \
    libgl1-mesa-dev \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y \
    python3-pip \
    python3.9-dev \
    python3.9 \
    python3.9-distutils \
    python-is-python3 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1



# COPY ./ ./
