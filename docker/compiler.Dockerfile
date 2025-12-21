# ==============================================================================
# xInfer Universal Compiler Dockerfile
# Targets: 15+ Platforms (NVIDIA, Intel, AMD, Rockchip, Hailo, Qualcomm, etc.)
# ==============================================================================

FROM ubuntu:22.04

# --- 1. System Essentials ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    python3-pip \
    python3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# --- 2. NVIDIA TensorRT (NVIDIA Target) ---
# Note: We install the development headers for model conversion
RUN v=$(lsb_release -rs | tr -d '.') && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y tensorrt-dev tensorrt-libs python3-libnvinfer-dev

# --- 3. Google Edge TPU (Google Target) ---
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y edgetpu-compiler

# --- 4. Intel OpenVINO & CoreML (Intel & Apple Targets) ---
RUN pip3 install --no-cache-dir \
    openvino-dev[onnx,tensorflow,pytorch] \
    coremltools \
    onnx \
    onnxsim

# --- 5. Rockchip RKNN-Toolkit2 (Rockchip Target) ---
# Installing dependencies for RKNN
RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install --no-cache-dir \
    rknn-toolkit2 \
    && pip3 install --no-cache-dir debugpy # For remote debugging xInfer scripts

# --- 6. Hailo Dataflow (Hailo Target) ---
RUN pip3 install --no-cache-dir hailo_sdk_client

# --- 7. Microchip VectorBlox (FPGA Target) ---
RUN git clone https://github.com/Microchip-FPGA-Tools/VectorBlox /opt/vectorblox && \
    pip3 install --no-cache-dir -r /opt/vectorblox/requirements.txt
ENV VECTORBLOX_SDK=/opt/vectorblox

# --- 8. PROPRIETARY ZONE (Qualcomm, Ambarella, MediaTek) ---
# These vendors require manual SDK downloads. 
# Instructions: Place your downloaded SDKs in 'xinfer/third_party/sdks/' before building.

# Qualcomm QNN Slot
ARG QNN_SDK_PATH=./third_party/sdks/qualcomm_qnn
COPY ${QNN_SDK_PATH}* /opt/qualcomm/qnn/
ENV QNN_SDK_ROOT=/opt/qualcomm/qnn

# Ambarella CVFlow Slot
ARG AMBA_SDK_PATH=./third_party/sdks/ambarella
COPY ${AMBA_SDK_PATH}* /opt/ambarella/
ENV AMBA_CVFLOW_SDK=/opt/ambarella

# MediaTek NeuroPilot Slot
ARG MTK_SDK_PATH=./third_party/sdks/mediatek
COPY ${MTK_SDK_PATH}* /opt/mediatek/
ENV NEUROPILOT_SDK=/opt/mediatek

# --- 9. Final Environment Setup ---
WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/opt/qualcomm/qnn/lib/python"

# Label the image
LABEL maintainer="xInfer Team"
LABEL description="Unified AI Compiler environment for 15+ Edge Platforms"

# Default command: Start xInfer-Doctor
CMD ["/bin/bash", "-c", "python3 /workspace/tools/xinfer-doctor && /bin/bash"]