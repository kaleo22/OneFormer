FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ARG USERNAME=user
ARG WORKDIR=/workspace/OneFormer
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

RUN conda create --name oneformer python=3.8 -y

RUN conda run -n oneformer conda install pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y && \
    conda run -n oneformer pip install -U opencv-python && \
    git clone https://github.com/kaleo22/OneFormer.git && \
    cd OneFormer && \
    conda run -n oneformer python tools/setup_detectron2.py && \
    conda run -n oneformer pip install git+https://github.com/cocodataset/panopticapi.git && \
    conda run -n oneformer pip install git+https://github.com/mcordts/cityscapesScripts.git && \
    conda run -n oneformer pip install "pip<24.1" && \
    conda run -n oneformer pip install -r requirements.txt

RUN conda run -n oneformer pip3 install wandb
ENV TORCH_CUDA_ARCH_LIST="8.6"
WORKDIR /OneFormer/oneformer/modeling/pixel_decoder/ops
RUN ls -l
RUN conda run -n oneformer sh make.sh
