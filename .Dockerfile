FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ARG USERNAME=user
ARG WORKDIR=/workspace/OneFormer
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

RUN conda install -n base conda=4.12.0 -y
RUN conda create --name oneformer python=3.8 -y \
    conda activate oneformer

RUN git clone https://github.com/SHI-Labs/OneFormer.git \
    && cd OneFormer \
    conda install pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge \
    && pip3 install -U opencv-python \
    && python tools/setup_detectron2.py \
    && pip3 install git+https://github.com/cocodataset/panopticapi.git \
    && pip3 install git+https://github.com/mcordts/cityscapesScripts.git \
    && pip3 install -r requirements.txt

RUN pip3 install wandb \
    && wandb login
WORKDIR /workspace/OneFormer/oneformer/modeling/pixel_decoder/ops
RUN sh make.sh
