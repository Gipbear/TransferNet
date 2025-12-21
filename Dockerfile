# 有了 Docker，可以使用任意老版本的 PyTorch 和 CUDA，而不会影响其他的项目
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

#设置非交互构建
ARG DEBIAN_FRONTEND=noninteractive
#定义时区参数
ENV TZ=Asia/Shanghai
#设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo '$TZ' > /etc/timezone

# Dockerfile 中 APT 更换 apt 源
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y git gdb vim curl wget tmux zip cmake ffmpeg libsm6 libxext6 htop nload
    # && rm -rf /var/lib/apt/lists/*

# environment.yml 需要放在 Dockerfile 同一目录下
# COPY environment.yml /tmp/environment.yml
# RUN conda env create -f /tmp/environment.yml
# RUN conda init bash
COPY requirements.txt /tmp/requirements.txt

# 除了 Conda，也可以直接用 pip 安装 Python 包，Docker 已经提供了环境隔离
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip3 config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip3 install -r /tmp/requirements.txt \
    && pip3 install h5py einops tqdm matplotlib tensorboard torch-tb-profiler ninja scipy pydantic ogb \
    && pip3 install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html \
    && pip3 install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html


CMD [ "python3" ]
