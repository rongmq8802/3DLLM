# FROM swr.cn-central-221.ovaijisuan.com/wair/pytorch_1_8_1:pytorch_1.8.1-cuda_11.1-py_3.7-ubuntu_18.04-x86_64-20221111

FROM python:3.8.5-slim-buster

# Install dependencies
USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy project
COPY . /app
WORKDIR /app

# 安装point transformer 对应的环境
WORKDIR /app/lavis/models/blip2_models/libs/pointops
RUN python setup.py install

# 运行服务
WORKDIR /app
COPY --chmod=a+x start.sh .
CMD ["/bin/bash", "start.sh"]
