# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="convlab"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip build-essential
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" ]

COPY . .
RUN pip install -e .

RUN ln -f -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]
