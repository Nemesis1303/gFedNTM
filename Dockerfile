# Ubuntu version: 20.04, 18.04
ARG VARIANT=22.04
FROM ubuntu:${VARIANT}

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends tzdata \
    python3.10 \
    python3-distutils \
    python3-pip \
    python3-apt \
    build-essential \
    gcc \
    python3-dev \
    python-is-python3 \
    google-perftools

# Clean up the package cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

VOLUME /data

COPY . /workspace

RUN rm -rf /workspace/static/datasets/dataset_federated/iter_0

RUN python3 -m pip install -r /workspace/requirements.txt