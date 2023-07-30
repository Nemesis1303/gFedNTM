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
    python-is-python3

# Clean up the package cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


# Install needed packages. Use a separate RUN statement to add your own dependencies.
#RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Base 
#RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends build-essential git

# Python 3
#RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends python3 python3-pip

RUN pip3 install --upgrade pip setuptools wheel

#RUN apt install python3-dev -y

VOLUME /data

COPY . /workspace

RUN python3 -m pip install -r /workspace/requirements.txt

