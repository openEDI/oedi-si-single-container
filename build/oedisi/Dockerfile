FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# INSTALL PYTHON
#===========================================
ARG PYTHON_VERSION=3.11.9
RUN apt-get update -y && \
    apt-get install -y \
	gcc \
	make \
	wget \
	zlib1g-dev \
	libffi-dev \
	libssl-dev \
	libbz2-dev \
	liblzma-dev

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --with-ensurepip=install --enable-shared && make && make install && \
    ldconfig && \
    ln -sf python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# Preconfigure tzdata to avoid interactive prompt
RUN echo "tzdata tzdata/Areas select America" | debconf-set-selections \
    && echo "tzdata tzdata/Zones/America select New_York" | debconf-set-selections

RUN apt update \
    && apt install -y tzdata python3-pip git nano python-is-python3 \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && mkdir /home/oedisi \
    && cd /home/oedisi \    
    && git clone --depth 1 --branch v0.4.1 https://github.com/openEDI/oedisi-example.git \	
    && git clone --depth 1 --branch v2.0.2 https://github.com/openEDI/oedisi.git \
    && git clone https://github.com/openEDI/oedisi-ieee123.git \
    && cd /home/oedisi/oedisi \    
    && pip install . \
    && cd /home/oedisi/oedisi-example \        
    && pip install -r requirements.txt

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN rm -rf /app/Python-$PYTHON_VERSION.tgz
