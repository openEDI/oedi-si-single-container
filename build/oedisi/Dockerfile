FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

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
    && sed -i 's/pydantic/pydantic>=1.7,<2/g' requirements.txt \
    && pip install -r requirements.txt

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
