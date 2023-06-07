FROM ubuntu:20.04

RUN apt update

# Build a single layer to save space
RUN apt install -y python3-pip git nano python-is-python3 && \
	mkdir /home/oedisi && cd /home/oedisi && \
	git clone https://github.com/openEDI/sgidal-example.git && \
	git clone https://github.com/openEDI/GADAL.git && \
	git clone https://github.com/openEDI/gadal-ieee123.git && \
	cd /home/oedisi/GADAL && git checkout v0.2.4 && pip install -e . && \
	cd /home/oedisi/sgidal-example && pip install -r requirements.txt
