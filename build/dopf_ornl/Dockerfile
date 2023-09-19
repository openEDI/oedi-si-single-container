RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install -y octave octave-statistics && \
	pip install mat4py oct2py statistics
RUN echo "addpath(genpath('/home/dopf_ornl'))" >> ~/.octaverc
