FROM ubuntu:20.04
RUN apt update && apt upgrade -y
RUN apt install -y python3-pip
RUN pip install helics==3.3.0
RUN DEBIAN_FRONTEND="noninteractive" apt install -y git pkg-config coinor-libipopt-dev
RUN apt install -y libblas-dev liblapack-dev
RUN pip3 install cyipopt

# now install bleeding edge versions for ipopt (instead of old Ipopt 3.11.9 from coinor-libipopt-dev)
# and link it with bleeding edge version of Mumps
RUN apt install -y wget
RUN mkdir -p /home/app/deps/dopf_pnnl/lib && \
cd /home/app/deps/dopf_pnnl/ && \
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && \
cd ThirdParty-Mumps/ && \
./get.Mumps && ./configure && make && make install && \
cp -r .libs/* /home/app/deps/dopf_pnnl/lib

RUN mkdir -p /home/app/deps/dopf_pnnl/ipoptBuild &&\
cd /home/app/deps/dopf_pnnl/ && git clone https://github.com/coin-or/Ipopt.git && \
cd /home/app/deps/dopf_pnnl/Ipopt && ./configure --prefix /home/app/deps/dopf_pnnl/ipoptBuild && make && \
make install && cp -r /home/app/deps/dopf_pnnl/ipoptBuild/lib/* /home/app/deps/dopf_pnnl/lib && \
ln -s /home/app/deps/dopf_pnnl/lib/libipopt.so /home/app/deps/dopf_pnnl/lib/libipopt.so.1

RUN rm -r /home/app/deps/dopf_pnnl/ipoptBuild /home/app/deps/dopf_pnnl/Ipopt /home/app/deps/dopf_pnnl/ThirdParty-Mumps && \
mkdir -p /home/app/profiles/ && \
echo "export LD_LIBRARY_PATH=/home/app/deps/dopf_pnnl/lib:$LD_LIBRARY_PATH" >> /home/app/profiles/dopf_pnnl_rc

COPY federates/ /home/app/federates
COPY runner/ /home/app/runner
COPY logs/ /home/app/logs

WORKDIR /home/app/runner
ENTRYPOINT /home/app/runner/run.sh

