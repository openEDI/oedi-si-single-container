# Build a single layer to save space
RUN DEBIAN_FRONTEND="noninteractive" apt install -y git wget cmake build-essential m4 && \
	apt install -y libaprutil1-dev liblapack-dev libblas-dev libssl-dev libboost-dev libzmq5-dev && \
	mkdir /home/build && cd /home/build && \
	wget http://archive.apache.org/dist/activemq/activemq-cpp/3.9.5/activemq-cpp-library-3.9.5-src.tar.gz && \
	tar -xzf activemq-cpp-library-3.9.5-src.tar.gz && \
	cd activemq-cpp-library-3.9.5 && \
	./configure && make -j 8 && make -j 8 install && \
	cd /home/build && \
	git clone https://github.com/GMLC-TDC/HELICS && cd HELICS && mkdir build && \
	cd build && cmake -DCMAKE_INSTALL_PREFIX=/ -DHELICS_BUILD_CXX_SHARED_LIB=True ../ && make -j 8 install && \
	cd /home/build && \
	git clone --depth 1 --branch GADAL.0.6 https://github.com/GRIDAPPSD/gridappsd-state-estimator && \
	cd gridappsd-state-estimator && \
	git clone https://github.com/GRIDAPPSD/SuiteSparse && \
	git clone https://github.com/GRIDAPPSD/json && \
	cd /home/build/gridappsd-state-estimator && \
	LD_LIBRARY_PATH=/home/build/gridappsd-state-estimator/SuiteSparse/lib/ make -C SuiteSparse LAPACK=-llapack BLAS=-lblas && \
	sed -i "s/local_feeder/feeder/g" "/home/build/gridappsd-state-estimator/state-estimator/include/PlatformInterfaceGADAL.hpp" && \
	make -j 8 -C state-estimator && mkdir /home/output && cd /home/build && \
	rm -r activemq-cpp-library-3.9.5-src.tar.gz activemq-cpp-library-3.9.5 HELICS gridappsd-state-estimator/json gridappsd-state-estimator/SuiteSparse
