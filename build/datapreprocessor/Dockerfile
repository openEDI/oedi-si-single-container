WORKDIR /home/datapreprocessor
RUN pip install scikit-learn>=1.1.2 tqdm>=4.64.0 py7zr>=0.19.0 tensorflow>=2.17.0 keras>=3.4.1
COPY datapreprocessor /home/.
RUN cd /home/datapreprocessor && pip install -e .
RUN mkdir -p /home/datapreprocessor/logs
RUN touch /home/datapreprocessor/logs/distmodel.log
