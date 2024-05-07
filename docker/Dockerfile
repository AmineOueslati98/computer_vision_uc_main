FROM --platform=linux/amd64 python:3.8

# Install apt dependencies
RUN apt-get update && apt-get install -y \
  curl \
  git \
  zip \
  libgl1-mesa-dev

# Install protobuf and python dependencies
RUN curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-linux-x86_64.zip && unzip protoc-3.15.6-linux-x86_64.zip
RUN rm protoc-3.15.6-linux-x86_64.zip

RUN cp bin/protoc /usr/local/bin

RUN git clone --depth 1 https://github.com/tensorflow/models

# Compile protobuf configs
RUN (cd models/research && protoc object_detection/protos/*.proto --python_out=.)

RUN (cd models/research && cp object_detection/packages/tf2/setup.py /models/research && python -m pip install .)

RUN rm -r /models

ADD requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt --use-deprecated=legacy-resolver
ADD core/utils/realsr_predictor.py .
RUN cp realsr_predictor.py /usr/local/lib/python3.8/site-packages/ppgan/apps/realsr_predictor.py
