# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Dockerfile for building ARM.
#
# Host should install Nvidia driver and Docker Nvidia runtime.
# Nvidia runtime can be set up by following README of:
# https://github.com/NVIDIA/nvidia-container-runtime
#
# To run the image, specify Nvidia runtime to enable GPU support.
# Enable privileged mode so that Docker container can access USB camera device.
# Share X11 window sockets for the container to access display.
# Example:
#   xhost local:  # Allow access to X server through UNIX domain socket.
#   docker container run --runtime=nvidia -it --rm --privileged  \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  \
#       --name arm <image name> bash

# During development, tensorflow/tensorflow:devel-gpu may need to be used when
# there are TF features used (in ARM or the models) that have not been open
# sourced yet.
FROM arm/tf:2.7.0

# Install required and utility tools.
RUN  \
    apt update &&  \
    apt -y install  \
        curl wget less vim emacs tmux \
        pkg-config  \
        zip unzip zlib1g-dev  \
        gcc g++ git  \
        libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev \
        qt5-default  \
        libraw1394-dev  \
        libsm6 libxext6 libxrender-dev  \
        fakeroot

# For git tools.
RUN pip install future

ARG UNIX_USER="arm-user"
ARG USER_ID=1001
ARG GROUP_ID=1001
RUN groupadd -g $GROUP_ID $UNIX_USER
RUN useradd -m -s /bin/bash -u $USER_ID -g $GROUP_ID $UNIX_USER

USER $UNIX_USER
ARG HOMEDIR=/home/$UNIX_USER
WORKDIR $HOMEDIR
ARG ARMDIR=$HOMEDIR/ar_microscope
COPY ar_microscope $ARMDIR
COPY LICENSE $ARMDIR

USER root

ARG TFDIR=/tensorflow_src
WORKDIR $TFDIR
RUN sed -i 's/\[\"\:internal\"\]/\[\"\/\/visibility\:public\"\]/' $TFDIR/tensorflow/BUILD

# For more information on supported compute capabilities see: https://developer.nvidia.com/cuda-gpus.
# 6.1: GTX 10xx and Titan X series
# 7.5: RTX 20xx and Titan RTX series
ENV TF_CUDA_COMPUTE_CAPABILITIES="6.1,7.5"
RUN ./configure

ENV TMP="/tmp"

RUN echo "import $TFDIR/.bazelrc" > $ARMDIR/.bazelrc

# Install Jenoptik SDK.
COPY DijSDK-2.1.0.1261-Linux.deb /tmp
RUN dpkg -i /tmp/DijSDK-2.1.0.1261-Linux.deb
RUN ln -s ../Jenoptik/DijSDK /usr/local/include/jenoptik
RUN ln -s ../Jenoptik/DijSDK/lib/libDijSDK.so.2.1.0 /usr/local/lib
RUN ln -s libDijSDK.so.2.1.0 /usr/local/lib/libDijSDK.so.2
RUN ln -s libDijSDK.so.2 /usr/local/lib/libDijSDK.so

# Build ar_microscope
WORKDIR $ARMDIR
RUN $ARMDIR/build/build.sh

USER $UNIX_USER
WORKDIR $HOMEDIR
