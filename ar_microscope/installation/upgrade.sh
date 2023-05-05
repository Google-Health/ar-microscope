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
#!/bin/bash

# Before running this script, the upgrade folder should be copied to the ARM's
# home directory.
HOME_DIR=/home/arm
# Directory copied from installation media to the home directory.
UPGRADE_DIR=$HOME_DIR/arm_upgrade
DEPENDENCIES_DIR=$UPGRADE_DIR/dependency_packages

# First download cuDNN packages and CUDA toolkit. Note that the order matters.
# See the ARM's initial_setup document for more details on CUDA installation.
sudo dpkg -i $DEPENDENCIES_DIR/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i $DEPENDENCIES_DIR/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i $DEPENDENCIES_DIR/cuda-repo-ubuntu1804-11-2-local_11.2.2-460.32.03-1_amd64.deb
# Add dependencies folder to apt sources.
echo "deb [trusted=yes] file:${DEPENDENCIES_DIR} ./" \
                  | sudo tee -a /etc/apt/sources.list
sudo apt-get update
sudo apt -y install cuda-11-2

# Rerun ldconfig so that the newly install libraries can be found.
sudo ldconfig

# Reinstall ar-microscope
sudo apt -y remove ar-microscope
sudo apt -y install $UPGRADE_DIR/ar-microscope_*_amd64.deb

mkdir -p $HOME_DIR/arm_logs/event_logs
mkdir -p $HOME_DIR/arm_logs/snapshots
# Add snapshots link on desktop for easier access by doctors.
ln -s $HOME_DIR/arm_logs/snapshots $HOME_DIR/Desktop/arm_snapshots

# Run apt-hold command to prevent upgrades to critical packages.
apt-mark hold cuda-11-2 libcudnn8 libcudnn8-dev libnccl-dev libopencv-core3.2 \
    libopencv-imgcodecs3.2 libopencv-imgproc3.2 qt5-default

echo "Please restart your machine to finish the CUDA installation."
