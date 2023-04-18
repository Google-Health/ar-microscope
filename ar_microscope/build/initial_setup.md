# ARM Host Initial Setup

In order to get all of ARM's dependencies in place, run the following
commands in terminal:

```
sudo apt update
sudo dpkg -i DijSDK-2.1.0.1261-Linux.deb
sudo apt --fix-broken install

sudo apt -y install libopencv-core3.2 libopencv-imgcodec3.2 libopencv-imgproc3.2 qt5-default

# Install nvidia driver
sudo ubuntu-drivers autoinstall
sudo reboot

export CUDA=11
export CUDNN=8.0.2.39-1
sudo apt -y install cuda-$CUDA-0
sudo apt install libcudnn8=$CUDNN+cuda$CUDA.0 libcudnn8-dev=$CUDNN+cuda$CUDA.0

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb

# Install Google software
sudo apt -y install ar-microscope_*_amd64.deb

# Create some required folders.
export HOME_DIR=/home/arm
mkdir -p $HOME_DIR/arm_logs/event_logs
mkdir -p $HOME_DIR/arm_logs/snapshots

# Add snapshots link on desktop for easier access.
ln -s $HOME_DIR/arm_logs/snapshots $HOME_DIR/Desktop/arm_snapshots

# Run apt-hold command to prevent upgrades to critical packages.
apt-mark hold cuda-11-0 libcudnn8 libcudnn8-dev libnccl-dev libopencv-core3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 qt5-default
```