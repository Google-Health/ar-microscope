# ARM Host Initial Setup

In order to get all of ARM's software dependencies in place you

1. Run the following commands in terminal to create a User called `arm`, set a
password, and become the `arm` user:

  ```shell
  foo@bar$ sudo useradd -m arm
  foo@bar$ sudo passwd arm
  foo@bar$ su - arm
  ```

2. Run the following commands to update apt repositories and fix broken
packages:

  ```shell
  foo@bar$ sudo apt update
  foo@bar$ sudo apt --fix-broken install
  ```

3. Install DijSDK, this package is available through Jenoptik. Run this in the
directory the `.deb` is in:

  ```shell
  foo@bar$ sudo dpkg -i DijSDK-2.1.0.1261-Linux.deb
  ```

4. Install ARM's OpenCV & QT dependencies:

  ```shell
  foo@bar$ sudo apt -y install libopencv-core3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 qt5-default
  ```

5. Install NVIDIA Software. These steps assume you have a fresh Ubuntu 18.04
machine. If you have a previous installation of CUDA or NVIDIA's drivers these
steps may not work for you.

  1. Install the CUDA Toolkit:

    ```shell
    foo@bar$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    foo@bar$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    foo@bar$ wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
    foo@bar$ sudo dpkg -i cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
    foo@bar$ sudo apt-key add /var/cuda-repo-ubuntu1804-11-2-local/7fa2af80.pub
    foo@bar$ sudo apt-get update
    foo@bar$ sudo apt-get -y install cuda-11-2
    ```

  2. Validate the installation was successful by running:

    ```shell
    foo@bar$ nvidia-smi
    ```

  3. Install cuDNN v8.1.1:

    ```shell
    foo@bar$ wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/Ubuntu18_04-x64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
    foo@bar$ wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/Ubuntu18_04-x64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
    foo@bar$ sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
    foo@bar$ sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
    ```

6. Create some required folders:

  ```shell
  foo@bar$ export ARM_DIR=/home/arm
  foo@bar$ mkdir -p $ARM_DIR/arm_logs/event_logs
  foo@bar$ mkdir -p $ARM_DIR/arm_logs/snapshots
  foo@bar$ mkdir -p $ARM_DIR/Desktop/arm_snapshots
  ```

7. Add snapshots link on desktop for easier access:

```shell
foo@bar$ ln -s $ARM_DIR/arm_logs/snapshots $ARM_DIR/Desktop/arm_snapshots
```


8. **OPTIONAL** If your plan to keep your ARM machine connected to the internet
make sure to run the `apt-hold` command to prevent upgrades to critical
packages.

```shell
foo@bar$ apt-mark hold cuda-11-2 libcudnn8 libcudnn8-dev libnccl-dev libopencv-core3.2 libopencv-imgcodecs3.2 libopencv-imgproc3.2 qt5-default
```

9. Make sure to reboot your computer by running the following:

```shell
foo@bar$ sudo reboot
```

10. **OPTIONAL** If you have already built ARM's software install it by running:

```shell
foo@bar$ sudo apt -y install ar-microscope_*_amd64.deb
```

11. Make sure to reboot your machine after the above command if it's your first
time installing ARM.
