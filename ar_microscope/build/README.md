# Installing ARM

WARNING: ARM uses proprietary Jenoptik drivers. Without access to these drivers,
the build process will not work.

NOTE: Ensure you have installed docker first:
[Docker](https://docs.docker.com/engine/install/).

## Cloning and Building the ARM binary

### Cloning ARM's Drivers

1. Run the following commands to make an installation directory for ARM:

  ```shell
  foo@bar$ export ARM_DIR=$HOME/arm
  foo@bar$ export ARM_DRIVERS_DIR=$ARM_DIR/arm_drivers
  foo@bar$ export ARM_MODELS_DIR=$ARM_DIR/arm_models
  foo@bar$ mkdir $ARM_DIR
  foo@bar$ mkdir $ARM_DRIVERS_DIR
  foo@bar$ mkdir $ARM_MODELS_DIR
  foo@bar$ cd $ARM_DRIVERS_DIR
  ```

2. Clone ARM into the directory you just created:

  ```shell
  foo@bar$ git clone https://github.com/Google-Health/ar-microscope.git $ARM_DRIVERS_DIR
  ```

3. Install ARM's dependencies. **This step is not yet possible!**

  ```shell
  foo@bar$ mv $JENOPTIK_SDK_PATH/DijSDK-2.1.0.1261-Linux.deb $ARM_DRIVERS_DIR.
  ```

WARNING: If you do not have access to ARM's models, or have not added your own,
the build process will fail at the end. The models are not yet available, but
will be soon via a request form.


### Build the ARM Binary using docker

1.  Build the base gpu docker image.

  ```shell
    foo@bar$ docker build --tag arm/tf:2.7.0  \
    -f $ARM_DRIVERS_DIR/ar_microscope/build/gpu.Dockerfile  \
    $ARM_DRIVERS_DIR/ar_microscope/build/
    ```
2. Build the ARM docker image.

  ```shell
    foo@bar$ docker build  \
               --tag arm:latest  \
               --build-arg UNIX_USER=$USER  \
               --build-arg USER_ID=$(id -u)  \
               --build-arg GROUP_ID=$(id -g)  \
               -f $ARM_DRIVERS_DIR/ar_microscope/build/Dockerfile $ARM_DRIVERS_DIR
    ```

Building the docker image takes ~1 hour, and the image can be saved so that
re-building the binary with new models or minor code changes can be much
faster.

### Copy the ARM Binary out of the container.

In the previous steps, we built a Docker image and within that image, we
built a debian package. That debian package contains the necessary files and
executables to run ARM. The debian package is located within the container at
`/home/$USER/ar_microscope/ar-microscope_*_amd64.deb`. We need to copy it out of
the container onto our host machine so we can run ARM. To determine the exact
debian package name you can follow the below steps.

1. First, we need to start the container that contains the debian package and
then let it continue running in the background. Run the below command:

  ```shell
  foo@bar$ docker run -it --name arm-latest arm:latest
  ```

If you see an error like *The container name "/arm-latest" is already in use by container ...*
you may need to run `docker stop arm-latest` and `docker rm arm-latest` first.

2. Type Ctrl+P, Ctrl+Q to detach from the container and run it in the background.

3. Get the version from the control file.  You can replace the `*` in
    `ar-microscope_*_amd64.deb` with the `Version` from the control file for
    the <deb package name> below.

    ```shell
    foo@bar$ cat $ARM_DRIVERS_DIR/ar_microscope/deb_package/DEBIAN/control
    ```

4. The package can be copied out of the container with the following command:

  ```shell
  foo@bar$ docker cp arm-latest:/home/$USER/ar_microscope/<deb package name> $ARM_DRIVERS_DIR
   ```

5. You can install the debian package on the ARM machine by running the
following command:

  ```shell
  foo@bar$ sudo dpkg -i $ARM_DRIVERS_DIR/<deb package name>
  ```
