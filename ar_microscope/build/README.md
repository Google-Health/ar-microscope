# Installing ARM

WARNING: ARM uses proprietary Jenoptik drivers. Without access to these drivers,
the build process will not work.

NOTE: Ensure you have installed docker first:
[Docker](https://docs.docker.com/engine/install/).

## Setting Up Your ARM Machine

You can set up the ARM machine later, once you have the ARM Debian. However, to
install the Debian you build in the subsequent steps, you will need to set up an
ARM machine that meets ARM's software requirements. Ensure your ARM machine has
a CUDA enabled GPU by running `lspci | grep -i nvidia`. If you see an NVIDIA
device listed you can use the `initial_setup.md` document, which is available [here](https://github.com/Google-Health/ar-microscope/blob/main/ar_microscope/build/initial_setup.md),
to set up a machine that meets ARM's software requirements. You only need to run
through these steps once. Be sure to reboot your machine once the setup is
complete.

## Cloning and Building the ARM binary
ARM builds happen within a Docker container. This means that you do not need to
build ARM on the same machine that you will run the ARM software on.

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
    the <deb_package_name> below.

    ```shell
    foo@bar$ cat $ARM_DRIVERS_DIR/ar_microscope/deb_package/DEBIAN/control
    ```

4. The package can be copied out of the container with the following command:

  ```shell
  foo@bar$ docker cp arm-latest:/home/$USER/ar_microscope/<deb_package_name> $ARM_DRIVERS_DIR
   ```

5. You can install the debian package on the ARM machine by running the following
command:

  ```shell
  foo@bar$ sudo dpkg -i $ARM_DRIVERS_DIR/<deb_package_name>
  ```

### Rebuilding ARM

As long as you aren't modifying the Dockerfiles, or introducing new dependencies
, ARM can be rebuilt via a script that can save significant time. The script is
available [here](https://github.com/Google-Health/ar-microscope/blob/main/ar_microscope/build/rebuild.sh).
You can run the following command to execute the script:

```shell
  foo@bar$ bash ~/arm/arm_drivers/ar_microscope/build/rebuild.sh <deb_package_version> <container_name_override> 
```
`deb_package_version` is the version from the control file above, and
`container_name_override` is an optional override parameter in case your
container is not named `arm-latest`.


