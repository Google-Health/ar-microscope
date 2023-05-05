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

# Convenience script for rebuilding ARM by using an existing ARM container. ARM
# is rebuilt within the provided container and the deb package and underlying
# code are scp-ed to the ARM's machine.
#
# This script assumes that the existing ARM container is up and running, which
# can be done via:
#   1. docker run -it --name arm-latest arm:latest
#   2. Ctrl+P, Ctrl+Q to detach from the container while keeping it running.
#
# Sample invocation:
# bash ~/arm/arm_drivers/ar_microscope/build/rebuild.sh <debian_version> <container_name_override> 
# container_name_override is optional and defaults to `arm-latest`

# Constants
ARM_DIR=$HOME/arm
ARM_DRIVERS_DIR=$ARM_DIR/arm_drivers
ARM_MODELS_DIR=$ARM_DIR/arm_models

DEB_VERSION=$1
DEB_PACKAGE_NAME=ar-microscope_"$DEB_VERSION"_amd64.deb
echo $DEB_PACKAGE_NAME
# Container to which the ARM code is copied to and built.
CONTAINER=${2-'arm-latest'}

# Check if the DEB_VERSION argument is empty. debian_version must be provided
if [ -z "$DEB_VERSION" ]; then
  echo "Please specify the version you are building. You can get the version from the control file."
  exit 1
fi

echo "Copying ARM Models into ARM Drivers"
cp -r $ARM_MODELS_DIR/arm_models $ARM_DRIVERS_DIR/ar_microscope/deb_package/usr/local/share/

echo "Copying ARM source code to $CONTAINER"
docker start "$CONTAINER"
docker container cp "$ARM_DRIVERS_DIR"/ar_microscope "$CONTAINER":/home/"${USER}"
# Remove any previously built ARM debians.
docker exec -i "$CONTAINER" bash -c "rm /home/${USER}/ar_microscope/ar-microscope_*.deb"

echo "Starting build in $CONTAINER"
docker exec -i "$CONTAINER" bash -c "cd /home/${USER}/ar_microscope && build/build.sh"

echo "Copying build artifacts to $ARM_DIR"
docker container cp "$CONTAINER":/home/"${USER}"/ar_microscope/$DEB_PACKAGE_NAME "$ARM_DIR"/$DEB_PACKAGE_NAME


echo "Rebuild complete."
