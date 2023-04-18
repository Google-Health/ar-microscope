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

function die() {
  echo $@ >&2
  exit 1
}

bazel build -c opt --config=cuda --config=monolithic  \
    --define camera=jenoptik  \
    arm_app:arm_inferer main_looper:capture_image  \
    || die "Failed to build ARM binary"

cp -f ./bazel-bin/arm_app/arm_inferer  \
    ./bazel-bin/main_looper/capture_image  \
    deb_package/usr/local/bin/  \
    || die "Failed to copy binaries to package directory"

chmod -R 0755 deb_package
chmod 0644 deb_package/usr/local/share/arm_models/*/saved_model.pb
chmod 0644 deb_package/usr/local/share/arm_models/*/variables/*

fakeroot dpkg-deb --build deb_package .  \
    || die "Failed to create deb package"
