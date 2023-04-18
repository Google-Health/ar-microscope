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
# BUILD when using the github serial repo (during Docker builds).

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "serial",
    srcs = [
        "src/impl/list_ports/list_ports_linux.cc",
        "src/impl/unix.cc",
        "src/serial.cc",
    ],
    copts = [
        "-fexceptions",
        "-Wno-mismatched-new-delete",
    ],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    includes = ["include"],
    textual_hdrs = [
        "include/serial/impl/unix.h",
        "include/serial/serial.h",
        "include/serial/v8stdint.h",
    ],
)
