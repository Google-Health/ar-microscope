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
# This assumes that OpenCV is installed on your system.
#
# Please ensure you have OpenCV 3.2 installed.
cc_library(
  name = "opencv",
  linkopts = [
    "-l:libopencv_core.so",
    "-l:libopencv_imgcodecs.so",
    "-l:libopencv_imgproc.so",
  ],
  visibility = ["//visibility:public"],
)
