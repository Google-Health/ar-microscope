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
# Image processing for ARM.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:internal"],
)

licenses(["notice"])

cc_library(
    name = "debayer",
    srcs = ["debayer.cc"],
    hdrs = ["debayer.h"],
    deps = [
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "debayer_test",
    srcs = ["debayer_test.cc"],
    deps = [
        ":debayer",
        "@googletest//:gtest_main",
        "@opencv//:opencv",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "image_utils",
    srcs = ["image_utils.cc"],
    hdrs = ["image_utils.h"],
    deps = [
        "@opencv//:opencv",
    ],
)

cc_library(
    name = "inferer",
    srcs = ["inferer.cc"],
    hdrs = ["inferer.h"],
    deps = [
        "@opencv//:opencv",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tensorflow_inferer",
    srcs = ["tensorflow_inferer.cc"],
    hdrs = ["tensorflow_inferer.h"],
    copts = ["-DGOOGLE_CUDA=1"],
    deps = [
        ":inferer",
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//arm_app:arm_config",
        "//microdisplay_server:heatmap_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/compiler/tf2tensorrt:trt_engine_op_op_lib",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/compiler/tf2tensorrt:trt_op_kernels",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
