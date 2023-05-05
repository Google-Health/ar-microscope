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
# Main looper of ARM inferrence.

load("//:qt_build.bzl", "qt5_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:internal"],
)

licenses(["notice"])

proto_library(
    name = "arm_event_proto",
    srcs = ["arm_event.proto"],
    deps = [],
)

cc_proto_library(
    name = "arm_event_cc_proto",
    deps = [":arm_event_proto"],
)

qt5_library(
    name = "looper",
    srcs = ["looper.cc"],
    hdrs = ["looper.h"],
    deps = [
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//arm_app:microdisplay",
        "//arm_app:previewer",
        "//image_captor",
        "//image_captor:image_captor_factory",
        "//image_processor:inferer",
        "//image_processor:tensorflow_inferer",
        "//microdisplay_server:heatmap_cc_proto",
        "//microdisplay_server:heatmap_util",
        "//microdisplay_server:inference_timings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

qt5_library(
    name = "capture_image_lib",
    srcs = [
        "capture_image.cc",
    ],
    moc_hdrs = [
        "capture_image.h",
    ],
    deps = [
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "//image_captor:image_captor_factory",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tf_cc_binary(
    name = "capture_image",
    deps = [
        ":capture_image_lib",
    ],
)

qt5_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":arm_event_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
