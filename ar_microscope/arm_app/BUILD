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
# ARM main application and views.

load("//:qt_build.bzl", "qt5_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:internal"],
)

licenses(["notice"])

proto_library(
    name = "arm_config_proto",
    srcs = ["arm_config.proto"],
    deps = [],
)

cc_proto_library(
    name = "arm_config_cc_proto",
    deps = [":arm_config_proto"],
)

qt5_library(
    name = "main_window",
    srcs = ["main_window.cc"],
    moc_hdrs = ["main_window.h"],
    deps = [
        ":microdisplay",
        ":previewer",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "//image_processor:inferer",
        "//image_processor:tensorflow_inferer",
        "//main_looper:arm_event_cc_proto",
        "//main_looper:logger",
        "//main_looper:looper",
        "//serial_interface:objective_serial",
    ],
)

qt5_library(
    name = "microdisplay",
    srcs = ["microdisplay.cc"],
    hdrs = ["microdisplay.h"],
    deps = [
        ":arm_config",
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "//image_processor:image_utils",
        "//image_processor:inferer",
        "//microdisplay_server:heatmap_cc_proto",
        "//microdisplay_server:heatmap_util",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

qt5_library(
    name = "previewer",
    srcs = ["previewer.cc"],
    hdrs = ["previewer.h"],
    deps = [
        ":arm_config",
        "@opencv//:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//image_processor:image_utils",
        "//image_processor:inferer",
        "//microdisplay_server:heatmap_util",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

qt5_library(
    name = "main_lib",
    srcs = ["main.cc"],
    deps = [
        ":arm_config",
        ":main_window",
        ":microdisplay",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "//main_looper:arm_event_cc_proto",
        "//main_looper:logger",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

qt5_library(
    name = "arm_config",
    srcs = ["arm_config.cc"],
    hdrs = ["arm_config.h"],
    deps = [
        ":arm_config_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//image_processor",
        "//image_processor:inferer","@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tf_cc_binary(
    name = "arm_inferer",
    deps = [
        ":main_lib",
    ],
)
