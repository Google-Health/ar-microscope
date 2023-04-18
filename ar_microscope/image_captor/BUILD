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
# Image capture modules for Augmented Reality Microscope.

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:internal"],
)

licenses(["notice"])

cc_library(
    name = "image_captor",
    srcs = ["image_captor.cc"],
    hdrs = ["image_captor.h"],
    deps = [
        "@opencv//:opencv",
        "//image_processor:debayer",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "jenoptik_captor",
    srcs = ["jenoptik_captor.cc"],
    hdrs = ["jenoptik_captor.h"],
    deps = [
        ":image_captor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
        "//:jenoptik",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

config_setting(
    name = "jenoptik",
    values = {"define": "camera=jenoptik"},
)

camera_deps = select({
    ":jenoptik": [":jenoptik_captor"],
    "//conditions:default": [],
})

camera_cops = select({
    ":jenoptik": ["-DCAMERA_JENOPTIK"],
    "//conditions:default": [],
})

cc_library(
    name = "image_captor_factory",
    srcs = ["image_captor_factory.cc"],
    hdrs = ["image_captor_factory.h"],
    copts = camera_cops,
    deps = [
        ":image_captor",
        "@com_google_absl//absl/flags:flag",
    ] + camera_deps,
)
