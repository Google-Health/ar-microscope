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
"""Bazel build definition for Qt5."""

def qt5_library(name, srcs = [], hdrs = [], deps = [], moc_hdrs = []):
    """Bazel build rule to build Qt source code.

    Args:
      name: Name of the rule.
      srcs: Source file which is not processed by MOC.
      hdrs: Header file which is not processed by MOC.
      deps: Dependencies.
      moc_hdrs: Header files to be processed by MOC.
    """
    new_srcs = [] + srcs
    for moc_hdr in moc_hdrs:
        moc_out = moc_hdr + ".moc.cc"
        moc_rule_name = "moc_" + moc_hdr
        native.genrule(
            name = moc_rule_name,
            srcs = [moc_hdr],
            outs = [moc_out],
            cmd = "moc $(location %s) -o $@" % moc_hdr,
        )
        new_srcs += [moc_out]

    native.cc_library(
        name = name,
        srcs = new_srcs,
        hdrs = hdrs + moc_hdrs,
        copts = [
            "-fPIC",
            "-I/usr/include/x86_64-linux-gnu/qt5",
            "-I/usr/include/x86_64-linux-gnu/qt5/QtCore",
            "-I/usr/include/x86_64-linux-gnu/qt5/QtGui",
            "-I/usr/include/x86_64-linux-gnu/qt5/QtWidgets",
        ],
        linkopts = [
            "-lQt5Core",
            "-lQt5Gui",
            "-lQt5Widgets",
        ],
        deps = deps,
    )
