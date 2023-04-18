// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "image_captor/image_captor_factory.h"
#include "image_captor/jenoptik_captor.h"
#include "absl/flags/flag.h"

ABSL_FLAG(std::string, capture_device, "jenoptik",
          "Image capture device type. ");

namespace image_captor {


ImageCaptor* ImageCaptorFactory::Create() {
  std::string device = absl::GetFlag(FLAGS_capture_device);
  LOG(INFO) << "Device type: " << device;
  constexpr char kDeviceJenoptik[] = "jenoptik";
  if (device == kDeviceJenoptik) {
    return new JenoptikCaptor();
  }

  LOG(FATAL) << "Unsupported device type: " << device;
  return nullptr;
}

}  // namespace image_captor
