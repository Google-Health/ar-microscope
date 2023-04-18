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
#include "image_captor/image_captor.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace image_captor {

ImageCaptor::~ImageCaptor() {
  tensorflow::Status status = Finalize();
  if (!status.ok()) {
    LOG(ERROR) << "Image captor finalize error: " << status;
  }
}

tensorflow::Status ImageCaptor::GetImage(
    bool is_rgb, cv::Mat* output, std::function<void()> on_image_captured) {
  uint8_t* raw_image;
  TF_RETURN_IF_ERROR(CaptureImage(&raw_image));

  if (on_image_captured) {
    on_image_captured();
  }

  cv::Mat bayer_image(GetSensorHeight(), GetSensorWidth(), GetOpenCvPixelType(),
                      raw_image);
  tensorflow::Status status = debayer_.HalfDebayer(bayer_image, is_rgb, output);

  tensorflow::Status release_result = ReleaseImage();
  if (!release_result.ok()) {
    return release_result;
  }

  return status;
}

int ImageCaptor::GetOpenCvPixelType() {
  switch (GetBytesPerPixel()) {
    case 1:
      return CV_8UC1;
    case 2:
      return CV_16UC1;
    default:
      return 0;
  }
}

}  // namespace image_captor
