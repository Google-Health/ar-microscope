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
#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_H_

#include <functional>

#include "opencv2/core.hpp"
#include "image_processor/debayer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace image_captor {

class ImageCaptor {
 public:
  virtual ~ImageCaptor();

  virtual tensorflow::Status Initialize() { return tensorflow::Status(); }

  virtual tensorflow::Status Finalize() { return tensorflow::Status(); }

  // Captures image from the device, and copy to output. It allocates
  // memory for cv::Mat if necessary. If is_rgb is true, the output image
  // is in RGB format (with each color 8-bit), otherwise BGR.
  // Args:
  //   is_rgb: Boolean to indicate whether the output format if RGB or BGR.
  //   output: Output image.
  //   on_image_captured: Callback called when the image is captured.
  virtual tensorflow::Status GetImage(
      bool is_rgb, cv::Mat* output,
      std::function<void()> on_image_captured = [] {});

  // Returns height and width of the sensor, which is equal to the dimension
  // of the Bayer pattern image.
  virtual int GetSensorHeight() = 0;
  virtual int GetSensorWidth() = 0;
  // Returns height and width of the RGB image. By default, it's the half
  // size of the sensor because we assume HalfDebayer.
  virtual int GetImageHeight() { return GetSensorHeight() / 2; }
  virtual int GetImageWidth() { return GetSensorWidth() / 2; }

  virtual bool SupportsAutoExposure() { return false; }

  // Returns the current exposure time used. Returns -1 if functionality not
  // supported.
  virtual int GetExposureTimeInMicroseconds() { return -1; }

  // Sets exposure time used by the camera. Note that this will turn off
  // auto-exposure.
  virtual tensorflow::Status SetExposureTime(int microseconds) {
    return tensorflow::Status();
  }

  // Sets the target brightness that a camera's auto-exposure uses to adjust the
  // exposure of capturing an image. The target brightness is a percentage in
  // [0, 100].
  virtual tensorflow::Status SetAutoExposureBrightness(int target_brightness) {
    return tensorflow::errors::Unimplemented("Auto exposure not available.");
  }

 protected:
  virtual tensorflow::Status CaptureImage(uint8_t** image) = 0;

  virtual tensorflow::Status ReleaseImage() { return tensorflow::Status(); }

  // Returns bytes per single Bayer pattern pixel.
  virtual int GetBytesPerPixel() = 0;

  // Pixel type for Bayer image.
  virtual int GetOpenCvPixelType();

  virtual tensorflow::Status SetRgbGains(double red_gain, double green_gain,
                                         double blue_gain) {
    return tensorflow::Status();
  }

  image_processor::Debayer debayer_;
};

}  // namespace image_captor

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_H_
