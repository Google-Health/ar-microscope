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
#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_DEBAYER_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_DEBAYER_H_

#include <memory>

#include "opencv2/core.hpp"
#include "tensorflow/core/lib/core/status.h"

namespace image_processor {

// Class to perform Debayer in multi-thread.
class Debayer {
 public:
  Debayer();
  virtual ~Debayer() {}

  tensorflow::Status HalfDebayer(const cv::Mat& input, cv::Mat* output) {
    return HalfDebayer(input, false, output);
  }

  // Debayer the image in multiple threads. Since 2x2 debayer pattern is
  // put into single output pixel, the output has half width and half hight
  // of the input image.
  //
  // Args:
  //   input: Input image.
  //   is_rgb: Output is in RGB if true, otherwise output is in BGR.
  //   output: Output image.
  tensorflow::Status HalfDebayer(const cv::Mat& input, bool is_rgb,
                                 cv::Mat* output);

  // Sets RGB gains of each color. In some devices, such as Jenoptik,
  // white balance adjustment is not applied to raw Bayer image, therefore
  // we must apply when we Debayer.
  void SetRgbGains(double red, double green, double blue);

 private:
  template <typename T>
  tensorflow::Status HalfDebayerInternal(const cv::Mat& input, bool is_rgb,
                                         cv::Mat* output);

  // Debayer the partial image. Offset and height is specified in
  // output image dimension.
  // typename T: Type of pixel value.
  template <typename T>
  void PartialHalfDebayer(const cv::Mat& input, int offset, int height,
                          bool is_rgb, cv::Mat* output);

  template <typename T>
  inline uint8_t GetPixelValue(T bayer_value, double gain);

  int num_debayer_threads_;

  // RGB gains. Multipliers for each color channel.
  bool has_gain_ = false;
  double red_gain_ = 1.0;
  double green_gain_ = 1.0;
  double blue_gain_ = 1.0;
};

}  // namespace image_processor

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_DEBAYER_H_
