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
#include "image_processor/binarizer.h"

#include "opencv2/imgproc.hpp"

namespace image_processor {

tensorflow::Status Binarizer::ProcessImage(const cv::Mat& input,
                                           cv::Mat* output) {
  cv::Mat resized(output->rows, output->cols, CV_8UC3);
  cv::resize(input, resized, cv::Size(output->rows, output->cols));
  cv::Mat gray_scale(output->rows, output->cols, CV_8UC1);
  cv::cvtColor(resized, gray_scale, cv::COLOR_BGR2GRAY);
  cv::threshold(gray_scale, *output, 128, 255, cv::THRESH_BINARY_INV);
  return tensorflow::Status();
}

}  // namespace image_processor
