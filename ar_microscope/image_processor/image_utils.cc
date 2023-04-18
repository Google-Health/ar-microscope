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
#include "image_processor/image_utils.h"

#include <math.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace {

const cv::Scalar kContourColor(0, 255, 0);  // Green

// Number of target circles to draw on the image.
constexpr int kNumTargetCircles = 3;

constexpr int kCalibrationLineWidth = 3;
}

namespace image_processor {

void RenderCalibrationTarget(cv::Mat* preview_image) {
  const int image_size = std::max(preview_image->rows, preview_image->cols);
  const int circle_center_int = (image_size - 1) / 2;
  const auto circle_center = cv::Point(circle_center_int, circle_center_int);
  const int radius = (image_size / 2) / kNumTargetCircles;
  for (int i = 0; i < kNumTargetCircles; ++i) {
    cv::circle(*preview_image, circle_center, radius * (i + 1), kContourColor,
               kCalibrationLineWidth, cv::LineTypes::LINE_AA);
  }
}

}  // namespace image_processor
