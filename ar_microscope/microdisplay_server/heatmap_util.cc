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
#include "microdisplay_server/heatmap_util.h"

#include <algorithm>

#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "arm_app/arm_config.h"
#include "image_processor/inferer.h"

ABSL_FLAG(int, relative_threshold, 96,
          "If heatmap value changes more than this threshold, it's considered "
          "as changed.");

namespace microdisplay_server {

void HeatmapUtil::UpdateConfigForModel(
    image_processor::ModelType model_type,
    image_processor::ObjectiveLensPower objective) {
  const auto& model_config =
      arm_app::GetArmConfig().GetModelConfig(model_type, objective);

  config_.positive_threshold = model_config.positive_threshold();
  config_.transformation_scaling = model_config.transformation_scaling();
  config_.blur_size = model_config.blur_size();
  config_.use_morph_open = model_config.use_morph_open();
  config_.morph_size = model_config.morph_size();
}

void HeatmapUtil::CreateHeatmapContourInternal(
    const uint8_t* heatmap, int width, int height, int target_width,
    int target_height, std::vector<std::vector<cv::Point>>* contours,
    std::vector<bool>* is_inner) {
  MaybePrepareMask(width, height);
  int heatmap_size = width * height;

  contours->clear();
  is_inner->clear();
  std::vector<cv::Vec4i> hierarchy;

  if (heatmap_image_.empty() || heatmap_image_.size() != heatmap_size) {
    heatmap_image_.clear();
    heatmap_image_.resize(heatmap_size, 0);
  }

  int relative_threshold = absl::GetFlag(FLAGS_relative_threshold);

  for (int i = 0; i < heatmap_size; i++) {
    int diff =
        static_cast<int>(static_cast<uint8_t>(heatmap[i])) - heatmap_image_[i];
    if (std::abs(diff) >= relative_threshold) {
      // Use the new value of the pixel only when the new value is
      // different enough.
      heatmap_image_[i] = static_cast<uint8_t>(heatmap[i]);
    }
  }

  if (config_.blur_size > 0) {
    CreateSmoothedHeatmapContour(target_width, target_height, contours,
                                 &hierarchy);
  } else {
    CreateStraightHeatmapContour(target_width, target_height, contours,
                                 &hierarchy);
  }

  std::for_each(hierarchy.begin(), hierarchy.end(),
                [is_inner](const cv::Vec4i& vec) {
                  // hierarchy[i][3] indicates the parent polygon index if
                  // any, or negative value if the polygon is at root level.
                  is_inner->push_back(vec[3] >= 0);
                });

  CHECK(contours->size() == is_inner->size());
}

void HeatmapUtil::MaybePrepareMask(int width, int height) {
  if (!mask_.empty() && width == heatmap_width_ && height == heatmap_height_) {
    // If the mask is already prepared, and its demension is the same
    // as the last run, don't need to re-create.
    return;
  }

  mask_.reserve(width * height);
  mask_.resize(0);
  const double radius = std::max(width, height) / 2.0;
  const double radius_square = radius * radius;
  const double center_x = width / 2.0 - 0.5;
  const double center_y = height / 2.0 - 0.5;

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      mask_.push_back(
          DistanceSquare(x, y, center_x, center_y) > radius_square ? 0 : 0xff);
    }
  }
  CHECK(mask_.size() == width * height) << "Invalid mask generation";
  heatmap_width_ = width;
  heatmap_height_ = height;
}

void HeatmapUtil::CreateStraightHeatmapContour(
    int target_width, int target_height,
    std::vector<std::vector<cv::Point>>* contours,
    std::vector<cv::Vec4i>* hierarchy) {
  const int scale_factor_x = target_width / heatmap_width_;
  const int scale_factor_y = target_height / heatmap_height_;
  // Set initial value to zero.
  cv::Mat scaled_masked_thresholded_heatmap_image(target_width, target_height,
                                                  CV_8UC1, cv::Scalar(0));

  const uint8_t threshold = static_cast<uint8_t>(config_.positive_threshold);
  for (int i = 0; i < heatmap_image_.size(); i++) {
    const int x = i % heatmap_width_;
    const int y = i / heatmap_width_;
    const uint8_t pixel_value = heatmap_image_[i];
    if (pixel_value >= threshold && mask_[i] != 0) {
      scaled_masked_thresholded_heatmap_image(
          cv::Rect(x * scale_factor_x, y * scale_factor_y, scale_factor_x,
                   scale_factor_y))
          .setTo(cv::Scalar(0xff));
    }
  }

  // Create contour.
  cv::findContours(scaled_masked_thresholded_heatmap_image, *contours,
                   *hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
}

void HeatmapUtil::CreateSmoothedHeatmapContour(
    int target_width, int target_height,
    std::vector<std::vector<cv::Point>>* contours,
    std::vector<cv::Vec4i>* hierarchy) {
  // Scale up the heatmap bitmap.
  const int transformation_scaling = config_.transformation_scaling;
  cv::Mat scaled_heatmap(heatmap_width_ * transformation_scaling,
                         heatmap_height_ * transformation_scaling, CV_8UC1,
                         cv::Scalar(0));
  for (int i = 0; i < heatmap_image_.size(); i++) {
    const int x = i % heatmap_width_;
    const int y = i / heatmap_width_;
    const uint8_t pixel_value = heatmap_image_[i] & mask_[i];
    scaled_heatmap(cv::Rect(x * transformation_scaling,
                            y * transformation_scaling, transformation_scaling,
                            transformation_scaling))
        .setTo(cv::Scalar(pixel_value));
  }

  // Blur the image.
  // Make the size odd number as required by ::cv::GaussianBlur().
  const int blur_size = config_.blur_size | 1;
  cv::GaussianBlur(scaled_heatmap, scaled_heatmap,
                   cv::Size(blur_size, blur_size), 0.0);

  if (config_.use_morph_open) {
    const int morph_size = config_.morph_size | 1;  // Odd number required.
    cv::morphologyEx(scaled_heatmap, scaled_heatmap, cv::MORPH_OPEN,
                     cv::getStructuringElement(
                         cv::MORPH_RECT, cv::Size(morph_size, morph_size)));
  }

  // Apply threshold.
  const uint8_t threshold = static_cast<uint8_t>(config_.positive_threshold);
  scaled_heatmap.forEach<uint8_t>(
      [&threshold](uint8_t& value, const int position[]) {
        if (value < threshold) {
          value = 0;
        }
      });

  // Create contour polygons.
  cv::findContours(scaled_heatmap, *contours, *hierarchy, cv::RETR_CCOMP,
                   cv::CHAIN_APPROX_SIMPLE);
  // Scale up to the final target size.
  const double scale_x =
      static_cast<double>(target_width) / scaled_heatmap.rows;
  const double scale_y =
      static_cast<double>(target_height) / scaled_heatmap.cols;
  std::for_each(contours->begin(), contours->end(),
                [scale_x, scale_y](std::vector<cv::Point>& polygon) {
                  std::for_each(polygon.begin(), polygon.end(),
                                [&scale_x, &scale_y](cv::Point& point) {
                                  point.x *= scale_x;
                                  point.y *= scale_y;
                                });
                });
}

}  // namespace microdisplay_server
