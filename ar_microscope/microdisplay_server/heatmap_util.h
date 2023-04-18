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
#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_MICRODISPLAY_SERVER_HEATMAP_UTIL_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_MICRODISPLAY_SERVER_HEATMAP_UTIL_H_

#include <vector>

#include "opencv2/core.hpp"
#include "absl/time/time.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap.pb.h"

namespace microdisplay_server {

// Settings for creating the heatmap contours.
struct HeatmapUtilConfig {
  // Threshold for considering a heatmap value to be positive.
  int positive_threshold = 128;
  // Scaling factor before Gaussian blur is applied to heatmap
  int transformation_scaling = 1;
  // Blur size to smooth heatmap contour, in scaled heatmap pixels. If this is
  // not positive, a straight heatmap contour is used.
  int blur_size = 0;
  // Use morphological opening (erosion followed by dilation).
  bool use_morph_open = false;
  // Kernel size for the morphological opening.
  int morph_size = 0;
};

class HeatmapUtil {
 public:
  void UpdateConfigForModel(image_processor::ModelType model_type,
                            image_processor::ObjectiveLensPower objective);

  void CreateHeatmapContour(const Heatmap& heatmap, int target_width,
                            int target_height,
                            std::vector<std::vector<cv::Point>>* contours,
                            std::vector<bool>* is_inner) {
    CreateHeatmapContourInternal(
        reinterpret_cast<const uint8_t*>(heatmap.image_binary().data()),
        heatmap.width(), heatmap.height(), target_width, target_height,
        contours, is_inner);
  }

  void CreateHeatmapContour(const cv::Mat& heatmap, int target_width,
                            int target_height,
                            std::vector<std::vector<cv::Point>>* contours,
                            std::vector<bool>* is_inner) {
    CreateHeatmapContourInternal(heatmap.ptr(), heatmap.cols, heatmap.rows,
                                 target_width, target_height, contours,
                                 is_inner);
  }

  int GetPositiveThreshold() { return config_.positive_threshold; }

 private:
  void CreateHeatmapContourInternal(
      const uint8_t* heatmap, int width, int height, int target_width,
      int target_height, std::vector<std::vector<cv::Point>>* contours,
      std::vector<bool>* is_inner);

  void MaybePrepareMask(int width, int height);

  void CreateStraightHeatmapContour(
      int target_width, int target_height,
      std::vector<std::vector<cv::Point>>* contours,
      std::vector<cv::Vec4i>* hierarchy);

  void CreateSmoothedHeatmapContour(
      int target_width, int target_height,
      std::vector<std::vector<cv::Point>>* contours,
      std::vector<cv::Vec4i>* hierarchy);

  inline static double DistanceSquare(double x1, double y1, double x2,
                                      double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
  }

  // Cache of the mask to avoid repeated creation of heatmap mask. Note
  // heatmap size is constant as long as the model is the same.
  std::vector<uint8_t> mask_;

  // Width and height of the mask.
  int32_t heatmap_width_;
  int32_t heatmap_height_;

  // Settings for heatmap contour drawing.
  HeatmapUtilConfig config_;

  // Heatmap image binary to be rendered.
  std::vector<uint8_t> heatmap_image_;
};

class InferenceTimings {
 public:
  InferenceTimings();

  void AddTiming(const Heatmap& heatmap);

  static void SetTimingCheckpoint(InferenceCheckpoint::Type type,
                                  Heatmap* heatmap);

 private:
  void Clear();
  std::string GetAverageDurationTime(InferenceCheckpoint::Type type);

  static const absl::Time GetCheckpoint(const Heatmap& heatmap,
                                        InferenceCheckpoint::Type type);

  int64_t count_ = 0;
  // Total duration in microseconds.
  absl::Duration total_ = absl::ZeroDuration();

  // Accumulated time for each step. Index number is equal to type number.
  // Therefore, steps_[0] is empty, since there's no timing for
  // UNSPECIFIED_CHECKPOINT.
  std::vector<absl::Duration> steps_;
};

}  // namespace microdisplay_server

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_MICRODISPLAY_SERVER_HEATMAP_UTIL_H_
