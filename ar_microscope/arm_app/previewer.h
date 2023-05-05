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
#ifndef AR_MICROSCOPE_MAIN_LOOPER_PREVIEWER_H_
#define AR_MICROSCOPE_MAIN_LOOPER_PREVIEWER_H_

#include <QGridLayout>
#include <QImage>
#include <QLabel>
#include <QWidget>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>  // NOLINT

#include "opencv2/core.hpp"
#include "absl/synchronization/mutex.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace arm_app {

// Function to provide the preview image in RGB format and the heatmap and
// output tensor.
using PreviewProvider =
    std::function<tensorflow::Status(cv::Mat*, cv::Mat*, cv::Mat*)>;

class Previewer : public QWidget {
 public:
  Previewer();
  void Start();
  void Stop();

  // Takes a snapshot of the input image, heatmap, and preview image with the
  // heatmap contours. A json output tensor, metadata, and a comment are also
  // stored. A custom prefix can be provided for the file names to be used in
  // addition to the normal naming schema.
  void TakeSnapshot(image_processor::ObjectiveLensPower objective,
                    image_processor::ModelType model_type,
                    int target_brightness, const std::string& comment,
                    const std::string& custom_prefix = "");

  void SetProvider(PreviewProvider provider) { provider_ = provider; }
  void SetDisplayHeatmap(bool should_display);
  void SetDisplayInference(bool should_display);
  void SetDisplayCalibrationTarget(bool should_display);
  void UpdateHeatmapConfigForModel(
      image_processor::ModelType model_type,
      image_processor::ObjectiveLensPower objective);

  int GetCounter() const { return counter_; }

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  // Renders the preview image and heatmap contour (if display_inference) to
  // the preview window. A snapshot is taken of the various images if the
  // previewer is in snapshot mode.
  void RenderPreview(const cv::Mat& heatmap, const cv::Mat& output_tensor,
                     bool display_inference, cv::Mat* preview_image);

  // Renders the the heatmap contour on the preview image.
  void RenderPreviewHeatmapContour(const cv::Mat& heatmap,
                                   const cv::Mat& output_tensor,
                                   cv::Mat* preview_image);

  // Custom RGB heatmap creation Gleason Patterns.
  std::unique_ptr<QImage> CreateGleasonHeatmap(const cv::Mat& heatmap,
                                               const cv::Mat& output_tensor,
                                               int positive_threshold);

  // Conditionally set column stretch so that the preview display takes up all
  // the space when there is no heatmap display and only half the space when
  // there is.
  void SetLayoutColumnStretch();

  std::unique_ptr<QGridLayout> layout_;
  std::unique_ptr<QLabel> preview_display_;
  std::unique_ptr<QLabel> heatmap_display_;

  PreviewProvider provider_;
  std::unique_ptr<QImage> preview_image_;
  std::unique_ptr<QImage> heatmap_image_;
  absl::Mutex preview_image_mutex_;
  absl::Mutex heatmap_image_mutex_;
  std::atomic_bool display_heatmap_ = {false};
  std::atomic_bool display_inference_ = {true};
  // Note that display calibration overrides display inference so that only the
  // calibration target is displayed.
  std::atomic_bool display_calibration_target_ = {false};

  microdisplay_server::HeatmapUtil heatmap_util_;
  int heatmap_line_width_;
  std::unique_ptr<std::thread> thread_;
  std::atomic_bool to_exit_ = {false};

  // `TakeSnaphot` puts the previewer in snapshot mode where the next update
  // takes the snapshot images and removes the previewer from snapshot mode.
  absl::Mutex snapshot_mutex_;
  bool take_snapshot_ = false;
  std::string snapshot_file_prefix_ = "";

  image_processor::ModelType model_type_ =
      image_processor::ModelType::UNSPECIFIED_MODEL_TYPE;

  // Counter that increments each time a new preview is rendered. This is for
  // clients to check that a new preview was rendered after some new setting was
  // enabled.
  int counter_ = 0;
};

}  // namespace arm_app

#endif  // AR_MICROSCOPE_MAIN_LOOPER_PREVIEWER_H_
