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
#include "arm_app/previewer.h"

#include <QGridLayout>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QSizePolicy>
#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <memory>
#include <thread>  // NOLINT
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "arm_app/arm_config.h"
#include "image_processor/image_utils.h"
#include "image_processor/inferer.h"
#include "tensorflow/core/platform/logging.h"

ABSL_FLAG(int, image_size, 1800, "Expected image size for the patch.");

ABSL_FLAG(int32_t, preview_delay, 100,
          "Delay in milliseconds added to the end of preview cycles.");

ABSL_FLAG(bool, use_rgb_gleason_heatmap, true,
          "Whether to use an RGB heatmap for Gleason, where green, yellow, and "
          "red represent Gleason Patterns 3, 4, and 5, respectively.");

extern absl::Flag<std::string> FLAGS_log_directory;
extern absl::Flag<bool> FLAGS_calibration_mode;

namespace arm_app {
namespace {

using ::image_processor::ModelType;
using ::image_processor::ObjectiveLensPower;

constexpr char kSnapshotDir[] = "snapshots";

constexpr char kInputFilename[] = "input.png";
constexpr char kHeatmapFilename[] = "heatmap.png";
constexpr char kPreviewFilename[] = "preview.png";
constexpr char kCommentFilename[] = "comment.txt";
constexpr char kMetadataFilename[] = "metadata.txt";
constexpr char kOutputTensorFilename[] = "output_tensor.json";

void DisplayImage(const QImage& image, QLabel* display) {
  QImage resized_image;
  // Resize the image to fit the display, keeping aspect ratio.
  resized_image = image.scaled(display->size(), Qt::KeepAspectRatio);
  display->setPixmap(QPixmap::fromImage(resized_image));
}

// Converts and RGB image to BGR (required by OpenCV) and stores it.
void StoreRgbImage(const std::string& filename, const cv::Mat& image) {
  cv::Mat image_bgr;
  cv::cvtColor(image, image_bgr, cv::COLOR_RGB2BGR);
  cv::imwrite(filename, image_bgr);
}

void StoreSnapshotTextFile(const std::string& filename,
                           const std::string& text) {
  std::ofstream text_file(filename);
  text_file << text << "\n";
  text_file.close();
}

void StoreTensorAsJson(const std::string& filename, const cv::Mat& tensor) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "output_tensor" << tensor;
  fs.release();
}

}  // namespace

Previewer::Previewer()
    : display_calibration_target_(absl::GetFlag(FLAGS_calibration_mode)) {
  layout_ = std::make_unique<QGridLayout>(this);
  this->setLayout(layout_.get());
  preview_display_ = std::make_unique<QLabel>(this);
  heatmap_display_ = std::make_unique<QLabel>(this);
  preview_display_->setAlignment(Qt::AlignCenter);
  heatmap_display_->setAlignment(Qt::AlignCenter);
  preview_display_->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
  heatmap_display_->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
  preview_display_->setMinimumSize(1, 1);
  heatmap_display_->setMinimumSize(1, 1);
  layout_->addWidget(preview_display_.get(), 0, 0);
  layout_->addWidget(heatmap_display_.get(), 0, 1);
  heatmap_display_->setVisible(display_heatmap_);
  SetLayoutColumnStretch();
}

void Previewer::Start() {
  show();

  thread_ = std::make_unique<std::thread>([this]() {
    cv::Mat preview;
    cv::Mat heatmap;
    cv::Mat output_tensor;
    while (!to_exit_.load()) {
      if (!provider_) {
        LOG(WARNING) << "Preview provider not assigned";
        std::this_thread::sleep_for(std::chrono::seconds(5));
        continue;
      }
      {
        // Protect preview image from updating while it's rendered.
        // Since this preview update procedure runs in a separate thread,
        // another thread may be using the preview image for redraw
        // (e.g. when the windows is resized). Note that `preview_image_` and
        // `preview` share data, so this lock must be held before `preview` is
        // updated by the preview provider.
        absl::MutexLock unused_preview_lock(&preview_image_mutex_);
        absl::MutexLock unused_heatmap_lock(&heatmap_image_mutex_);
        tensorflow::Status result =
            provider_(&preview, &heatmap, &output_tensor);
        if (!result.ok()) {
          LOG(WARNING) << "Error on preview provider: " << result;
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        {
          absl::MutexLock unused_snapshot_lock(&snapshot_mutex_);
          RenderPreview(heatmap, output_tensor, display_inference_, &preview);
          take_snapshot_ = false;
        }
        preview_image_ =
            std::make_unique<QImage>(preview.ptr(), preview.cols, preview.rows,
                                     preview.step, QImage::Format_RGB888);

        if (display_heatmap_) {
          if (model_type_ == ModelType::GLEASON &&
              absl::GetFlag(FLAGS_use_rgb_gleason_heatmap)) {
            heatmap_image_ = CreateGleasonHeatmap(
                heatmap, output_tensor, heatmap_util_.GetPositiveThreshold());
          } else {
            heatmap_image_ = std::make_unique<QImage>(
                heatmap.ptr(), heatmap.cols, heatmap.rows, heatmap.step,
                QImage::Format_Grayscale8);
          }
        }
      }
      counter_++;
      update();

      int delay_milliseconds = absl::GetFlag(FLAGS_preview_delay);
      if (delay_milliseconds > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(delay_milliseconds));
      }
    }
  });
}

void Previewer::Stop() {
  to_exit_.store(true);
  thread_->join();
  thread_.reset();
}

void Previewer::SetLayoutColumnStretch() {
  layout_->setColumnStretch(0, display_heatmap_);
  layout_->setColumnStretch(1, display_heatmap_);
}

void Previewer::paintEvent(QPaintEvent* event) {
  if (preview_image_) {
    absl::MutexLock unused_preview_lock(&preview_image_mutex_);
    DisplayImage(*preview_image_, preview_display_.get());
  }
  if (display_heatmap_ && heatmap_image_) {
    absl::MutexLock unused_heatmap_lock(&heatmap_image_mutex_);
    DisplayImage(*heatmap_image_, heatmap_display_.get());
  }
}

void Previewer::SetDisplayHeatmap(bool should_display) {
  display_heatmap_.store(should_display);
  heatmap_display_->setVisible(display_heatmap_);
  SetLayoutColumnStretch();
}

void Previewer::SetDisplayInference(bool should_display) {
  display_inference_.store(should_display);
}

void Previewer::SetDisplayCalibrationTarget(bool should_display) {
  display_calibration_target_.store(should_display);
}

void Previewer::UpdateHeatmapConfigForModel(
    image_processor::ModelType model_type,
    image_processor::ObjectiveLensPower objective) {
  model_type_ = model_type;
  heatmap_util_.UpdateConfigForModel(model_type, objective);
  heatmap_line_width_ = arm_app::GetArmConfig()
                            .GetModelConfig(model_type, objective)
                            .heatmap_line_width();
}

void Previewer::TakeSnapshot(ObjectiveLensPower objective, ModelType model_type,
                             int target_brightness, const std::string& comment,
                             const std::string& custom_prefix) {
  absl::MutexLock unused_lock(&snapshot_mutex_);
  take_snapshot_ = true;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int epoch_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now).count();
  if (custom_prefix.empty()) {
    snapshot_file_prefix_ = absl::StrFormat(
        "%s/%s/%d_%s_%s", absl::GetFlag(FLAGS_log_directory), kSnapshotDir,
        epoch_seconds, image_processor::ModelTypeToString(model_type),
        image_processor::ObjectiveToString(objective));
  } else {
    snapshot_file_prefix_ = absl::StrFormat(
        "%s/%s/%s_%s_%s", absl::GetFlag(FLAGS_log_directory), kSnapshotDir,
        custom_prefix, image_processor::ModelTypeToString(model_type),
        image_processor::ObjectiveToString(objective));
  }
  StoreSnapshotTextFile(
      absl::StrCat(snapshot_file_prefix_, "_", kCommentFilename), comment);
  auto metadata = absl::StrFormat(
      "{model_version: %s, model_type: %s, objective: %s, target_brightness: "
      "%d, timestamp: %d}",
      arm_app::GetArmConfig()
          .GetModelConfig(model_type, objective)
          .model_version(),
      image_processor::ModelTypeToString(model_type),
      image_processor::ObjectiveToString(objective), target_brightness,
      epoch_seconds);
  StoreSnapshotTextFile(
      absl::StrCat(snapshot_file_prefix_, "_", kMetadataFilename), metadata);
  LOG(INFO) << "Storing a snapshot at " << snapshot_file_prefix_;
}

void Previewer::RenderPreviewHeatmapContour(const cv::Mat& heatmap,
                                            const cv::Mat& output_tensor,
                                            cv::Mat* preview_image) {
  std::vector<std::vector<cv::Point>> polygons;
  std::vector<bool> is_inner;
  const int image_size = absl::GetFlag(FLAGS_image_size);
  heatmap_util_.CreateHeatmapContour(heatmap, image_size, image_size, &polygons,
                                     &is_inner);
  int left_padding = (image_size - preview_image->cols) / 2;
  int top_padding = (image_size - preview_image->rows) / 2;
  for (int i = 0; i < polygons.size(); i++) {
    int line_width = heatmap_line_width_;
    if (is_inner[i]) {
      line_width /= 2;
    }
    auto& polygon = polygons[i];
    std::for_each(polygon.begin(), polygon.end(),
                  [top_padding, left_padding](cv::Point& point) {
                    point.x -= left_padding;
                    point.y -= top_padding;
                  });
    int num_vertices[1];
    num_vertices[0] = polygon.size();
    const cv::Point* vertices[1];
    vertices[0] = polygon.data();
    cv::polylines(*preview_image, vertices, num_vertices, 1, true,
                  cv::Scalar(0, 255, 0), line_width);
  }
}

void Previewer::RenderPreview(const cv::Mat& heatmap,
                              const cv::Mat& output_tensor,
                              bool display_inference, cv::Mat* preview_image) {
  if (take_snapshot_) {
    std::string input_filename =
        absl::StrCat(snapshot_file_prefix_, "_", kInputFilename);
    std::string heatmap_filename =
        absl::StrCat(snapshot_file_prefix_, "_", kHeatmapFilename);
    std::string output_tensor_filename =
        absl::StrCat(snapshot_file_prefix_, "_", kOutputTensorFilename);

    StoreRgbImage(input_filename, *preview_image);
    cv::imwrite(heatmap_filename, heatmap);
    StoreTensorAsJson(output_tensor_filename, output_tensor);
  }
  if (display_calibration_target_) {
    image_processor::RenderCalibrationTarget(preview_image);
  } else if (display_inference) {
    RenderPreviewHeatmapContour(heatmap, output_tensor, preview_image);
  }
  if (take_snapshot_) {
    std::string preview_filename =
        absl::StrCat(snapshot_file_prefix_, "_", kPreviewFilename);
    StoreRgbImage(preview_filename, *preview_image);
  }
}

std::unique_ptr<QImage> Previewer::CreateGleasonHeatmap(
    const cv::Mat& heatmap, const cv::Mat& output_tensor,
    int positive_threshold) {
  const int height = output_tensor.size[0];
  const int width = output_tensor.size[1];
  cv::Mat rgb_heatmap_tensor = cv::Mat::zeros(height, width, CV_8UC3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (heatmap.at<uint8_t>(y, x) < positive_threshold) {
        // Skip pixels that aren't positive.
        continue;
      }
      int max_class_idx = 1;
      int max_pixel_value = output_tensor.at<uint8_t>(y, x, max_class_idx);
      for (int class_idx = 2; class_idx < 4; class_idx++) {
        uint8_t new_pixel_value = output_tensor.at<uint8_t>(y, x, class_idx);
        if (new_pixel_value > max_pixel_value) {
          max_class_idx = class_idx;
          max_pixel_value = new_pixel_value;
        }
      }
      if (max_class_idx == 1) {  // Gleason Pattern 3 -> green
        rgb_heatmap_tensor.ptr(y, x)[1] = max_pixel_value;
      } else if (max_class_idx == 2) {  // Gleason Pattern 4 -> yellow
        rgb_heatmap_tensor.ptr(y, x)[0] = max_pixel_value;
        rgb_heatmap_tensor.ptr(y, x)[1] = max_pixel_value;
      } else if (max_class_idx == 3) {  // Gleason Pattern 5 -> red
        rgb_heatmap_tensor.ptr(y, x)[0] = max_pixel_value;
      }
    }
  }
  return std::make_unique<QImage>(
      rgb_heatmap_tensor.ptr(), rgb_heatmap_tensor.cols,
      rgb_heatmap_tensor.rows, rgb_heatmap_tensor.step, QImage::Format_RGB888);
}

}  // namespace arm_app
