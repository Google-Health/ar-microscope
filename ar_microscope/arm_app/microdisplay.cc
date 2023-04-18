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
#include "arm_app/microdisplay.h"

#include <math.h>

#include <QGuiApplication>
#include <QPainter>
#include <QPalette>
#include <QScreen>
#include <algorithm>
#include <memory>

#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "arm_app/arm_config.h"
#include "image_processor/image_utils.h"
#include "tensorflow/core/platform/logging.h"

ABSL_FLAG(int, display_index, -1,
          "The display index to show heatmap on. If -1, show the heatmap "
          "on the last display.");

ABSL_FLAG(int, image_width, 1050, "Width of the microdisplay");
ABSL_FLAG(int, image_height, 1050, "Height of the microdisplay");

ABSL_FLAG(bool, contour, true, "Show contour of the heatmap");

extern absl::Flag<bool> FLAGS_calibration_mode;

namespace arm_app {
namespace {

using microdisplay_server::Heatmap;

QImage* MakeCalibrationImage(int image_size) {
  cv::Mat target_buffer = cv::Mat::zeros(image_size, image_size, CV_8UC3);
  image_processor::RenderCalibrationTarget(&target_buffer);
  auto target_image = new QImage(target_buffer.data, image_size, image_size,
                                 target_buffer.step, QImage::Format_RGB888);
  return target_image;
}

}  // namespace

HeatmapView::HeatmapView(QWidget* parent)
    : QWidget(parent),
      display_calibration_target_(absl::GetFlag(FLAGS_calibration_mode)) {
  LOG(INFO) << "Window size (" << parent->width() << ", " << parent->height()
            << ")";
  const auto& microdisplay_config =
      arm_app::GetArmConfig().GetMicrodisplayConfig();
  display_margin_left_.store(microdisplay_config.image_margin_left());
  display_margin_top_.store(microdisplay_config.image_margin_top());
  setGeometry(display_margin_left_, display_margin_top_,
              absl::GetFlag(FLAGS_image_width),
              absl::GetFlag(FLAGS_image_height));

  QPalette widget_palette(palette());
  widget_palette.setColor(QPalette::Background, Qt::black);
  setAutoFillBackground(true);
  setPalette(widget_palette);
}

void HeatmapView::LoadHeatmap(Heatmap* heatmap) {
  if (display_calibration_target_ || !display_inference_) {
    // Remove all inference artifacts.
    absl::MutexLock unused_image_lock(&image_mutex_);
    absl::MutexLock unused_polygons_lock(&polygons_mutex_);
    polygons_.clear();
    image_ = nullptr;
  } else {  // display inference
    if (absl::GetFlag(FLAGS_contour)) {
      CreateContourPolygons(*heatmap);
    } else {
      RenderHeatmapImage(*heatmap);
    }
  }

  // Notify Qt to redraw this widget.
  update();
}

void HeatmapView::UpdateHeatmapConfigForModel(
    image_processor::ModelType model_type,
    image_processor::ObjectiveLensPower objective) {
  heatmap_util_.UpdateConfigForModel(model_type, objective);
  heatmap_line_width_ = arm_app::GetArmConfig()
                            .GetModelConfig(model_type, objective)
                            .heatmap_line_width();
}

void HeatmapView::paintEvent(QPaintEvent* event) {
  QPainter painter(this);

  if (display_calibration_target_) {
    if (!calibration_image_) {
      const int image_size = std::max(absl::GetFlag(FLAGS_image_width),
                                      absl::GetFlag(FLAGS_image_height));
      calibration_image_ = absl::WrapUnique(MakeCalibrationImage(image_size));
    }
    painter.drawImage(rect(), *calibration_image_, calibration_image_->rect());
  } else if (display_inference_) {
    if (image_) {
      // Render image.
      painter.drawImage(rect(), *image_, image_->rect());
    }

    // Render polygons.
    QPen contour_pen(Qt::green);
    contour_pen.setWidth(heatmap_line_width_);
    // Thin pen for inner loop.
    QPen contour_thin_pen(contour_pen);
    contour_thin_pen.setWidth(contour_pen.width() / 2);

    {
      absl::MutexLock unused_lock(&polygons_mutex_);
      for (int i = 0; i < polygons_.size(); i++) {
        auto polygon = polygons_[i];
        painter.setPen(is_inner_[i] ? contour_thin_pen : contour_pen);
        painter.drawPolygon(polygon);
      }
    }
  }
}

void HeatmapView::CreateContourPolygons(const Heatmap& heatmap) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<bool> is_inner;

  heatmap_util_.CreateHeatmapContour(heatmap, absl::GetFlag(FLAGS_image_width),
                                     absl::GetFlag(FLAGS_image_height),
                                     &contours, &is_inner);

  {
    absl::MutexLock unused_lock(&polygons_mutex_);
    // Convert OpenCV polygon to QPolygon.
    polygons_.clear();
    for (const auto& cv_polygon : contours) {
      QPolygon polygon;
      for (const cv::Point& cv_point : cv_polygon) {
        polygon << QPoint(cv_point.x, cv_point.y);
      }
      polygons_.push_back(polygon);
    }
    is_inner_ = is_inner;
  }
}

void HeatmapView::RenderHeatmapImage(const Heatmap& heatmap) {
  std::vector<uint8_t> buffer;
  const std::string& image = heatmap.image_binary();
  buffer.resize(image.size() * 3);
  for (int i = 0; i < image.size(); i++) {
    buffer[i * 3] = 0;             // Red
    buffer[i * 3 + 1] = image[i];  // Green
    buffer[i * 3 + 2] = 0;         // Blue
  }
  {
    absl::MutexLock unused_lock(&image_mutex_);
    image_ = std::make_unique<QImage>(buffer.data(), heatmap.width(),
                                      heatmap.height(), QImage::Format_RGB888);
  }
}

void HeatmapView::AdjustMarginLeft(int diff) {
  const auto& microdisplay_config =
      arm_app::GetArmConfig().GetMicrodisplayConfig();
  display_margin_left_.store(microdisplay_config.image_margin_left() + diff);
  setGeometry(display_margin_left_, display_margin_top_,
              absl::GetFlag(FLAGS_image_width),
              absl::GetFlag(FLAGS_image_height));
}

void HeatmapView::AdjustMarginTop(int diff) {
  const auto& microdisplay_config =
      arm_app::GetArmConfig().GetMicrodisplayConfig();
  display_margin_top_.store(microdisplay_config.image_margin_top() + diff);
  setGeometry(display_margin_left_, display_margin_top_,
              absl::GetFlag(FLAGS_image_width),
              absl::GetFlag(FLAGS_image_height));
}

Microdisplay::Microdisplay() {
  SelectDisplay();
  heatmap_view_ = std::make_unique<HeatmapView>(this);
  show();
}

void Microdisplay::SelectDisplay() {
  // Get the list of all screens.
  QList<QScreen*> screens = QGuiApplication::screens();
  LOG(INFO) << "Number of screens: " << screens.size();
  if (screens.empty()) {
    LOG(FATAL) << "No display attached";
  }

  const QScreen* target = nullptr;
  int index = absl::GetFlag(FLAGS_display_index);
  if (index >= 0 && index < screens.size()) {
    // If the display number is specified (and is valid), that's the
    // display we want to show on.
    target = screens[index];
  } else {
    // Otherwise, show the image on the last display.
    target = screens.last();
  }
  setGeometry(target->geometry());
}

}  // namespace arm_app
