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
// Qt widgets to show inference result heatmap on microdisplay embedded to
// microscope.

#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_

#include <QWidget>
#include <memory>
#include <vector>

#include "opencv2/core.hpp"
#include "absl/synchronization/mutex.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap.pb.h"
#include "microdisplay_server/heatmap_util.h"

namespace arm_app {

// Widget to render heatmap contour or heatmap bitmap.
class HeatmapView : public QWidget {
 public:
  explicit HeatmapView(QWidget* parent);
  void LoadHeatmap(microdisplay_server::Heatmap* heatmap);

  void UpdateHeatmapConfigForModel(
      image_processor::ModelType model_type,
      image_processor::ObjectiveLensPower objective);

  void SetDisplayInference(bool should_display) {
    display_inference_ = should_display;
  }

  void SetDisplayCalibrationTarget(bool should_display) {
    display_calibration_target_ = should_display;
  }

  void AdjustMarginLeft(int diff);
  void AdjustMarginTop(int diff);

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  void CreateContourPolygons(const microdisplay_server::Heatmap& heatmap);
  void RenderHeatmapImage(const microdisplay_server::Heatmap& heatmap);

  absl::Mutex polygons_mutex_;
  std::vector<QPolygon> polygons_;
  std::vector<bool> is_inner_;

  absl::Mutex image_mutex_;
  std::unique_ptr<QImage> image_;
  std::unique_ptr<QImage> calibration_image_;
  std::atomic_bool display_inference_{true};
  // Note that display calibration overrides display inference so that only the
  // calibration target is displayed.
  std::atomic_bool display_calibration_target_;
  std::atomic_int display_margin_left_;
  std::atomic_int display_margin_top_;

  int heatmap_line_width_;

  microdisplay_server::HeatmapUtil heatmap_util_;
};

// Root window to show heatmap result. It expands to the last screen.
class Microdisplay : public QWidget {
 public:
  Microdisplay();

  void SetDisplayInference(bool should_display) {
    heatmap_view_->SetDisplayInference(should_display);
  }

  void SetDisplayCalibrationTarget(bool should_display) {
    heatmap_view_->SetDisplayCalibrationTarget(should_display);
  }

  void AdjustMarginLeft(int diff) { heatmap_view_->AdjustMarginLeft(diff); }
  void AdjustMarginTop(int diff) { heatmap_view_->AdjustMarginTop(diff); }

  void ShowHeatmap(microdisplay_server::Heatmap* heatmap) {
    heatmap_view_->LoadHeatmap(heatmap);
  }

  void UpdateHeatmapConfigForModel(
      image_processor::ModelType model_type,
      image_processor::ObjectiveLensPower objective) {
    heatmap_view_->UpdateHeatmapConfigForModel(model_type, objective);
  }

 private:
  void SelectDisplay();

  std::unique_ptr<HeatmapView> heatmap_view_;
};

}  // namespace arm_app

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_
