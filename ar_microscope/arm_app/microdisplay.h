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

#ifndef AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_
#define AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_

#include <QWidget>

#include "arm_app/heatmap_view.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap.pb.h"

namespace arm_app {

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

#endif  // AR_MICROSCOPE_ARM_APP_MICRODISPLAY_H_
