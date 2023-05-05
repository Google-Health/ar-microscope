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

#include <QGuiApplication>
#include <QPainter>
#include <QPalette>
#include <QScreen>
#include <memory>

#include "absl/flags/flag.h"
#include "arm_app/heatmap_view.h"
#include "image_processor/image_utils.h"

ABSL_FLAG(int, display_index, -1,
          "The display index to show heatmap on. If -1, show the heatmap "
          "on the last display.");

namespace arm_app {

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
