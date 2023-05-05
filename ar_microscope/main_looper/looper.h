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
// Class to run the main loop of capturing image, performing inference
// and showing heatmap result on microdisplay. It also manages Qt
// application and windows.

#ifndef AR_MICROSCOPE_MAIN_LOOPER_LOOPER_H_
#define AR_MICROSCOPE_MAIN_LOOPER_LOOPER_H_

#include <atomic>
#include <memory>
#include <thread>  // NOLINT

#include "absl/synchronization/mutex.h"
#include "arm_app/microdisplay.h"
#include "arm_app/previewer.h"
#include "image_captor/image_captor.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap.pb.h"
#include "microdisplay_server/inference_timings.h"
#include "tensorflow/core/lib/core/status.h"

namespace main_looper {

// Callback provided by `MainWindow` for emitting a signal to display warnings.
using DisplayWarningCallback = std::function<void(const std::string)>;

class Looper {
 public:
  Looper(image_processor::ObjectiveLensPower objective,
         image_processor::ModelType model_type, arm_app::Previewer* previewer,
         arm_app::Microdisplay* microdisplay,
         DisplayWarningCallback display_warning_callback);

  // Runs a thread for image capturing, inference, and displaying.
  void Run();
  void Stop();

  void SetObjectiveAndModelType(image_processor::ObjectiveLensPower objective,
                                image_processor::ModelType model_type);

  bool ImageCaptorSupportsAutoExposure() {
    return image_captor_->SupportsAutoExposure();
  }

  tensorflow::Status SetAutoExposureBrightness(int target_brightness) {
    return image_captor_->SetAutoExposureBrightness(target_brightness);
  }

  // Takes multiple snapshots at different brightness settings. This is used for
  // test snapshot collection.
  void TakeTestSnapshots(image_processor::ObjectiveLensPower objective,
                         image_processor::ModelType model_type,
                         int target_brightness, const std::string& comment);

  // Methods for setting testing parameters for snapshots.
  void SetEVMin(int new_min) { ev_min_ = new_min; }
  void SetEVMax(int new_max) { ev_max_ = new_max; }
  void SetEVStepsPerUnit(int new_steps_per_unit) {
    ev_steps_per_unit_ = new_steps_per_unit;
  }

  // Methods that set the positive model classes used by the inferer for
  // determining what is positive/negative.
  void SetPositiveGleasonClasses(
      const absl::flat_hash_set<image_processor::GleasonClasses>&
          positive_gleason_classes);
  void SetPositiveCervicalClasses(
      const absl::flat_hash_set<image_processor::CervicalClasses>&
          positive_cervical_classes);

 private:
  tensorflow::Status LoopOnce();
  void UpdateModelDisplayConfigs();

  std::unique_ptr<image_captor::ImageCaptor> image_captor_;
  std::unique_ptr<image_processor::Inferer> inferer_;
  std::unique_ptr<microdisplay_server::Heatmap> heatmap_;
  microdisplay_server::InferenceTimings timings_;

  // A flag indicating whether the model should be updated.
  std::atomic_bool should_update_model_ = {false};
  absl::Mutex model_lock_;
  image_processor::ObjectiveLensPower current_objective_;
  image_processor::ModelType current_model_type_;

  arm_app::Previewer* previewer_;
  arm_app::Microdisplay* microdisplay_;

  // The thread that runs the looper.
  std::unique_ptr<std::thread> thread_;

  // A flag indicating whether the looper should exit.
  std::atomic_bool to_exit_ = {false};

  DisplayWarningCallback display_warning_callback_;

  // Snapshot parameters for testing mode. These should match the initial values
  // in main_window.
  int ev_min_ = -3;
  int ev_max_ = 3;
  int ev_steps_per_unit_ = 1;
};

}  // namespace main_looper

#endif  // AR_MICROSCOPE_MAIN_LOOPER_LOOPER_H_
