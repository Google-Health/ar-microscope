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
#include "main_looper/looper.h"

#include <chrono>  // NOLINT
#include <cmath>
#include <cstdlib>
#include <memory>
#include <thread>  // NOLINT

#include "opencv2/core.hpp"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "arm_app/microdisplay.h"
#include "arm_app/previewer.h"
#include "image_captor/image_captor_factory.h"
#include "image_processor/inferer.h"
#include "image_processor/tensorflow_inferer.h"
#include "microdisplay_server/inference_timings.h"
#include "tensorflow/core/lib/core/errors.h"

ABSL_FLAG(int32_t, delay, 0,
          "Delay in milliseconds added to the end of cycles.");
ABSL_FLAG(
    int, initial_brightness, 50,
    "Initial target auto-exposure brightness as a percentage in [0, 100]. ");

extern absl::Flag<std::string> FLAGS_server_socket_name;
extern absl::Flag<bool> FLAGS_test_mode;
extern absl::Flag<int32_t> FLAGS_preview_delay;

namespace main_looper {
namespace {

using image_processor::ModelType;
using image_processor::ObjectiveLensPower;

// Preview cycles to wait for new settings to take effect.
constexpr int kPreviewWaitCycles = 2;

}  // namespace

Looper::Looper(ObjectiveLensPower objective, ModelType model_type,
               arm_app::Previewer* previewer,
               arm_app::Microdisplay* microdisplay,
               DisplayWarningCallback display_warning_callback)
    : current_objective_(objective),
      current_model_type_(model_type),
      previewer_(previewer),
      microdisplay_(microdisplay),
      display_warning_callback_(display_warning_callback) {
  UpdateModelDisplayConfigs();
  inferer_ = std::make_unique<image_processor::TensorflowInferer>();
  const auto inferer_init_status = inferer_->Initialize(objective, model_type);
  if (inferer_init_status.ok()) {
    LOG(INFO) << "Initialized inferer.";
  } else {
    LOG(WARNING) << inferer_init_status;
  }

  image_captor_ = absl::WrapUnique(image_captor::ImageCaptorFactory::Create());
  TF_CHECK_OK(image_captor_->Initialize())
      << "Image captor failed to initialize.";
  if (image_captor_->SupportsAutoExposure()) {
    TF_CHECK_OK(image_captor_->SetAutoExposureBrightness(
        absl::GetFlag(FLAGS_initial_brightness)))
        << "Image captor could not set initial auto exposure brightness.";
  }
  LOG(INFO) << "Initialized image captor.";

  previewer_->SetProvider(inferer_->GetPreviewProvider());
  previewer_->Start();
}

void Looper::Run() {
  LOG(INFO) << "Starting looper.";
  thread_ = std::make_unique<std::thread>([this]() {
    while (!to_exit_.load()) {
      tensorflow::Status result = LoopOnce();
      if (!result.ok()) {
        LOG(WARNING) << "Loop error: " << result;
      }
      int delay_milliseconds = absl::GetFlag(FLAGS_delay);
      if (delay_milliseconds > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(delay_milliseconds));
      }
    }
  });
}

void Looper::Stop() {
  to_exit_.store(true);
  thread_->join();
  thread_.reset();
}

void Looper::SetObjectiveAndModelType(ObjectiveLensPower objective,
                                      ModelType model_type) {
  absl::MutexLock model_lock(&model_lock_);
  should_update_model_.store(true);
  current_objective_ = objective;
  current_model_type_ = model_type;
}

void Looper::UpdateModelDisplayConfigs() {
  previewer_->UpdateHeatmapConfigForModel(current_model_type_,
                                          current_objective_);
  microdisplay_->UpdateHeatmapConfigForModel(current_model_type_,
                                             current_objective_);
}

tensorflow::Status Looper::LoopOnce() {
  heatmap_ = std::make_unique<microdisplay_server::Heatmap>();
  microdisplay_server::InferenceTimings::SetTimingCheckpoint(
      microdisplay_server::InferenceCheckpoint::PREPARE, heatmap_.get());
  cv::Mat debayered_image = inferer_->GetImageBuffer(
      image_captor_->GetImageWidth(), image_captor_->GetImageHeight());
  microdisplay_server::InferenceTimings::SetTimingCheckpoint(
      microdisplay_server::InferenceCheckpoint::GRAB_IMAGE, heatmap_.get());
  TF_RETURN_IF_ERROR(
      image_captor_->GetImage(/*is_rgb=*/true, &debayered_image, [this]() {
        microdisplay_server::InferenceTimings::SetTimingCheckpoint(
            microdisplay_server::InferenceCheckpoint::DEBAYER, heatmap_.get());
      }));
  microdisplay_server::InferenceTimings::SetTimingCheckpoint(
      microdisplay_server::InferenceCheckpoint::INFERENCE, heatmap_.get());

  cv::Mat heatmap_image;
  TF_RETURN_IF_ERROR(inferer_->ProcessImage(&heatmap_image));
  microdisplay_server::InferenceTimings::SetTimingCheckpoint(
      microdisplay_server::InferenceCheckpoint::DISPLAY_HEATMAP,
      heatmap_.get());
  heatmap_->set_height(heatmap_image.rows);
  heatmap_->set_width(heatmap_image.cols);
  heatmap_->set_image_binary(heatmap_image.ptr(),
                             heatmap_image.elemSize() * heatmap_image.total());

  microdisplay_->ShowHeatmap(heatmap_.get());
  microdisplay_server::InferenceTimings::SetTimingCheckpoint(
      microdisplay_server::InferenceCheckpoint::END, heatmap_.get());
  timings_.AddTiming(*heatmap_);

  if (should_update_model_.load()) {
    absl::MutexLock model_lock(&model_lock_);
    should_update_model_.store(false);
    const auto load_model_status =
        inferer_->LoadModel(current_objective_, current_model_type_);
    if (load_model_status.ok()) {
      UpdateModelDisplayConfigs();
    } else {
      display_warning_callback_(load_model_status.ToString());
    }
    return load_model_status;
  }
  return tensorflow::Status();
}

void Looper::TakeTestSnapshots(ObjectiveLensPower objective,
                               ModelType model_type, int target_brightness,
                               const std::string& comment) {
  LOG(INFO) << "Taking test snapshots for ev_min, ev_max, ev_steps: " << ev_min_
            << ", " << ev_max_ << ", " << ev_steps_per_unit_;

  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int epoch_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now).count();

  int target_exposure_time = image_captor_->GetExposureTimeInMicroseconds();
  if (target_exposure_time <= 0) {
    LOG(ERROR) << "Exposure time not retrieved";
    return;
  }
  int total_ev_steps = (ev_max_ - ev_min_) * ev_steps_per_unit_ + 1;
  float ev_step_size = 1.0 / ev_steps_per_unit_;
  for (int i = 0; i < total_ev_steps; ++i) {
    const auto snapshot_prefix =
        absl::StrFormat("%d_%s_%02d", epoch_seconds, "te", i);
    float ev_delta = ev_step_size * i + ev_min_;
    // EV = AV + TV
    //   where EV is exposure value, AV is aperture number, and TV is log2(1 /
    //   exposure time). Therefore,
    // ev_delta =
    // Test EV - Target EV =
    // log2(1 / new_exposure_time) - log2(1 / target_exposure_time)
    //   and
    // new_exposure_time = target_exposure_time / 2^ev_delta
    int sample_exposure_time =
        static_cast<int>(target_exposure_time / std::exp2(ev_delta));
    auto status = image_captor_->SetExposureTime(sample_exposure_time);
    if (status.ok()) {
      int preview_counter = previewer_->GetCounter();
      while (previewer_->GetCounter() - preview_counter < kPreviewWaitCycles) {
        // Wait for new preview.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(absl::GetFlag(FLAGS_preview_delay)));
      }
      previewer_->TakeSnapshot(objective, model_type, sample_exposure_time,
                               comment, snapshot_prefix);
      preview_counter = previewer_->GetCounter();
      while (previewer_->GetCounter() - preview_counter < kPreviewWaitCycles) {
        // Wait for snapshot to be taken by previewer before moving to next
        // exposure time.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(absl::GetFlag(FLAGS_preview_delay)));
      }
    } else {
      LOG(ERROR) << "Error setting exposure time: " << status;
      continue;
    }
  }
  auto ae_status = image_captor_->SetAutoExposureBrightness(target_brightness);
  if (!ae_status.ok()) {
    LOG(ERROR) << "Auto-exposure not reset correctly: " << ae_status;
  }
}

void Looper::SetPositiveGleasonClasses(
    const absl::flat_hash_set<image_processor::GleasonClasses>&
        positive_gleason_classes) {
  if (inferer_) {
    inferer_->SetPositiveGleasonClasses(positive_gleason_classes);
  }
}

void Looper::SetPositiveCervicalClasses(
    const absl::flat_hash_set<image_processor::CervicalClasses>&
        positive_cervical_classes) {
  if (inferer_) {
    inferer_->SetPositiveCervicalClasses(positive_cervical_classes);
  }
}

}  // namespace main_looper
