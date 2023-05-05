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
#include "microdisplay_server/inference_timings.h"


#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "microdisplay_server/heatmap.pb.h"
#include "tensorflow/core/platform/logging.h"


ABSL_FLAG(int, show_stats_every_n, 50,
          "Show timing stats for every N inference.");

namespace microdisplay_server {

InferenceTimings::InferenceTimings() { Clear(); }

void InferenceTimings::AddTiming(const Heatmap& heatmap) {
  if (heatmap.timing_size() != InferenceCheckpoint::Type_MAX) {
    LOG(ERROR) << "Invalid number of timings. " << InferenceCheckpoint::Type_MAX
               << " expected, but actual " << heatmap.timing_size();
    return;
  }
  count_++;
  total_ += GetCheckpoint(heatmap, InferenceCheckpoint::Type_MAX) -
            GetCheckpoint(heatmap, static_cast<InferenceCheckpoint::Type>(1));
  for (int i = 1; i < InferenceCheckpoint::Type_MAX; i++) {
    steps_[i] +=
        GetCheckpoint(heatmap, static_cast<InferenceCheckpoint::Type>(i + 1)) -
        GetCheckpoint(heatmap, static_cast<InferenceCheckpoint::Type>(i));
  }
  if (count_ >= absl::GetFlag(FLAGS_show_stats_every_n)) {
    LOG(INFO) << "Timing stats (average) for " << count_ << " captures";
    LOG(INFO) << "  Total: " << absl::ToInt64Milliseconds(total_ / count_)
              << " ms";
    LOG(INFO) << "    Prepare: "
              << GetAverageDurationTime(InferenceCheckpoint::PREPARE);
    LOG(INFO) << "    Grab image: "
              << GetAverageDurationTime(InferenceCheckpoint::GRAB_IMAGE);
    LOG(INFO) << "    Debayer: "
              << GetAverageDurationTime(InferenceCheckpoint::DEBAYER);
    LOG(INFO) << "    Inference: "
              << GetAverageDurationTime(InferenceCheckpoint::INFERENCE);
    LOG(INFO) << "    Display heatmap: "
              << GetAverageDurationTime(InferenceCheckpoint::DISPLAY_HEATMAP);
    Clear();
  }
}

void InferenceTimings::SetTimingCheckpoint(InferenceCheckpoint::Type type,
                                           Heatmap* heatmap) {
  Timing* timing = heatmap->add_timing();
  timing->set_checkpoint_type(type);
  timing->set_timestamp_microseconds(absl::ToUnixMicros(absl::Now()));
}

void InferenceTimings::Clear() {
  count_ = 0;
  total_ = absl::ZeroDuration();
  steps_.clear();
  steps_.resize(static_cast<int>(InferenceCheckpoint::Type_MAX));
}

const absl::Time InferenceTimings::GetCheckpoint(
    const Heatmap& heatmap, InferenceCheckpoint::Type type) {
  const Timing& timing = heatmap.timing(static_cast<int>(type) - 1);
  CHECK(timing.checkpoint_type() == type);
  return absl::FromUnixMicros(timing.timestamp_microseconds());
}

std::string InferenceTimings::GetAverageDurationTime(
    InferenceCheckpoint::Type type) {
  // Convert microseconds to milliseconds.
  int64_t milliseconds =
      absl::ToInt64Milliseconds(steps_[static_cast<int>(type)] / count_);
  if (milliseconds == 0) {
    return std::string("<1 ms");
  } else {
    return absl::StrFormat("%d ms", milliseconds);
  }
}

}  // namespace microdisplay_server
