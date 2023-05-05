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
#ifndef AR_MICROSCOPE_MICRODISPLAY_SERVER_INFERENCE_TIMINGS_H_
#define AR_MICROSCOPE_MICRODISPLAY_SERVER_INFERENCE_TIMINGS_H_

#include <vector>

#include "absl/time/time.h"
#include "microdisplay_server/heatmap.pb.h"

namespace microdisplay_server {

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

#endif  // AR_MICROSCOPE_MICRODISPLAY_SERVER_INFERENCE_TIMINGS_H_