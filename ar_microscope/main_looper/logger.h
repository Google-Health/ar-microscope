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
#ifndef THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_LOGGER_H_
#define THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_LOGGER_H_

#include <fstream>

#include "absl/synchronization/mutex.h"
#include "main_looper/arm_event.pb.h"

namespace main_looper {

class Logger {
 public:
  Logger();

  void LogEvent(ArmEvent event);

 private:
  absl::Mutex event_log_mutex_;
  std::ofstream event_file_;
};

Logger& GetLogger();

}  // namespace main_looper

#endif  // THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_LOGGER_H_
