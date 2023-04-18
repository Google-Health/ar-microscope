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
#include "main_looper/logger.h"

#include <chrono>  // NOLINT
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "main_looper/arm_event.pb.h"
#include "tensorflow/core/platform/logging.h"

extern absl::Flag<std::string> FLAGS_log_directory;

namespace main_looper {
namespace {

constexpr char kEventLogDir[] = "event_logs";
constexpr char kEventLogFilename[] = "event_log";

}  // namespace

Logger::Logger() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int epoch_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now).count();
  std::string event_filename =
      absl::StrFormat("%s/%s/%d_%s", absl::GetFlag(FLAGS_log_directory),
                      kEventLogDir, epoch_seconds, kEventLogFilename);
  event_file_.open(event_filename);
}

void Logger::LogEvent(ArmEvent event) {
  absl::MutexLock unused_lock_(&event_log_mutex_);
  if (!event_file_.is_open()) {
    LOG(ERROR) << "Error writing to event file.";
    return;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const long long timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const auto event_log =
      absl::StrFormat("%d,%s\n", timestamp, ArmEvent_Name(event));
  event_file_ << event_log << std::flush;
  LOG(INFO) << "logged event: " << event_log;
}

Logger& GetLogger() {
  static Logger* const kLogger = new Logger();
  return *kLogger;
}

}  // namespace main_looper
