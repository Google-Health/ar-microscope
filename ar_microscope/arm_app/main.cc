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
#include <QApplication>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "arm_app/arm_config.h"
#include "arm_app/main_window.h"
#include "arm_app/microdisplay.h"
#include "main_looper/arm_event.pb.h"
#include "main_looper/logger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

ABSL_FLAG(std::string, log_directory, "/home/arm/arm_logs",
          "Directory to store ARM logs and snapshots.");

ABSL_FLAG(std::string, default_config_file, "",
          "Path to ArmConfigProto textproto file with default settings.");

ABSL_FLAG(std::string, custom_config_file, "",
          "Path to ArmConfigProto textproto file with custom settings.");

ABSL_FLAG(bool, test_mode, false,
          "Test mode, which enables multiple snapshots for testing.");

ABSL_FLAG(bool, calibration_mode, false,
          "Calibration mode, which allows for the display of a calibration "
          "target that facilitates adjusting the microdisplay offset "
          "parameters in the ARM config.");

ABSL_FLAG(bool, aos, true,
          "Whether to use automatic objective switching (AOS). The objective "
          "control box must be installed for this; otherwise, it defaults to "
          "manual objective switching.");

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Augmented Reality Microscope");

  auto residual_args = absl::ParseCommandLine(argc, argv);
  int new_argc = residual_args.size();

  main_looper::GetLogger().LogEvent(main_looper::ArmEvent::ARM_START);

  auto& arm_config = arm_app::GetArmConfig();
  const auto config_init_status =
      arm_config.Initialize(absl::GetFlag(FLAGS_default_config_file),
                            absl::GetFlag(FLAGS_custom_config_file));
  if (!config_init_status.ok()) {
    LOG(ERROR) << "Failed to initialize ARM config from "
               << absl::GetFlag(FLAGS_default_config_file) << " and "
               << absl::GetFlag(FLAGS_custom_config_file) << " with error "
               << config_init_status;
  }

  QApplication app(new_argc, argv);

  arm_app::Microdisplay microdisplay;
  arm_app::MainWindow main_window(&microdisplay);

  main_window.show();
  microdisplay.show();

  return app.exec();
}
