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
#ifndef THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_
#define THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "arm_app/arm_config.pb.h"
#include "image_processor/image_processor.h"
#include "image_processor/inferer.h"
#include "tensorflow/core/lib/core/status.h"

namespace arm_app {

class ArmConfig {
 public:
  tensorflow::Status Initialize(const std::string& default_config_filepath,
                                const std::string& custom_config_filepath);

  const ModelConfig& GetModelConfig(
      image_processor::ModelType model_type,
      image_processor::ObjectiveLensPower objective);

  const MicrodisplayConfig& GetMicrodisplayConfig() {
    return arm_config_proto_.microdisplay_config();
  }

  const image_processor::ObjectiveLensPower GetObjectiveForPosition(
      int position);

 private:
  void InitializeObjectivePositions(
      const ObjectivePositionConfig& objective_position_config);

  ArmConfigProto arm_config_proto_;
  // Map from model_type + objective key to a model config.
  absl::flat_hash_map<std::string, ModelConfig> model_config_map_;
  // Map from nosepiece position to objective lens.
  absl::flat_hash_map<int, image_processor::ObjectiveLensPower>
      objective_from_position_map_;
};

// Method for getting ArmConfig singleton. Must be initialized before first use.
ArmConfig& GetArmConfig();

}  // namespace arm_app

#endif  // THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_
