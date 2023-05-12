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
#ifndef AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_
#define AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "arm_app/arm_config.pb.h"
#include "image_processor/inferer.h"
#include "tensorflow/core/lib/core/status.h"

namespace arm_app {

class ArmConfig {
 public:
  // Initializes the ArmConfig with the default and custom config files.
  tensorflow::Status Initialize(const std::string& default_config_filepath,
                                const std::string& custom_config_filepath);

  // Gets the ModelConfig for the specified model type and objective lens power.
  const ModelConfig& GetModelConfig(
      image_processor::ModelType model_type,
      image_processor::ObjectiveLensPower objective);

  // Gets the ModelTypes that were explicitly configured by users.
  const absl::flat_hash_set<image_processor::ModelType>&
  GetConfiguredModelTypes() const;

  // Determines if the ModelConfig is explicitly specified by the user.
  bool IsModelConfigOverridden(image_processor::ModelType model_type,
                               image_processor::ObjectiveLensPower objective);

  // Gets the MicrodisplayConfig.
  const MicrodisplayConfig& GetMicrodisplayConfig() const;

  // Gets the ObjectiveLensPower for the specified nosepiece position.
  image_processor::ObjectiveLensPower GetObjectiveForPosition(
      int position) const;

  // Gets the configured ObjectiveLensPower for the specified Model Type.
  const absl::flat_hash_set<image_processor::ObjectiveLensPower>&
  GetSupportedObjectivesForModelType(
      image_processor::ModelType model_type) const;

 private:
  // Initializes the ObjectivePositions map.
  void InitializeObjectivePositions(
      const ObjectivePositionConfig& objective_position_config);

  // The ArmConfigProto object.
  ArmConfigProto arm_config_proto_;

  // The map from model_type + objective key to a model config.
  absl::flat_hash_map<std::string, ModelConfig> model_config_map_;

  // The map from model_type to supported objectives.
  absl::flat_hash_map<image_processor::ModelType,
                      absl::flat_hash_set<image_processor::ObjectiveLensPower>>
      model_type_objective_map_;

  // The hash set of configured model types.
  absl::flat_hash_set<image_processor::ModelType> configured_model_types_;

  // The map from nosepiece position to objective lens.
  absl::flat_hash_map<int, image_processor::ObjectiveLensPower>
      objective_from_position_map_;
};

// Gets the ArmConfig singleton. Must be initialized before first use.
ArmConfig& GetArmConfig();

}  // namespace arm_app

#endif  // AR_MICROSCOPE_ARM_APP_ARM_CONFIG_H_
