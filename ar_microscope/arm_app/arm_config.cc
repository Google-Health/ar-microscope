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
#include "arm_app/arm_config.h"

#include <fcntl.h>

#include <iostream>

#include "absl/strings/str_cat.h"
#include "arm_app/arm_config.pb.h"
#include "image_processor/inferer.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace arm_app {
namespace {

using ::google::protobuf::TextFormat;
using ::google::protobuf::io::FileInputStream;

std::string MakeModelKey(const std::string& model_type,
                         const std::string& objective) {
  return absl::StrCat(model_type, "_", objective);
}

std::string MakeModelKey(image_processor::ModelType model_type,
                         image_processor::ObjectiveLensPower objective) {
  return MakeModelKey(image_processor::ModelTypeToString(model_type),
                      image_processor::ObjectiveToString(objective));
}

std::string GetDefaultKey() {
  return MakeModelKey(
      image_processor::ModelType::UNSPECIFIED_MODEL_TYPE,
      image_processor::ObjectiveLensPower::UNSPECIFIED_OBJECTIVE_LENS_POWER);
}

tensorflow::StatusOr<ArmConfigProto> ParseConfigFromFile(
    const std::string& config_filepath) {
  const int config_file_descriptor = open(config_filepath.c_str(), O_RDONLY);
  if (config_file_descriptor < 0) {
    return tensorflow::errors::Unavailable(config_filepath +
                                           " is not available.");
  }
  FileInputStream config_fstream(config_file_descriptor);
  ArmConfigProto arm_config_proto;
  TextFormat::Parse(&config_fstream, &arm_config_proto);
  return arm_config_proto;
}

}  // namespace

const MicrodisplayConfig& ArmConfig::GetMicrodisplayConfig() const {
  return arm_config_proto_.microdisplay_config();
}

void ArmConfig::InitializeObjectivePositions(
    const ObjectivePositionConfig& objective_position_config) {
  
  if (objective_position_config.has_position_2x()) {
    objective_from_position_map_.emplace(
        objective_position_config.position_2x(),
        image_processor::ObjectiveLensPower::OBJECTIVE_2x);
  }
  if (objective_position_config.has_position_4x()) {
    objective_from_position_map_.emplace(
        objective_position_config.position_4x(),
        image_processor::ObjectiveLensPower::OBJECTIVE_4x);
  }
  if (objective_position_config.has_position_10x()) {
    objective_from_position_map_.emplace(
        objective_position_config.position_10x(),
        image_processor::ObjectiveLensPower::OBJECTIVE_10x);
  }
  if (objective_position_config.has_position_20x()) {
    objective_from_position_map_.emplace(
        objective_position_config.position_20x(),
        image_processor::ObjectiveLensPower::OBJECTIVE_20x);
  }
  if (objective_position_config.has_position_40x()) {
    objective_from_position_map_.emplace(
        objective_position_config.position_40x(),
        image_processor::ObjectiveLensPower::OBJECTIVE_40x);
  }
}

tensorflow::Status ArmConfig::Initialize(
    const std::string& default_config_filepath,
    const std::string& custom_config_filepath) {
  const auto default_config_or = ParseConfigFromFile(default_config_filepath);
  const auto custom_config_or = ParseConfigFromFile(custom_config_filepath);
  if (!default_config_or.ok()) {
    return default_config_or.status();
  }
  arm_config_proto_ = *default_config_or;
  // Override default config with the optional custom config if it exists.
  if (custom_config_or.ok()) {
    arm_config_proto_.MergeFrom(*custom_config_or);
  }
  if (!arm_config_proto_.has_model_config_default()) {
    return tensorflow::errors::FailedPrecondition(
        "No default model config provided at " + default_config_filepath);
  } else {
    model_config_map_[GetDefaultKey()] =
        arm_config_proto_.model_config_default();
  }
  for (const auto& model_config : arm_config_proto_.custom_model_configs()) {
    if (model_config.model_type().empty() || model_config.objective().empty()) {
      continue;
    }
    ModelConfig config_with_defaults = model_config_map_[GetDefaultKey()];
    config_with_defaults.MergeFrom(model_config);
    model_config_map_[MakeModelKey(model_config.model_type(),
                                   model_config.objective())] =
        config_with_defaults;
  }
  if (!arm_config_proto_.has_objective_positions()) {
    return tensorflow::errors::FailedPrecondition(
        "No objective positions configs provided at " +
        default_config_filepath);
  } else {
    InitializeObjectivePositions(arm_config_proto_.objective_positions());
  }
  LOG(INFO) << "ARM being run with the following config:\n"
            << arm_config_proto_.DebugString();
  return tensorflow::Status();
}

const ModelConfig& ArmConfig::GetModelConfig(
    image_processor::ModelType model_type,
    image_processor::ObjectiveLensPower objective) {
  auto it = model_config_map_.find(MakeModelKey(model_type, objective));
  if (it == model_config_map_.end()) {
    return model_config_map_[GetDefaultKey()];
  } else {
    return it->second;
  }
}

const image_processor::ObjectiveLensPower ArmConfig::GetObjectiveForPosition(
    int position) const {
  auto it = objective_from_position_map_.find(position);
  if (it == objective_from_position_map_.end()) {
    return image_processor::ObjectiveLensPower::
        UNSPECIFIED_OBJECTIVE_LENS_POWER;
  } else {
    return it->second;
  }
}

ArmConfig& GetArmConfig() {
  static ArmConfig* const kArmConfig = new ArmConfig();

  return *kArmConfig;
}

}  // namespace arm_app
