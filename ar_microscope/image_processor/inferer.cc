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
#include "image_processor/inferer.h"

#include <memory>
#include <string>

#include "opencv2/imgproc.hpp"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"

namespace image_processor {
namespace {

const absl::flat_hash_map<ObjectiveLensPower, std::string>
    kObjectiveLensPowerToString = {
        {ObjectiveLensPower::UNSPECIFIED_OBJECTIVE_LENS_POWER, "Unspecified"},
        {ObjectiveLensPower::OBJECTIVE_2x, "2x"},
        {ObjectiveLensPower::OBJECTIVE_4x, "4x"},
        {ObjectiveLensPower::OBJECTIVE_10x, "10x"},
        {ObjectiveLensPower::OBJECTIVE_20x, "20x"},
        {ObjectiveLensPower::OBJECTIVE_40x, "40x"},
};
const absl::flat_hash_map<std::string, ObjectiveLensPower>
    kStringToObjectiveLensPower = {
        {"Unspecified", ObjectiveLensPower::UNSPECIFIED_OBJECTIVE_LENS_POWER},
        {"2x", ObjectiveLensPower::OBJECTIVE_2x},
        {"4x", ObjectiveLensPower::OBJECTIVE_4x},
        {"10x", ObjectiveLensPower::OBJECTIVE_10x},
        {"20x", ObjectiveLensPower::OBJECTIVE_20x},
        {"40x", ObjectiveLensPower::OBJECTIVE_40x},
};
constexpr char kUnknownModel[] = "unknown-model";

const absl::flat_hash_map<std::string, ModelType> kStringToModelType = {
    {"lymph", ModelType::LYNA},
    {"prostate", ModelType::GLEASON},
    {"mitotic", ModelType::MITOTIC},
    {"cervical", ModelType::CERVICAL},
};

const absl::flat_hash_map<ModelType, std::string> kModelTypeToString = {
    {ModelType::LYNA, "lymph"},
    {ModelType::GLEASON, "prostate"},
    {ModelType::MITOTIC, "mitotic"},
    {ModelType::CERVICAL, "cervical"},
};
constexpr char kUnspecifiedModelType[] = "UNSPECIFIED_MODEL_TYPE";

const absl::flat_hash_map<ModelType, std::string> kModelTypeToPrettyString = {
    {ModelType::LYNA, "Lymph"},
    {ModelType::GLEASON, "Prostate"},
    {ModelType::MITOTIC, "Mitotic"},
    {ModelType::CERVICAL, "Cervical"},
};
constexpr char kUnspecifiedPrettyModelType[] = "Unknown Model";

}  // namespace

std::string ObjectiveToString(ObjectiveLensPower objective) {
  auto it = kObjectiveLensPowerToString.find(objective);

  // If the iterator points to the end of the map, the key is not found.
  if (it == kObjectiveLensPowerToString.end()) {
    return kUnknownModel;
  } else {
    return it->second;
  }
}

ObjectiveLensPower StringToObjective(std::string objective) {
  auto it = kStringToObjectiveLensPower.find(objective);

  // If the iterator points to the end of the map, the key is not found.
  if (it == kStringToObjectiveLensPower.end()) {
    return ObjectiveLensPower::UNSPECIFIED_OBJECTIVE_LENS_POWER;
  } else {
    return it->second;
  }
}

std::string ModelTypeToString(ModelType model_type) {
  auto it = kModelTypeToString.find(model_type);

  // If the iterator points to the end of the map, the key is not found.
  if (it == kModelTypeToString.end()) {
    return kUnspecifiedModelType;
  } else {
    return it->second;
  }
}

ModelType StringToModelType(std::string model_type) {
  auto it = kStringToModelType.find(model_type);

  // If the iterator points to the end of the map, the key is not found.
  if (it == kStringToModelType.end()) {
    return ModelType::UNSPECIFIED_MODEL_TYPE;
  } else {
    return it->second;
  }
}

std::string ModelTypeToPrettyString(ModelType model_type) {
  auto it = kModelTypeToPrettyString.find(model_type);

  // If the iterator points to the end of the map, the key is not found.
  if (it == kModelTypeToPrettyString.end()) {
    return kUnspecifiedPrettyModelType;
  } else {
    return it->second;
  }
}

absl::flat_hash_set<int> Inferer::GetOutputClassesForHeatmap(
    ModelType model_type) {
  if (model_type == ModelType::LYNA) {
    return {static_cast<int>(LynaClasses::TUMOR)};
  } else if (model_type == ModelType::GLEASON) {
    absl::flat_hash_set<int> temp;
    for (const auto& gleason_class : positive_gleason_classes_) {
      temp.emplace(static_cast<int>(gleason_class));
    }
    return temp;
  } else if (model_type == ModelType::MITOTIC) {
    return {static_cast<int>(MitoticClasses::MITOTIC)};
  } else if (model_type == ModelType::CERVICAL) {
    absl::flat_hash_set<int> temp;
    for (const auto& cervical_class : positive_cervical_classes_) {
      temp.emplace(static_cast<int>(cervical_class));
    }
    return temp;
  } else {
    return {};
  }
}

void Inferer::SetPositiveGleasonClasses(
    const absl::flat_hash_set<GleasonClasses>& positive_gleason_classes) {
  positive_gleason_classes_ = positive_gleason_classes;
  LOG(INFO) << "Updated gleason classes.";
}

void Inferer::SetPositiveCervicalClasses(
    const absl::flat_hash_set<CervicalClasses>& positive_cervical_classes) {
  positive_cervical_classes_ = positive_cervical_classes;
  LOG(INFO) << "Updated cervical classes.";
}

void InputOutputBuffers::CreateTensor(int patch_size) {
  input_as_matrix = std::make_unique<cv::Mat>(patch_size, patch_size, CV_8UC3);
  // This is needed to avoid a dangling pointer to the old input_as_matrix.
  input_image = nullptr;
}

void InputOutputBuffers::CreateInputImage(const cv::Rect& roi) {
  input_image = std::make_unique<cv::Mat>(*input_as_matrix, roi);
}

}  // namespace image_processor
