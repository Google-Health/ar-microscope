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

#include "opencv2/imgproc.hpp"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"

namespace image_processor {
namespace {

// Model versions (for logging).
constexpr char kGleason2x[] = "gleason-sensitivity-20220708";
constexpr char kGleason4x[] = "gleason-sensitivity-20220718";
constexpr char kGleason10x[] = "gleason-sensitivity-20201012";
constexpr char kGleason20x[] = "gleason-sensitivity-20201013";
constexpr char kLyna2x[] = "lyna-sensitivity-20220629";
constexpr char kLyna4x[] = "lyna-sensitivity-20220629";
constexpr char kLyna10x[] = "lyna-sensitivity-20201012";
constexpr char kLyna20x[] = "lyna-sensitivity-20201008";
constexpr char kLyna40x[] = "lyna-sensitivity-20201008";
constexpr char kMitotic40x[] = "20210622_combined_mc_arm";
constexpr char kCervical2x[] = "20220708_cd_arm";
constexpr char kCervical4x[] = "20220708_cd_arm";
constexpr char kCervical10x[] = "20211206_arm";
constexpr char kCervical20x[] = "20211206_arm";
constexpr char kCervical40x[] = "20211206_arm";

constexpr char kUnknownModel[] = "unknown-model";

}  // namespace

std::string ObjectiveToString(ObjectiveLensPower objective) {
  switch (objective) {
    case ObjectiveLensPower::OBJECTIVE_2x:
      return "2x";
    case ObjectiveLensPower::OBJECTIVE_4x:
      return "4x";
    case ObjectiveLensPower::OBJECTIVE_10x:
      return "10x";
    case ObjectiveLensPower::OBJECTIVE_20x:
      return "20x";
    case ObjectiveLensPower::OBJECTIVE_40x:
      return "40x";
    default:
      return "UNSPECIFIED_OBJECTIVE_LENS_POWER";
  }
}

std::string ModelTypeToString(ModelType model_type) {
  switch (model_type) {
    case ModelType::LYNA:
      return "lymph";
    case ModelType::GLEASON:
      return "prostate";
    case ModelType::MITOTIC:
      return "mitotic";
    case ModelType::CERVICAL:
      return "cervical";
    default:
      return "UNSPECIFIED_MODEL_TYPE";
  }
}

std::string GetModelVersion(ModelType model_type,
                            ObjectiveLensPower objective) {
  if (model_type == ModelType::LYNA &&
      objective == ObjectiveLensPower::OBJECTIVE_2x) {
    return kLyna2x;
  } else if (model_type == ModelType::LYNA &&
      objective == ObjectiveLensPower::OBJECTIVE_4x) {
    return kLyna4x;
  } else if (model_type == ModelType::LYNA &&
      objective == ObjectiveLensPower::OBJECTIVE_10x) {
    return kLyna10x;
  } else if (model_type == ModelType::LYNA &&
             objective == ObjectiveLensPower::OBJECTIVE_20x) {
    return kLyna20x;
  } else if (model_type == ModelType::LYNA &&
             objective == ObjectiveLensPower::OBJECTIVE_40x) {
    return kLyna40x;
  } else if (model_type == ModelType::GLEASON &&
             objective == ObjectiveLensPower::OBJECTIVE_2x) {
    return kGleason2x;
  } else if (model_type == ModelType::GLEASON &&
             objective == ObjectiveLensPower::OBJECTIVE_4x) {
    return kGleason4x;
  } else if (model_type == ModelType::GLEASON &&
             objective == ObjectiveLensPower::OBJECTIVE_10x) {
    return kGleason10x;
  } else if (model_type == ModelType::GLEASON &&
             objective == ObjectiveLensPower::OBJECTIVE_20x) {
    return kGleason20x;
  } else if (model_type == ModelType::MITOTIC &&
             objective == ObjectiveLensPower::OBJECTIVE_40x) {
    return kMitotic40x;
  } else if (model_type == ModelType::CERVICAL &&
             objective == ObjectiveLensPower::OBJECTIVE_2x) {
    return kCervical2x;
  } else if (model_type == ModelType::CERVICAL &&
             objective == ObjectiveLensPower::OBJECTIVE_4x) {
    return kCervical4x;
  } else if (model_type == ModelType::CERVICAL &&
             objective == ObjectiveLensPower::OBJECTIVE_10x) {
    return kCervical10x;
  } else if (model_type == ModelType::CERVICAL &&
             objective == ObjectiveLensPower::OBJECTIVE_20x) {
    return kCervical20x;
  } else if (model_type == ModelType::CERVICAL &&
             objective == ObjectiveLensPower::OBJECTIVE_40x) {
    return kCervical40x;
  } else {
    return kUnknownModel;
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
