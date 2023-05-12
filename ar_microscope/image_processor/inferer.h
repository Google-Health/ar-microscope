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
// Class to run TensorFlow inference against the given image.

#ifndef AR_MICROSCOPE_IMAGE_PROCESSOR_INFERER_H_
#define AR_MICROSCOPE_IMAGE_PROCESSOR_INFERER_H_

#include <functional>
#include <memory>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/lib/core/status.h"

namespace image_processor {

using PreviewProvider =
    std::function<tensorflow::Status(cv::Mat*, cv::Mat*, cv::Mat*)>;

enum class ObjectiveLensPower {
  UNSPECIFIED_OBJECTIVE_LENS_POWER,
  OBJECTIVE_2x,
  OBJECTIVE_4x,
  OBJECTIVE_10x,
  OBJECTIVE_20x,
  OBJECTIVE_40x,
};

std::string ObjectiveToString(ObjectiveLensPower objective);

ObjectiveLensPower StringToObjective(std::string objective);

enum class ModelType {
  UNSPECIFIED_MODEL_TYPE,
  LYNA,
  GLEASON,
  MITOTIC,
  CERVICAL,
};

std::string ModelTypeToString(ModelType model_type);

std::string ModelTypeToPrettyString(ModelType model_type);

ModelType StringToModelType(std::string model_type);

// Classes of the output indices of the Lyna model.
enum class LynaClasses : int {
  BENIGN = 0,
  TUMOR = 1,
};

// Classes of the output indices of the Gleason model.
enum class GleasonClasses : int {
  BENIGN = 0,
  GP_3 = 1,
  GP_4 = 2,
  GP_5 = 3,
};

// Classes of the output indices of the Mitotic model.
enum class MitoticClasses : int {
  BENIGN = 0,
  MITOTIC = 1,
};

// Classes of the output indices of the Cervical model.
enum class CervicalClasses : int {
  BENIGN = 0,
  CIN_1 = 1,
  CIN_2_PLUS = 2,
};

// Input and output data buffers for inference.
struct InputOutputBuffers {
  virtual ~InputOutputBuffers() {}

  // Sub-matrix of input_as_matrix_ for the area of input image (i.e.
  // excluding the padding area around the image).
  std::unique_ptr<cv::Mat> input_image;

  // Output heatmap.
  std::unique_ptr<cv::Mat> heatmap;

  // Output tensor. Used for models with more than two output classes.
  std::unique_ptr<cv::Mat> output_tensor;

  // Matrix that represents the whole input_tensor area.
  std::unique_ptr<cv::Mat> input_as_matrix;

  void CreateInputImage(const cv::Rect& roi);

  // Helper hook that child classes can use to update temporary buffers
  // etc when the patch size changes
  virtual void CreateTensor(int patch_size);
};

class Inferer {
 public:
  Inferer() {}
  virtual ~Inferer() {}
  virtual tensorflow::Status Initialize(ObjectiveLensPower objective,
                                        ModelType model_type) = 0;

  virtual cv::Mat GetImageBuffer(int width, int height) = 0;

  virtual tensorflow::Status ProcessImage(cv::Mat* output) = 0;
  virtual tensorflow::Status LoadModel(ObjectiveLensPower power,
                                       ModelType model_type) = 0;

  // Returns PreviewProvider function, which returns the latest preview
  // upon request.
  virtual PreviewProvider GetPreviewProvider() = 0;

  void SetPositiveGleasonClasses(
      const absl::flat_hash_set<GleasonClasses>& positive_gleason_classes);

  void SetPositiveCervicalClasses(
      const absl::flat_hash_set<CervicalClasses>& positive_cervical_classes);

 protected:
  void RenderPreviewHeatmapContour(cv::Mat* preview_image, cv::Mat* heatmap);

  // Returns the class indices of the inference output that should be
  // highlighted in the heatmap.
  absl::flat_hash_set<int> GetOutputClassesForHeatmap(ModelType model_type);

  // Input tensor, cv::Mat of input image as view of the tensor's part,
  // and output heatmap. We keep 3 sets of these to minimize lock for preview.
  // We have 2 threads, TensorFlow inference thread and preview thread.
  // Inference thread swaps "current" with "previous" when the inference is
  // finished. Preview thread swaps "previous" with "preview" and work on
  // "preview". In this way, inference thread can work on "current" and preview
  // thread can work on "preview" concurrently without waiting for the lock.
  // The only lock we have is for swapping the data sets.

  absl::Mutex tensor_mutex_;
  int patch_size_;
  ModelType model_type_;
  ObjectiveLensPower objective_;
  // Store the current model directory. For some `model_type_` and `objective_`
  // combinations, there is no model, and we do not apply model inference.
  std::string model_directory_;

  // Model classes considered to be positive for heatmapping. The initial values
  // should be in sync with the initial values displayed in the UI.
  absl::flat_hash_set<GleasonClasses> positive_gleason_classes_ = {
      GleasonClasses::GP_3, GleasonClasses::GP_4, GleasonClasses::GP_5};
  absl::flat_hash_set<CervicalClasses> positive_cervical_classes_ = {
      CervicalClasses::CIN_2_PLUS};
};

}  // namespace image_processor

#endif  // AR_MICROSCOPE_IMAGE_PROCESSOR_INFERER_H_
