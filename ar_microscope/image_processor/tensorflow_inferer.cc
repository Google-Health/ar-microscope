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
#include "image_processor/tensorflow_inferer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "arm_app/arm_config.h"
#include "image_processor/inferer.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"

ABSL_FLAG(int, num_gpus, 1, "Number of GPUs to use.");

ABSL_FLAG(bool, tf_debug, false,
          "If true, logs things like device placement...");
ABSL_FLAG(int, patch_size, 2575, "Inference patch size.");
ABSL_FLAG(std::string, output_tensor_name, "ArmOutputTensor",
          "Output heatmap tensor name.");

extern absl::Flag<int> FLAGS_image_size;

namespace image_processor {

constexpr char kTagName[] = "serve";
// Model output index that corresponds to the benign prediction.

void InputOutputBuffersWithTensor::CreateTensor(int patch_size) {
  input_tensor = absl::WrapUnique(new tensorflow::Tensor(
      tensorflow::DT_UINT8, {1, patch_size, patch_size, 3}));
  input_tensor->flat<uint8_t>().setZero();
  input_as_matrix = std::make_unique<cv::Mat>(
      patch_size, patch_size, CV_8UC3, input_tensor->flat<uint8_t>().data());
  // This is needed to avoid a dangling pointer to the old input_as_matrix.
  input_image = nullptr;
}

tensorflow::Status TensorflowInferer::Initialize(ObjectiveLensPower objective,
                                                 ModelType model_type) {
  MaybeCreateInputTensors();
  SetModelOptions();
  return LoadModel(objective, model_type);
}

cv::Mat TensorflowInferer::GetImageBuffer(int width, int height) {
  MaybeCreateInputTensors();
  CHECK(width < patch_size_)
      << "Width: " << width << "  patch size: " << patch_size_;
  CHECK(height < patch_size_)
      << "Height: " << height << "  patch size: " << patch_size_;
  int left_padding = (patch_size_ - width) / 2;
  int top_padding = (patch_size_ - height) / 2;

  cv::Rect roi(left_padding, top_padding, width, height);
  current_->CreateInputImage(roi);
  return *current_->input_image;
}

void TensorflowInferer::ProcessImageWithoutInference(cv::Mat* output) {
  auto trivial_output = cv::Mat(1, 1, CV_8UC1);
  *trivial_output.ptr(0, 0) = 0;

  current_->heatmap = std::make_unique<cv::Mat>(trivial_output.clone());
  current_->output_tensor = std::make_unique<cv::Mat>(trivial_output.clone());
  *output = trivial_output.clone();

  {
    VLOG(1) << "SWAP: input <--> previous";
    absl::MutexLock unused_lock(&tensor_mutex_);
    std::swap(current_, previous_);
  }
}

tensorflow::Status TensorflowInferer::ProcessImage(cv::Mat* output) {
  if (model_directory_.empty()) {
    ProcessImageWithoutInference(output);
    return tensorflow::Status();
  }

  CHECK(saved_model_bundle_.session) << "TensorFlow model not initialized";

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  inputs.emplace_back(input_tensor_name_, *current_->input_tensor);
  std::vector<std::string> output_tensor_names{output_tensor_name_};
  std::vector<tensorflow::Tensor> outputs;

  // Run TensorFlow inference.
  TF_RETURN_IF_ERROR(saved_model_bundle_.session->Run(
      inputs, output_tensor_names, {}, &outputs));
  CHECK(outputs.size() == output_tensor_names.size())
      << "Invalid inference output size: " << outputs.size();

  // Get the first slice of the first result. Note we are supposed to have
  // only one output Tensor.
  const tensorflow::Tensor& output_tensor = outputs[0].SubSlice(0);

  // output_tensor has 3 dimensions, 1st and 2nd for y and x of the result
  // heatmap image, and 3rd for output category.
  CHECK(output_tensor.dims() == 3)
      << "Unexpected output tensor dimension: " << output_tensor.dims();

  // Copy the requested output class values to output Mat.
  int height = output_tensor.dim_size(0);
  int width = output_tensor.dim_size(1);
  int depth = output_tensor.dim_size(2);  // Number of classes.
  auto output_tensor_shape = std::vector<int>({height, width, depth});
  current_->heatmap = std::make_unique<cv::Mat>(height, width, CV_8UC1);
  current_->output_tensor =
      std::make_unique<cv::Mat>(output_tensor_shape, CV_8UC1);
  auto output_eigen_tensor = output_tensor.tensor<uint8_t, 3>();
  const auto positive_classes = GetOutputClassesForHeatmap(model_type_);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int total_positive_pixel_value = 0;
      for (const auto positive_class : positive_classes) {
        total_positive_pixel_value += output_eigen_tensor(y, x, positive_class);
      }
      *current_->heatmap->ptr(y, x) = total_positive_pixel_value;
      for (int z = 0; z < depth; z++) {
        *current_->output_tensor->ptr(y, x, z) = output_eigen_tensor(y, x, z);
      }
    }
  }
  current_->heatmap->copyTo(*output);

  {
    VLOG(1) << "SWAP: input <--> previous";
    absl::MutexLock unused_lock(&tensor_mutex_);
    std::swap(current_, previous_);
  }

  return tensorflow::Status();
}

tensorflow::Status TensorflowInferer::LoadModel(ObjectiveLensPower power,
                                                ModelType model_type) {
  if (arm_app::GetArmConfig().IsModelConfigOverridden(model_type, power)) {
    std::string folder = arm_app::GetArmConfig()
                             .GetModelConfig(model_type, power)
                             .absolute_model_path();
    model_type_ = model_type;
    objective_ = power;
    new_input_tensors_needed_ = true;
    return SetTensorflowModel(folder);
  } else {
    model_directory_ = "";
    return tensorflow::errors::Unavailable(absl::StrFormat(
        "No model for objective lens power and model type: %s, %s",
        ObjectiveToString(power), ModelTypeToString(model_type)));
  }
}

PreviewProvider TensorflowInferer::GetPreviewProvider() {
  return [this](cv::Mat* preview, cv::Mat* heatmap, cv::Mat* output_tensor) {
    {
      VLOG(1) << "SWAP: previous <--> preview";
      absl::MutexLock unused_lock(&tensor_mutex_);
      std::swap(previous_, preview_);
    }

    if (!preview_->input_image) {
      return tensorflow::errors::NotFound("Preview image not yet ready");
    }

    preview_->input_image->copyTo(*preview);
    preview_->heatmap->copyTo(*heatmap);
    preview_->output_tensor->copyTo(*output_tensor);

    return tensorflow::Status();
  };
}

void TensorflowInferer::SetModelOptions() {
  run_options_.Clear();
  run_options_.set_trace_level(tensorflow::RunOptions::FULL_TRACE);

  session_options_ = std::make_unique<tensorflow::SessionOptions>();
  if (absl::GetFlag(FLAGS_tf_debug)) {
    session_options_->config.set_log_device_placement(true);
  }

  if (absl::GetFlag(FLAGS_num_gpus) > 0) {
    (*session_options_->config.mutable_device_count())["GPU"] = 1;
    session_options_->config.mutable_gpu_options()->set_visible_device_list(
        absl::StrCat(0));
  }

  tags_ = {kTagName};
}

tensorflow::Status TensorflowInferer::SetTensorflowModel(
    const std::string& model_directory) {
  TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(*session_options_, run_options_,
                                                model_directory, tags_,
                                                &saved_model_bundle_));

  // Get input and output tensor names from graph signature.
  const tensorflow::SignatureDef& signature_def =
      saved_model_bundle_.meta_graph_def.signature_def().at(
          tensorflow::kDefaultServingSignatureDefKey);
  input_tensor_name_ =
      signature_def.inputs().at(tensorflow::kPredictInputs).name();
  output_tensor_name_ = signature_def.outputs()
                            .at(absl::GetFlag(FLAGS_output_tensor_name))
                            .name();
  model_directory_ = model_directory;
  return tensorflow::Status();
}

void TensorflowInferer::MaybeCreateInputTensors() {
  if (!new_input_tensors_needed_) return;
  const auto& model_config =
      arm_app::GetArmConfig().GetModelConfig(model_type_, objective_);
  const int input_patch_size = model_config.input_patch_size();
  const int prediction_patch_size = model_config.prediction_patch_size();
  int new_patch_size =
      input_patch_size - prediction_patch_size +
      absl::GetFlag(FLAGS_image_size) -
      (absl::GetFlag(FLAGS_image_size) % prediction_patch_size);
  if (new_patch_size != patch_size_) {
    absl::MutexLock unused_lock(&tensor_mutex_);
    patch_size_ = new_patch_size;
    current_->CreateTensor(patch_size_);
    previous_->CreateTensor(patch_size_);
    preview_->CreateTensor(patch_size_);
  }
  new_input_tensors_needed_ = false;
}

}  // namespace image_processor
