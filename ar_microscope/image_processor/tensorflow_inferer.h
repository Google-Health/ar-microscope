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

#ifndef AR_MICROSCOPE_IMAGE_PROCESSOR_TENSORFLOW_INFERER_H_
#define AR_MICROSCOPE_IMAGE_PROCESSOR_TENSORFLOW_INFERER_H_

#ifndef NO_TENSORFLOW

#include <functional>
#include <memory>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "absl/synchronization/mutex.h"
#include "image_processor/inferer.h"
#include "microdisplay_server/heatmap_util.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace image_processor {

// Input and output tensor for TensorFlow inference.
struct InputOutputBuffersWithTensor : public InputOutputBuffers {
  virtual ~InputOutputBuffersWithTensor() {}

  // Tensor as input of TensorFlow inference.
  std::unique_ptr<tensorflow::Tensor> input_tensor;

  void CreateTensor(int patch_size) override;
};

class TensorflowInferer : public Inferer {
 public:
  TensorflowInferer() {
    current_ = &buffers[0];
    previous_ = &buffers[1];
    preview_ = &buffers[2];
  }

  virtual ~TensorflowInferer() {}

  virtual tensorflow::Status Initialize(ObjectiveLensPower objective,
                                        ModelType model_type);
  virtual cv::Mat GetImageBuffer(int width, int height);
  virtual tensorflow::Status ProcessImage(cv::Mat* output);
  virtual tensorflow::Status LoadModel(ObjectiveLensPower power,
                                       ModelType model_type);

  // Returns PreviewProvider function, which returns the latest preview
  // upon request.
  PreviewProvider GetPreviewProvider() override;

 private:
  void SetModelOptions();
  tensorflow::Status SetTensorflowModel(const std::string& model_directory);
  void MaybeCreateInputTensors();

  void ProcessImageWithoutInference(cv::Mat* output);

  std::unordered_set<std::string> tags_;
  std::unique_ptr<tensorflow::SessionOptions> session_options_;
  tensorflow::RunOptions run_options_;

  tensorflow::SavedModelBundle saved_model_bundle_;
  std::string input_tensor_name_;
  std::string output_tensor_name_;

 private:
  // Input tensor, cv::Mat of input image as view of the tensor's part,
  // and output heatmap. We keep 3 sets of these to minimize lock for preview.
  // We have 2 threads, TensorFlow inference thread and preview thread.
  // Inference thread swaps "current" with "previous" when the inference is
  // finished. Preview thread swaps "previous" with "preview" and work on
  // "preview". In this way, inference thread can work on "current" and preview
  // thread can work on "preview" concurrently without waiting for the lock.
  // The only lock we have is for swapping the data sets.
  InputOutputBuffersWithTensor buffers[3];

  InputOutputBuffersWithTensor* current_;
  InputOutputBuffersWithTensor* previous_;
  InputOutputBuffersWithTensor* preview_;

  // Boolean for deciding whether to possibly create new input tensors for a
  // newly loaded model.
  bool new_input_tensors_needed_ = true;
};

}  // namespace image_processor

#endif  // NO_TENSORFLOW
#endif  // AR_MICROSCOPE_IMAGE_PROCESSOR_TENSORFLOW_INFERER_H_
