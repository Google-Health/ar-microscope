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
#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_

#include <memory>

#include "opencv2/core.hpp"
#include "tensorflow/core/lib/core/status.h"

namespace image_processor {

// Base abstract class to process captured image.
class ImageProcessor {
 public:
  virtual ~ImageProcessor();

  virtual tensorflow::Status Initialize() { return tensorflow::Status(); }
  virtual tensorflow::Status Finalize() { return tensorflow::Status(); }

  virtual tensorflow::Status ProcessImage(const cv::Mat& input,
                                          cv::Mat* output) = 0;
};

}  // namespace image_processor

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_
