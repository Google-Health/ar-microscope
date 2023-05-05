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
#ifndef AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_FACTORY_H_
#define AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_FACTORY_H_

#include "image_captor/image_captor.h"

namespace image_captor {

class ImageCaptorFactory {
 public:
  ImageCaptorFactory() = delete;

  static ImageCaptor* Create();
};

}  // namespace image_captor

#endif  // AR_MICROSCOPE_IMAGE_CAPTOR_IMAGE_CAPTOR_FACTORY_H_
