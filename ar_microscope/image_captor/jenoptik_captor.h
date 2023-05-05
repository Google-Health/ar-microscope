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
#ifndef AR_MICROSCOPE_IMAGE_CAPTOR_JENOPTIK_CAPTOR_H_
#define AR_MICROSCOPE_IMAGE_CAPTOR_JENOPTIK_CAPTOR_H_

#include "jenoptik/include/dijsdk.h"
#include "jenoptik/include/dijsdkerror.h"
#include "image_captor/image_captor.h"
#include "tensorflow/core/lib/core/status.h"

namespace image_captor {

class JenoptikCaptor : public ImageCaptor {
 public:
  tensorflow::Status Initialize() override;
  tensorflow::Status Finalize() override;

  int GetSensorWidth() override { return width_; }

  int GetSensorHeight() override { return height_; }

  bool SupportsAutoExposure() override { return true; }

  tensorflow::Status SetAutoExposureBrightness(int target_brightness) override;

  int GetExposureTimeInMicroseconds() override;

  tensorflow::Status SetExposureTime(int microseconds) override;

 protected:
  tensorflow::Status CaptureImage(uint8_t** image) override;
  tensorflow::Status ReleaseImage() override;

  int GetBytesPerPixel() override { return 2; }

  tensorflow::Status SetRgbGains(double red_gain, double green_gain,
                                 double blue_gain) override;

 private:
  int width_;
  int height_;
  DijSDK_Handle camera_handle_;
  DijSDK_Handle image_handle_;
  bool auto_exposure_initialized_ = false;
};

}  // namespace image_captor

#endif  // AR_MICROSCOPE_IMAGE_CAPTOR_JENOPTIK_CAPTOR_H_
