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
#include "image_captor/jenoptik_captor.h"

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "jenoptik/include/dijsdk.h"
#include "jenoptik/include/parameterif.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

ABSL_FLAG(double, white_balance_red, 2.3,
          "Jenoptik camera white balance value for red.");
ABSL_FLAG(double, white_balance_green, 1.0,
          "Jenoptik camera white balance value for green.");
ABSL_FLAG(double, white_balance_blue, 1.8,
          "Jenoptik camera white balance value for blue.");
ABSL_FLAG(int, max_over_exposure, 100,
          "Jenoptik auto-exposure setting for max over exposure. Value is a "
          "percentage in [0, 100].");

namespace image_captor {
namespace {

// Parameters for auto-exposure. These were based off the pre-set values in the
// DijSDK-app.
constexpr int kExposureLimits[] = {51, 1000000};  // Microseconds.
// Multiplicative factor. Dependent on camera.
constexpr double kGainLimits[] = {1.0, 10.0};

tensorflow::Status JenoptikErrorToStatus(error_t error,
                                         const char* error_message) {
  if (IS_OK(error)) {
    return tensorflow::Status();
  }
  const char* description;
  const char* error_name;
  DijSDK_GetErrorDescription(error, &description, &error_name);
  std::string formatted_message =
      absl::StrFormat("%s: %s (%s)", error_message, description, error_name);
  return tensorflow::errors::Internal(formatted_message);
}

tensorflow::Status SetUpAutoExposure(int brightness,
                                     DijSDK_Handle* camera_handle) {
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(*camera_handle, ParameterIdExposureControlMode,
                             DijSDK_ExposureControlOn),
      "Failed to turn auto-exposure on."));
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(*camera_handle,
                             ParameterIdExposureControlAlgorithm,
                             DijSDK_EExposureControlAlgorithmCP29),
      "Failed to set auto-exposure algorithm."));
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(*camera_handle,
                             ParameterIdExposureControlBrightnessPercentage,
                             brightness),
      "Failed to set auto-exposure brightness."));
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameterArray(*camera_handle,
                                  ParameterIdExposureControlExposureLimits,
                                  kExposureLimits, /*num=*/2),
      "Failed to set auto-exposure exposure limits."));
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetDoubleParameterArray(*camera_handle,
                                     ParameterIdExposureControlGainLimits,
                                     kGainLimits, /*num=*/2),
      "Failed to set auto-exposure gain limits."));
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(*camera_handle,
                             ParameterIdExposureControlMaxOePercentage,
                             absl::GetFlag(FLAGS_max_over_exposure)),
      "Failed to set auto-exposure max over exposure."));
  return tensorflow::Status();
}

}  // namespace

constexpr DijSDK_CameraKey kCameraKey = "C941DD58617B5CA774Bf12B70452BF23";

tensorflow::Status JenoptikCaptor::Initialize() {
  error_t error = DijSDK_Init(&kCameraKey, 1);
  TF_RETURN_IF_ERROR(
      JenoptikErrorToStatus(error, "Failed to initialize Jenoptik SDK"));

  DijSDK_CamGuid cameras[4];
  unsigned int num_cameras = 4;
  error = DijSDK_FindCameras(cameras, &num_cameras);
  LOG(INFO) << "Jenoptik " << num_cameras << " cameras found";
  if (num_cameras <= 0) {
    return tensorflow::errors::Internal("No cameras found");
  }

  // Open the first camera.
  error = DijSDK_OpenCamera(cameras[num_cameras - 1], &camera_handle_);
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(error, "Failed to open camera"));

  // Get camera name.
  char camera_name[256];
  error = DijSDK_GetStringParameter(camera_handle_,
                                    ParameterIdGlobalSettingsCameraName,
                                    camera_name, sizeof(camera_name));
  LOG(INFO) << "Camera name: " << camera_name;

  // Check width and hight of the camera sensor.
  int dimensions[2];
  error = DijSDK_GetIntParameter(camera_handle_, ParameterIdSensorSize,
                                 dimensions, 2);
  TF_RETURN_IF_ERROR(
      JenoptikErrorToStatus(error, "Failed to obtain sensor size"));
  int sensor_width = dimensions[0];
  int sensor_height = dimensions[1];
  LOG(INFO) << "Sensor size " << sensor_width << " x " << sensor_height;

  error = DijSDK_GetIntParameter(camera_handle_, ParameterIdImageModeSize,
                                 dimensions, 2);
  TF_RETURN_IF_ERROR(
      JenoptikErrorToStatus(error, "Failed to obtain mode size"));
  width_ = dimensions[0];
  height_ = dimensions[1];
  LOG(INFO) << "Captured image size " << width_ << " x " << height_;

  // Set white balance.
  TF_RETURN_IF_ERROR(SetRgbGains(absl::GetFlag(FLAGS_white_balance_red),
                                 absl::GetFlag(FLAGS_white_balance_green),
                                 absl::GetFlag(FLAGS_white_balance_blue)));

  // Set image output format.
  error = DijSDK_SetIntParameter(camera_handle_,
                                 ParameterIdImageProcessingOutputFormat,
                                 DijSDK_EImageFormatBayerRaw16);
  TF_RETURN_IF_ERROR(
      JenoptikErrorToStatus(error, "Failed to set image output format"));

  // Start image capture.
  LOG(INFO) << "Starting image acquisition";
  error = DijSDK_StartAcquisition(camera_handle_, DijSDK_EAcqModeLive);
  return JenoptikErrorToStatus(error, "Failed to start image capture");
}

tensorflow::Status JenoptikCaptor::Finalize() {
  error_t error = DijSDK_AbortAcquisition(camera_handle_);
  tensorflow::Status status =
      JenoptikErrorToStatus(error, "Failed to stop image capture");
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  error = DijSDK_CloseCamera(camera_handle_);
  status = JenoptikErrorToStatus(error, "Failed to close camera");
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  error = DijSDK_Exit();
  return JenoptikErrorToStatus(error, "Failed to exit Jenoptik SDK session");
}

tensorflow::Status JenoptikCaptor::CaptureImage(uint8_t** image) {
  error_t error = DijSDK_GetImage(camera_handle_, &image_handle_,
                                  reinterpret_cast<void**>(image));
  return JenoptikErrorToStatus(error, "Failed to get image");
}

tensorflow::Status JenoptikCaptor::ReleaseImage() {
  error_t error = DijSDK_ReleaseImage(image_handle_);
  return JenoptikErrorToStatus(error, "Failed to release image");
}

tensorflow::Status JenoptikCaptor::SetRgbGains(double red_gain,
                                               double green_gain,
                                               double blue_gain) {
  // In Jenoptik camera, hardware whitebalance settings doesn't affect
  // Bayer image. (It affects RGB image if output format is set so.)
  // We adjust RGB gains during debayer.
  debayer_.SetRgbGains(red_gain, green_gain, blue_gain);
  return tensorflow::Status();
}

tensorflow::Status JenoptikCaptor::SetAutoExposureBrightness(
    int target_brightness) {
  if (!auto_exposure_initialized_) {
    TF_RETURN_IF_ERROR(SetUpAutoExposure(target_brightness, &camera_handle_));
    auto_exposure_initialized_ = true;
  } else {
    TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
        DijSDK_SetIntParameter(camera_handle_,
                               ParameterIdExposureControlBrightnessPercentage,
                               target_brightness),
        "Failed to set auto-exposure brightness."));
  }
  return tensorflow::Status();
}

int JenoptikCaptor::GetExposureTimeInMicroseconds() {
  int microseconds;
  error_t result = DijSDK_GetIntParameter(
      camera_handle_, ParameterIdImageCaptureExposureTimeUsec, &microseconds);
  return IS_OK(result) ? microseconds : -1;
}

tensorflow::Status JenoptikCaptor::SetExposureTime(int microseconds) {
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(camera_handle_, ParameterIdExposureControlMode,
                             DijSDK_ExposureControlOff),
      "Failed to turn auto-exposure off before setting exposure time."));
  auto_exposure_initialized_ = false;
  TF_RETURN_IF_ERROR(JenoptikErrorToStatus(
      DijSDK_SetIntParameter(camera_handle_,
                             ParameterIdImageCaptureExposureTimeUsec,
                             microseconds),
      "Failed to set exposure time."));
  return tensorflow::Status();
}

}  // namespace image_captor
