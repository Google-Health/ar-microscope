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
#include "image_processor/debayer.h"

#include <thread>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

ABSL_FLAG(int, num_debayer_threads, 8,
          "Number of debayer threads for quick debayer.");
ABSL_FLAG(bool, smooth_image, false,
          "Whether to perform image smoothing after debayer.");

namespace image_processor {

Debayer::Debayer() {
  num_debayer_threads_ = absl::GetFlag(FLAGS_num_debayer_threads);
}

tensorflow::Status Debayer::HalfDebayer(const cv::Mat& input, bool is_rgb,
                                        cv::Mat* output) {
  switch (input.elemSize()) {
    case 1:
      TF_RETURN_IF_ERROR(HalfDebayerInternal<uint8_t>(input, is_rgb, output));
      break;
    case 2:
      TF_RETURN_IF_ERROR(HalfDebayerInternal<uint16_t>(input, is_rgb, output));
      break;
    default:
      LOG(FATAL) << "Unsupported Bayer pixel byte size: " << input.elemSize();
  }

  // Blur the image for smoothing.
  if (absl::GetFlag(FLAGS_smooth_image)) {
    cv::Mat tmp;
    output->copyTo(tmp);
    cv::blur(tmp, *output, cv::Size(3, 3));
  }

  return tensorflow::Status();
}

void Debayer::SetRgbGains(double red, double green, double blue) {
  has_gain_ = true;
  red_gain_ = red;
  green_gain_ = green;
  blue_gain_ = blue;
}

template <typename T>
tensorflow::Status Debayer::HalfDebayerInternal(const cv::Mat& input,
                                                bool is_rgb, cv::Mat* output) {
  // Adjust the output to the right size if it's not already.
  output->create(input.rows / 2, input.cols / 2, CV_8UC3);

  if (num_debayer_threads_ <= 1) {
    PartialHalfDebayer<T>(input, 0, output->rows, is_rgb, output);
    return tensorflow::Status();
  }

  int height_per_thread = output->rows / num_debayer_threads_;
  absl::BlockingCounter blocking_counter(num_debayer_threads_);
  std::vector<std::thread> workers;
  workers.reserve(num_debayer_threads_);
  for (int thread_index = 0; thread_index < num_debayer_threads_;
       thread_index++) {
    workers.emplace_back([this, &blocking_counter, thread_index,
                          height_per_thread, &input, output, is_rgb] {
      PartialHalfDebayer<T>(input, height_per_thread * thread_index,
                            height_per_thread, is_rgb, output);
      blocking_counter.DecrementCount();
    });
  }
  blocking_counter.Wait();
  for (std::thread& w : workers) {
    w.join();
  }
  return tensorflow::Status();
}

template <typename T>
void Debayer::PartialHalfDebayer(const cv::Mat& input, int offset, int height,
                                 bool is_rgb, cv::Mat* output) {
  for (int y = offset; y < offset + height; y++) {
    int input_y = y << 1;
    const T* input_row1 = reinterpret_cast<const T*>(input.row(input_y).ptr());
    const T* input_row2 =
        reinterpret_cast<const T*>(input.row(input_y + 1).ptr());
    uint8_t* output_ptr = output->row(y).ptr();
    for (int x = 0; x < output->cols; x++) {
      // Bayer pixel pattern:
      //   RG
      //   GB
      int input_x1 = x << 1;
      int input_x2 = input_x1 + 1;
      if (is_rgb) {
        // Output is RGB.
        *(output_ptr++) =
            GetPixelValue(input_row1[input_x1], red_gain_);  // Red
        *(output_ptr++) =
            GetPixelValue(input_row1[input_x2], green_gain_);  // Green
        *(output_ptr++) =
            GetPixelValue(input_row2[input_x2], blue_gain_);  // Blue
      } else {
        // Output is BGR.
        *(output_ptr++) =
            GetPixelValue(input_row2[input_x2], blue_gain_);  // Blue
        *(output_ptr++) =
            GetPixelValue(input_row1[input_x2], green_gain_);  // Green
        *(output_ptr++) =
            GetPixelValue(input_row1[input_x1], red_gain_);  // Red
      }
    }
  }
}

template <typename T>
uint8_t Debayer::GetPixelValue(T bayer_value, double gain) {
  // Bit shift value, so that we take the most significant byte value as uint8.
  constexpr int bit_shift = (sizeof(T) - 1) * 8;

  if (has_gain_) {
    T adjusted_value = static_cast<T>(
        std::min<double>(gain * bayer_value, std::numeric_limits<T>::max()));
    return static_cast<uint8_t>(adjusted_value >> bit_shift);
  } else {
    return static_cast<uint8_t>(bayer_value >> bit_shift);
  }
}

template <>
uint8_t Debayer::GetPixelValue(uint8_t bayer_value, double gain) {
  // Bit shift value, so that we take the most significant byte value as uint8.
  if (has_gain_) {
    uint8_t adjusted_value = static_cast<uint8_t>(std::min<double>(
        gain * bayer_value, std::numeric_limits<uint8_t>::max()));
    return adjusted_value;
  } else {
    return bayer_value;
  }
}

}  // namespace image_processor
