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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "opencv2/core.hpp"
#include "tensorflow/core/lib/core/status.h"

extern absl::Flag<int> FLAGS_num_debayer_threads;

namespace {

using image_processor::Debayer;

using ::testing::Eq;
using ::testing::Ne;
using ::testing::Test;

// 6x4 8-bit Bayer pattern data.
uint8_t kBayer8bit[]{
    0xdb, 0x33, 0x72, 0x9d, 0xd6, 0x2a, 0xe4, 0xb1, 0xfb, 0x49, 0xcc, 0xa8,
    0x78, 0x75, 0xf8, 0x96, 0xdd, 0x7d, 0xcb, 0x8a, 0x5f, 0xd7, 0x62, 0xa2,
};

// 3x2 RGB as result of above Bayer.
uint8_t kRgb[]{
    0xdb, 0x33, 0xb1, 0x72, 0x9d, 0x49, 0xd6, 0x2a, 0xa8,
    0x78, 0x75, 0x8a, 0xf8, 0x96, 0xd7, 0xdd, 0x7d, 0xa2,
};

// 3x2 BGR as result of above Bayer.
uint8_t kBgr[]{
    0xb1, 0x33, 0xdb, 0x49, 0x9d, 0x72, 0xa8, 0x2a, 0xd6,
    0x8a, 0x75, 0x78, 0xd7, 0x96, 0xf8, 0xa2, 0x7d, 0xdd,
};

// 4x2 16-bit Bayer pattern data.
uint16_t kBayer16Bit[]{
    0x5c4e, 0xfbb0, 0x9c88, 0x5f5c, 0xa6e6, 0x41a1, 0xd44c, 0xefe2,
};

// 2x1 RGB as result of above Bayer.
uint8_t kRgbRaw[]{
    0x5c, 0xfb, 0x41, 0x9c, 0x5f, 0xef,
};

// 2x1 BGR as result of above Bayer.
uint8_t kBgrRaw[]{
    0x41, 0xfb, 0x5c, 0xef, 0x5f, 0x9c,
};

constexpr double kRedGain = 1.6;
constexpr double kGreenGain = 2.1;
constexpr double kBlueGain = 1.0;

uint8_t kRgbGainAdjusted[]{
    0x93, 0xff, 0x41, 0xfa, 0xc8, 0xef,
};

uint8_t kBgrGainAdjusted[]{
    0x41, 0xff, 0x93, 0xef, 0xc8, 0xfa,
};

void AssertEquals(const cv::Mat& a, const cv::Mat& b) {
  ASSERT_THAT(a.elemSize(), Eq(3));
  ASSERT_THAT(b.elemSize(), Eq(3));
  ASSERT_EQ(a.rows, b.rows);
  ASSERT_EQ(a.cols, b.cols);
  ASSERT_EQ(a.elemSize(), b.elemSize());
  for (int r = 0; r < a.rows; r++) {
    for (int c = 0; c < a.cols; c++) {
      ASSERT_EQ(a.ptr(r, c)[0], b.ptr(r, c)[0]);
      ASSERT_EQ(a.ptr(r, c)[1], b.ptr(r, c)[1]);
      ASSERT_EQ(a.ptr(r, c)[2], b.ptr(r, c)[2]);
    }
  }
}

TEST(DebayerTest, SingleThread) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  cv::Mat bayer(4, 6, CV_8UC1, kBayer8bit);
  cv::Mat rgb;
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(2, 3, CV_8UC3, kRgb);
  AssertEquals(expected, rgb);
}

TEST(DebayerTest, MultiThread) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 2);
  Debayer debayer;
  cv::Mat bayer(4, 6, CV_8UC1, kBayer8bit);
  cv::Mat rgb;
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(2, 3, CV_8UC3, kRgb);
  AssertEquals(expected, rgb);
}

TEST(DebayerTest, NotAllocated) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  cv::Mat bayer(4, 6, CV_8UC1, kBayer8bit);
  // Pre-allocate matrix memory with the same size as output.
  cv::Mat rgb(2, 3, CV_8UC3);
  uint8_t* original_pointer = rgb.ptr();
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(2, 3, CV_8UC3, kRgb);
  AssertEquals(expected, rgb);
  // Make sure the pointer is the same, i.e. memory is not newly allocated.
  ASSERT_THAT(rgb.ptr(), Eq(original_pointer));
}

TEST(DebayerTest, SizeDiscrepancyMemoryAllocation) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  cv::Mat bayer(4, 6, CV_8UC1, kBayer8bit);
  // Pre-allocate matrix memory with the different size from output.
  cv::Mat rgb(3, 4, CV_8UC3);
  uint8_t* original_pointer = rgb.ptr();
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(2, 3, CV_8UC3, kRgb);
  AssertEquals(expected, rgb);
  // Make sure the pointer is the different from original,
  // i.e. memory is newly allocated.
  ASSERT_THAT(rgb.ptr(), Ne(original_pointer));
}

TEST(DebayerTest, Bgr) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 2);
  Debayer debayer;
  cv::Mat bayer(4, 6, CV_8UC1, kBayer8bit);
  cv::Mat bgr;
  tensorflow::Status status = debayer.HalfDebayer(bayer, false, &bgr);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(2, 3, CV_8UC3, kBgr);
  AssertEquals(expected, bgr);
}

TEST(DebayerTest, Rgb16bit) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  cv::Mat bayer(2, 4, CV_16UC1, kBayer16Bit);
  cv::Mat rgb;
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(1, 2, CV_8UC3, kRgbRaw);
  AssertEquals(expected, rgb);
}

TEST(DebayerTest, Bgr16bit) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  cv::Mat bayer(2, 4, CV_16UC1, kBayer16Bit);
  cv::Mat bgr;
  tensorflow::Status status = debayer.HalfDebayer(bayer, false, &bgr);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(1, 2, CV_8UC3, kBgrRaw);
  AssertEquals(expected, bgr);
}

TEST(DebayerTest, Rgb16bitWhiteBalance) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  debayer.SetRgbGains(kRedGain, kGreenGain, kBlueGain);
  cv::Mat bayer(2, 4, CV_16UC1, kBayer16Bit);
  cv::Mat rgb;
  tensorflow::Status status = debayer.HalfDebayer(bayer, true, &rgb);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(1, 2, CV_8UC3, kRgbGainAdjusted);
  AssertEquals(expected, rgb);
}

TEST(DebayerTest, Bgr16bitWhiteBalance) {
  absl::SetFlag(&FLAGS_num_debayer_threads, 1);
  Debayer debayer;
  debayer.SetRgbGains(kRedGain, kGreenGain, kBlueGain);
  cv::Mat bayer(2, 4, CV_16UC1, kBayer16Bit);
  cv::Mat bgr;
  tensorflow::Status status = debayer.HalfDebayer(bayer, false, &bgr);
  ASSERT_TRUE(status.ok());
  cv::Mat expected(1, 2, CV_8UC3, kBgrGainAdjusted);
  AssertEquals(expected, bgr);
}

}  // namespace
