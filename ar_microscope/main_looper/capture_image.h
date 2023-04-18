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
#ifndef THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_CAPTURE_IMAGE_H_
#define THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_CAPTURE_IMAGE_H_

#include <QBoxLayout>
#include <QPaintEvent>
#include <QPushButton>
#include <QSizePolicy>
#include <QString>
#include <QWidget>
#include <memory>

#include "opencv2/core.hpp"
#include "absl/synchronization/mutex.h"

namespace main_looper {

// Widget to display image.
class ImageViewer : public QWidget {
 public:
  ImageViewer() {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  QSize sizeHint() const override {
    return QSize(500, 500);
  }

  void SetImage(cv::Mat* image);
  void SaveImage(const QString& filename);

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  std::unique_ptr<cv::Mat> image_;
  absl::Mutex mutex_;
};

// Root window of image capture application.
class ImageViewerMainWindow : public QWidget {
  Q_OBJECT

 public:
  ImageViewerMainWindow();

  void SetImage(cv::Mat* image);

 private slots:
  void SaveButtonClicked();

 private:
  std::unique_ptr<QBoxLayout> container_;
  std::unique_ptr<QPushButton> save_button_;
  std::unique_ptr<ImageViewer> viewer_;
};

}  // namespace main_looper

#endif  // THIRD_PARTY_PATHOLOGY_OFFLINE_AR_MICROSCOPE_MAIN_LOOPER_CAPTURE_IMAGE_H_
