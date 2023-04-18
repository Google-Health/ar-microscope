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
#include "main_looper/capture_image.h"

#include <QApplication>
#include <QFileDialog>
#include <QImage>
#include <QMessageBox>
#include <QPainter>
#include <QStandardPaths>
#include <atomic>
#include <chrono>  // NOLINT
#include <cstdio>
#include <memory>
#include <thread>  // NOLINT

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "image_captor/image_captor_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace main_looper {

void ImageViewer::SetImage(cv::Mat* image) {
  absl::MutexLock unused_lock(&mutex_);
  image_.reset(image);
  update();
}

void ImageViewer::SaveImage(const QString& filename) {
  cv::Mat bgr_image;
  {
    absl::MutexLock unused_lock(&mutex_);
    // image_ is in RGB. Convert to BGR so that we can save from OpenCV.
    cv::cvtColor(*image_, bgr_image, cv::COLOR_RGB2BGR);
  }
  cv::imwrite(filename.toUtf8().data(), bgr_image);
}

void ImageViewer::paintEvent(QPaintEvent* event) {
  absl::MutexLock unused_lock(&mutex_);
  if (image_) {
    QPainter painter(this);
    QImage qimage(image_->ptr(), image_->cols, image_->rows,
                  QImage::Format_RGB888);
    // Resize the preview image to fit the window, keeping aspect ratio.
    QImage resized_image = qimage.scaled(size(), Qt::KeepAspectRatio);
    painter.drawImage(resized_image.rect(), resized_image);
  }
}

ImageViewerMainWindow::ImageViewerMainWindow() {
  container_ = std::make_unique<QBoxLayout>(QBoxLayout::TopToBottom, this);
  save_button_ = std::make_unique<QPushButton>("Save Preview Image", this);
  connect(save_button_.get(), SIGNAL(clicked()), this,
          SLOT(SaveButtonClicked()));
  container_->addWidget(save_button_.get());
  viewer_ = std::make_unique<ImageViewer>();
  container_->addWidget(viewer_.get());
}

void ImageViewerMainWindow::SetImage(cv::Mat* image) {
  viewer_->SetImage(image);
}

void ImageViewerMainWindow::SaveButtonClicked() {
  QString filename = QFileDialog::getSaveFileName(
      this, "Filename to Save Iamge",
      QStandardPaths::writableLocation(QStandardPaths::DownloadLocation));
  viewer_->SaveImage(filename);
}

class ImageCaptureLooper {
 public:
  explicit ImageCaptureLooper(ImageViewerMainWindow* viewer)
      : image_viewer_(viewer) {}

  tensorflow::Status Start() {
    captor_ = absl::WrapUnique(image_captor::ImageCaptorFactory::Create());
    TF_RETURN_IF_ERROR(captor_->Initialize());

    thread_ = std::make_unique<std::thread>([this]() {
      while (!exit_.load()) {
        LogIfError(CaptureSingleImage());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      LogIfError(captor_->Finalize());
    });

    return tensorflow::Status();
  }

  void Stop() {
    exit_.store(true);
    thread_->join();
  }

 private:
  tensorflow::Status CaptureSingleImage() {
    auto image = std::make_unique<cv::Mat>();
    auto bayer_image = std::make_unique<cv::Mat>();
    VLOG(1) << "Capturing Image";
    TF_RETURN_IF_ERROR(captor_->GetImage(true, image.get()));
    image_viewer_->SetImage(image.release());
    VLOG(1) << "Image captured";
    return tensorflow::Status();
  }

  void LogIfError(const tensorflow::Status& status) {
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }

  ImageViewerMainWindow* image_viewer_;
  std::unique_ptr<image_captor::ImageCaptor> captor_;
  std::atomic_bool exit_{false};
  std::unique_ptr<std::thread> thread_;
};

int CaptureImages(int argc, char* argv[]) {
  QApplication application(argc, argv);
  ImageViewerMainWindow window;
  window.show();

  ImageCaptureLooper captor_looper(&window);
  tensorflow::Status status = captor_looper.Start();
  if (!status.ok()) {
    LOG(ERROR) << status;
    return EXIT_FAILURE;
  }
  int application_return_value = application.exec();
  captor_looper.Stop();
  return application_return_value;
}

}  // namespace main_looper

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  return main_looper::CaptureImages(argc, argv);
}
