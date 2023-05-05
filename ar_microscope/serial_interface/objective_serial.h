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
#ifndef AR_MICROSCOPE_SERIAL_INTERFACE_OBJECTIVE_SERIAL_H_
#define AR_MICROSCOPE_SERIAL_INTERFACE_OBJECTIVE_SERIAL_H_

#include <atomic>
#include <thread>  // NOLINT
#include <vector>

#include "absl/synchronization/mutex.h"
#include "image_processor/inferer.h"
#include "include/serial/serial.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace serial {

// Callback used on objective changes.
using ObjectiveChangeCallback =
    std::function<void(image_processor::ObjectiveLensPower)>;

// Singleton class that monitors the objective serial port for objective
// position changes.
class ObjectiveSerial {
 public:
  // Creates or retrieves the objective serial connection singleton.
  static std::shared_ptr<ObjectiveSerial> GetObjectiveSerial();

  // Adds a callback for objective changes.
  void AddCallback(ObjectiveChangeCallback callback);

  bool Initialized() { return initialized_; }

  // Initiates a thread for listening for objective changes. If an error is
  // thrown during the listening, the listening stops.
  tensorflow::Status StartListening();
  void StopListening();
  bool IsListening() { return listening_; }

  // Returns the current objective or an error if it is not listening.
  tensorflow::StatusOr<image_processor::ObjectiveLensPower> GetObjective();

  ObjectiveSerial(const ObjectiveSerial&) = delete;
  ObjectiveSerial& operator=(const ObjectiveSerial&) = delete;

 private:
  // Attempts to initialize a serial connection.
  void Initialize();

  // Finds the serial port that corresponds to the objective control box. This
  // assumes that at most one objective control box is connected.
  std::unique_ptr<serial::Serial> FindObjectiveSerialPortOrNull();

  // Reads the position from the serial. Returns -1 if no position was
  // retrieved.
  int ReadPositionFromSerial();

  ObjectiveSerial() = default;

  absl::Mutex serial_mutex_;
  std::unique_ptr<serial::Serial> serial_;

  absl::Mutex objective_mutex_;
  image_processor::ObjectiveLensPower current_objective_;

  absl::Mutex callbacks_mutex_;
  std::vector<ObjectiveChangeCallback> callbacks_;

  // Tracks whether the serial connection was initialized successfully.
  std::atomic_bool initialized_ = {false};

  // Tracks whether the serial port is listening or not.
  std::atomic_bool listening_ = {false};

  // Thread for listening for serial port messages.
  std::unique_ptr<std::thread> thread_;
  // Tracks whether to shut down the listening thread.
  std::atomic_bool to_exit_ = {false};
};

}  // namespace serial

#endif  // AR_MICROSCOPE_SERIAL_INTERFACE_OBJECTIVE_SERIAL_H_
