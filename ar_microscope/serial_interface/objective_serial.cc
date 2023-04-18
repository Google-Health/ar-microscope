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
#include "serial_interface/objective_serial.h"

#include <memory>
#include <thread>  // NOLINT

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "arm_app/arm_config.h"
#include "image_processor/inferer.h"
#include "include/serial/serial.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

ABSL_FLAG(
    std::string, objective_serial_port, "",
    "The port name, e.g. '/dev/ttyUSB0', of the objective serial port. When "
    "not provided, the objective serial port is automatically detected.");

namespace {

constexpr int kTimeoutMilliseconds = 500;
constexpr int kBaudRate = 19200;
constexpr serial::bytesize_t kByteSize = serial::eightbits;
constexpr serial::parity_t kParity = serial::parity_even;
constexpr serial::stopbits_t kStopBits = serial::stopbits_two;
constexpr char kEndOfLineMarker[] = "\r\n";
constexpr int kMaxReadLines = 65536;

// Maximum consecutive failed reads from the serial port before disconnecting
// from it.
constexpr int kMaxFailedReads = 5;

std::string RemoveEolMarker(const std::string& message) {
  return absl::StrReplaceAll(message, {{kEndOfLineMarker, ""}});
}

bool IsUsbPort(const std::string& port_name) {
  const auto location = port_name.find("USB");
  return location != std::string::npos;
}

// Parses the objective position from a objective serial message. There are
// several valid message formats:
//   1. "1OBM {i}": a move or initial position of i.
//   2. "1OB {i}": the existing position is i.
//   3. "1OBM X": the objective is between positions.
// For cases 1 and 2, the position i is returned. For case 3, 0 is returned. If
// an error occurs, -1 is returned.
int ParseObjectivePositionMessage(const std::string& message) {
  std::vector<std::string> submessages =
      absl::StrSplit(RemoveEolMarker(message), ' ');
  int result;
  if (submessages.size() != 2) {
    result = -1;
  } else if (submessages[0] != "1OBM" && submessages[0] != "1OB") {
    result = -1;
  } else {
    try {
      if (submessages[1] == "X" || submessages[1] == "x") {
        result = 0;
      } else {
        result = std::stoi(submessages[1]);
      }
    } catch (std::exception e) {
      result = -1;
    }
  }
  if (result == -1) {
    LOG_EVERY_N_SEC(WARNING, 10)
        << "Error parsing objective position message: " << message;
  }
  return result;
}

std::unique_ptr<serial::Serial> MaybeCreateSerialConnection(
    const std::string& port) {
  try {
    return std::make_unique<serial::Serial>(
        port, kBaudRate, serial::Timeout::simpleTimeout(kTimeoutMilliseconds),
        kByteSize, kParity, kStopBits);
  } catch (...) {
    LOG(WARNING) << "Serial port could not be opened: " << port;
    return nullptr;
  }
}

// Attempts to read the objective position from the given serial port. Returns:
//   1. -1 if there is an error.
//   2. 0 if the objective is in a transition state.
//   3. [1-7] if the objective is at that position.
int ReadPositionFromSerialPort(serial::Serial* serial_port) {
  try {
    serial_port->write(absl::StrCat("1OB?", kEndOfLineMarker));
    const auto messages =
        serial_port->readlines(kMaxReadLines, kEndOfLineMarker);
    if (messages.empty()) {
      LOG(WARNING) << "No messages from serial port when checking objective.";
      return -1;
    }
    const auto last_message = messages[messages.size() - 1];
    return ParseObjectivePositionMessage(last_message);
  } catch (...) {
    LOG(WARNING) << "Exception when reading position from serial port.";
    return -1;
  }
}

}  // namespace

namespace serial {

std::shared_ptr<ObjectiveSerial> ObjectiveSerial::GetObjectiveSerial() {
  static ObjectiveSerial* const kObjectiveSerial = new ObjectiveSerial();

  if (!kObjectiveSerial->Initialized()) {
    kObjectiveSerial->Initialize();
  }

  return std::shared_ptr<ObjectiveSerial>(kObjectiveSerial);
}

std::unique_ptr<serial::Serial> ObjectiveSerial::FindObjectiveSerialPort() {
  std::string objective_port = absl::GetFlag(FLAGS_objective_serial_port);
  if (!objective_port.empty()) {
    return MaybeCreateSerialConnection(objective_port);
  }
  const auto ports = serial::list_ports();
  for (const auto& port : ports) {
    if (!IsUsbPort(port.port)) continue;
    auto maybe_objective_serial = MaybeCreateSerialConnection(port.port);
    if (maybe_objective_serial != nullptr) {
      int position = ReadPositionFromSerialPort(maybe_objective_serial.get());
      if (position >= 0) {
        return maybe_objective_serial;
      }
    }
  }
  return nullptr;
}

void ObjectiveSerial::Initialize() {
  serial_ = FindObjectiveSerialPort();
  if (serial_ != nullptr) {
    LOG(INFO) << "Objective serial will be using port: " << serial_->getPort();
    initialized_.store(true);
  } else {
    LOG(WARNING) << "Valid objective serial port not found.";
    initialized_.store(false);
  }
}

tensorflow::Status ObjectiveSerial::StartListening() {
  absl::MutexLock unused_lock_(&serial_mutex_);
  // Log in for remote monitoring.
  serial_->write(absl::StrCat("1LOG IN", kEndOfLineMarker));
  const auto login_messages =
      serial_->readlines(kMaxReadLines, kEndOfLineMarker);
  if (login_messages.empty() ||
      RemoveEolMarker(login_messages[0]) != "1LOG +") {
    LOG(WARNING) << "Could not log into objective serial";
    return tensorflow::errors::Unavailable(
        "Could not log into objective serial");
  }
  // Turns on active notification from the objective control box.
  serial_->write(absl::StrCat("1SNDOB ON", kEndOfLineMarker));
  const auto sndob_messages =
      serial_->readlines(kMaxReadLines, kEndOfLineMarker);
  if (sndob_messages.size() < 2 ||
      RemoveEolMarker(sndob_messages[0]) != "1SNDOB +") {
    LOG(WARNING) << "Turning on active notification failed";
    return tensorflow::errors::Unavailable(
        "Turning on active notification failed");
  }
  const int starting_position =
      ParseObjectivePositionMessage(sndob_messages[sndob_messages.size() - 1]);
  current_objective_ =
      arm_app::GetArmConfig().GetObjectiveForPosition(starting_position);
  listening_.store(true);
  LOG(INFO) << "Started listening for objective changes. Initial objective: "
            << image_processor::ObjectiveToString(current_objective_);

  thread_ = std::make_unique<std::thread>([this]() {
    int failed_reads = 0;
    while (!to_exit_.load()) {
      absl::MutexLock unused_lock_(&serial_mutex_);
      if (failed_reads > kMaxFailedReads) {
        LOG(ERROR) << "Too many failed objective reads, shutting down "
                      "objective serial.";
        listening_.store(false);
        to_exit_.store(true);
        break;
      }
      try {
        const bool messages_detected = serial_->waitReadable();
        if (messages_detected) {
          const auto messages =
              serial_->readlines(kMaxReadLines, kEndOfLineMarker);
          if (messages.empty()) {
            LOG(ERROR) << "Mesages detected, but no messages could be read.";
            ++failed_reads;
            continue;
          }
          const auto last_message = messages[messages.size() - 1];
          const int position = ParseObjectivePositionMessage(last_message);
          if (position == 0) {
            // Objective is in transition.
            continue;
          } else if (position == -1) {
            ++failed_reads;
            continue;
          } else {
            current_objective_ =
                arm_app::GetArmConfig().GetObjectiveForPosition(position);
            LOG(INFO) << "Detected objective change to: "
                      << image_processor::ObjectiveToString(current_objective_);
            failed_reads = 0;
            for (const auto& callback : callbacks_) {
              callback(current_objective_);
            }
          }
        }
      } catch (const serial::SerialException& e) {
        LOG(ERROR) << "SerialException while listening for objective changes.";
        ++failed_reads;
      }
    }
  });

  return tensorflow::Status();
}

void ObjectiveSerial::StopListening() {
  to_exit_.store(true);
  if (thread_ != nullptr) {
    thread_->join();
    thread_.reset();
  }
  if (serial_ != nullptr) {
    serial_->close();
  }
  listening_.store(false);
}

tensorflow::StatusOr<image_processor::ObjectiveLensPower>
ObjectiveSerial::GetObjective() {
  if (listening_) {
    return current_objective_;
  } else {
    return tensorflow::errors::NotFound(
        "Serial port listening failed, possibly due to a disconnection.");
  }
}

void ObjectiveSerial::AddCallback(ObjectiveChangeCallback callback) {
  absl::MutexLock unused_lock_(&callbacks_mutex_);
  callbacks_.emplace_back(callback);
}

}  // namespace serial
