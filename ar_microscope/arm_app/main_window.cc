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
#include "arm_app/main_window.h"

#include <QApplication>
#include <QBoxLayout>
#include <QCheckBox>
#include <QDesktopWidget>
#include <QFont>
#include <QGroupBox>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QSizePolicy>
#include <QSpinBox>
#include <QVariant>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "arm_app/microdisplay.h"
#include "image_processor/inferer.h"
#include "image_processor/tensorflow_inferer.h"
#include "main_looper/arm_event.pb.h"
#include "main_looper/logger.h"
#include "serial_interface/objective_serial.h"

extern absl::Flag<int> FLAGS_initial_brightness;
extern absl::Flag<bool> FLAGS_test_mode;
extern absl::Flag<bool> FLAGS_calibration_mode;
extern absl::Flag<bool> FLAGS_aos;

namespace arm_app {
namespace {

using image_processor::ModelType;
using image_processor::ObjectiveLensPower;
using main_looper::ArmEvent;
using main_looper::GetLogger;

const ModelType kInitialModelType = ModelType::LYNA;
const ObjectiveLensPower kInitialObjective = ObjectiveLensPower::OBJECTIVE_10x;

constexpr char kObjectivePowerProperty[] = "ObjectivePower";
constexpr char kModelTypeProperty[] = "ModelType";
constexpr char kModelClassProperty[] = "ModelClass";

// User facing strings.
constexpr char kArmAppTitle[] = "AR Microscope [Research Use Only]";
constexpr char kModelTypeLabel[] = "Select model type";
constexpr char kLynaModelLabel[] = "Lymph";
constexpr char kGleasonModelLabel[] = "Prostate";
constexpr char kMitoticModelLabel[] = "Mitotic";
constexpr char kCervicalModelLabel[] = "Cervical";
constexpr char kObjectiveLabel[] = "Select objective";
constexpr char kBrightnessLabel[] = "Adjust brightness";
constexpr char kSnapshotLabel[] = "Take snapshot";
constexpr char kExitLabel[] = "Finish ARM";
constexpr char kExitMessage[] = "Done using ARM for now?";
constexpr char kSnapshotCommentTitle[] = "Snapshot feedback";
constexpr char kSnapshotCommentMessage[] =
    "Please write a short comment describing your snapshot.";
constexpr char kDisplayHeatmapLabel[] = "Show heatmap";
constexpr char kDisplayInferenceLabel[] = "Show inference";
constexpr char kSnapshotConfirmLabel[] = "Take snapshot";
constexpr char kAutoExposureBrightnessError[] = "Error setting brightness.";
constexpr char kResearchUseOnlyWarning[] = "Research Use Only.";
// Model classes labels.
constexpr char kGleasonGp3[] = "Gleason Pattern 3";
constexpr char kGleasonGp4[] = "Gleason Pattern 4";
constexpr char kGleasonGp5[] = "Gleason Pattern 5";
constexpr char kCervicalCin1[] = "CIN 1";
constexpr char kCervicalCin2Plus[] = "CIN 2+";
// Testing mode labels.
constexpr char kTestControlsLabel[] = "Test snapshot controls";
constexpr char kEVRangeLabel[] = "EV range [Min, Max]";
constexpr char kEVStepSizeLabel[] = "EV step size";
// Calibration mode labels.
constexpr char kCalibrationControlsLabel[] = "Calibration controls";
constexpr char kDisplayCalibrationTargetLabel[] = "Show calibration target";
constexpr char kCalibrationMarginLeftLabel[] = "Adjust left margin";
constexpr char kCalibrationMarginTopLabel[] = "Adjust top margin";

// Styles
constexpr char kButtonStyle[] =
    "QPushButton{font-size: 24px;font-family: Arial;}";
constexpr char kControlGroupLabelStyle[] =
    "QGroupBox{font-size: 18px;font-family: Arial;}";
constexpr char kCheckBoxStyle[] =
    "QCheckBox{font-size: 18px;font-family: Arial;}";

// Main window size, relative to primary screen size.
constexpr float kWidthRatio = 1.0;
constexpr float kHeightRatio = 1.0;

// Controls panel settings.
constexpr float kControlsSpacing = 50;
constexpr float kControlsWidth = 250;

// EV test controls.
constexpr int kEVMax = 3;
constexpr int kEVMin = -3;
constexpr int kInitialEVStepsPerUnit = 1;

// Calibration mode controls.
constexpr int kMarginMaxDiff = 250;

ArmEvent ObjectiveToEvent(ObjectiveLensPower objective) {
  switch (objective) {
    case ObjectiveLensPower::OBJECTIVE_2x:
      return ArmEvent::CLICK_2X;
    case ObjectiveLensPower::OBJECTIVE_4x:
      return ArmEvent::CLICK_4X;
    case ObjectiveLensPower::OBJECTIVE_10x:
      return ArmEvent::CLICK_10X;
    case ObjectiveLensPower::OBJECTIVE_20x:
      return ArmEvent::CLICK_20X;
    case ObjectiveLensPower::OBJECTIVE_40x:
      return ArmEvent::CLICK_40X;
    default:
      return ArmEvent::UNSPECIFIED_EVENT;
  }
}


ArmEvent ModelTypeToEvent(ModelType model_type) {
  switch (model_type) {
    case ModelType::LYNA:
      return ArmEvent::CLICK_LYMPH;
    case ModelType::GLEASON:
      return ArmEvent::CLICK_PROSTATE;
    case ModelType::MITOTIC:
      return ArmEvent::CLICK_MITOTIC;
    case ModelType::CERVICAL:
      return ArmEvent::CLICK_CERVICAL;
    default:
      return ArmEvent::UNSPECIFIED_EVENT;
  }
}
}  // namespace

MainWindow::MainWindow(Microdisplay* microdisplay)
    : display_calibration_target_(absl::GetFlag(FLAGS_calibration_mode)),
      current_objective_(kInitialObjective),
      current_model_type_(kInitialModelType),
      microdisplay_(microdisplay) {
  setWindowTitle(kArmAppTitle);
  QDesktopWidget dw;
  QRect screen_size = dw.availableGeometry(dw.primaryScreen());
  this->setGeometry(0, 0, screen_size.width() * kWidthRatio,
                    screen_size.height() * kHeightRatio);

  InitSignalMappers();
  SetUpMainLayout();
  SetUpControlPanel();
  SetUpModelTypeControls();
  SetUpObjectiveControls();
  SetUpBrightnessControls();
  if (absl::GetFlag(FLAGS_test_mode)) {
    SetUpTestingControls();
  }
  if (absl::GetFlag(FLAGS_calibration_mode)) {
    SetUpCalibrationControls();
  }

  // Check model buttons that correspond to `kInitialModelType` and
  // `current_objective_`.
  if (absl::GetFlag(FLAGS_aos)) {
    InitializeAutomaticObjectiveSwitching();
  }
  auto* objective_button = GetButtonForObjective(current_objective_);
  objective_button->setChecked(true);
  active_objective_button_ = objective_button;
  UpdateDisabledObjectives();
  button_lyna_->setChecked(true);
  active_model_type_button_ = button_lyna_.get();

  auto display_warning_callback = [this](const std::string& message) {
    absl::MutexLock unused_lock(&warning_lock_);
    warning_message_ = message;
    this->SignalDisplayWarning();
  };
  connect(this, &MainWindow::SignalDisplayWarning, this,
          &MainWindow::DisplayWarning);
  looper_ = std::make_unique<main_looper::Looper>(
      current_objective_, current_model_type_, previewer_.get(), microdisplay_,
      display_warning_callback);
  if (!looper_->ImageCaptorSupportsAutoExposure()) {
    brightness_controls_box_->hide();
  }
  looper_->Run();
}

void MainWindow::closeEvent(QCloseEvent* event) {
  // Show confirmation window.
  QMessageBox::StandardButton response = QMessageBox::question(
      this, kExitLabel, kExitMessage, QMessageBox::Yes | QMessageBox::No);
  if (response != QMessageBox::Yes) {
    event->ignore();
    return;
  }
  previewer_->Stop();
  looper_->Stop();
  if (AutomaticObjectiveSwitchingIsOn()) {
    objective_serial_->StopListening();
  }
  event->accept();
  QApplication::exit();
  GetLogger().LogEvent(main_looper::ArmEvent::ARM_STOP);
}

void MainWindow::DisplayHeatmapCheckBoxToggled() {
  display_heatmap_.store(checkbox_display_heatmap_->isChecked());
  previewer_->SetDisplayHeatmap(display_heatmap_);
  if (display_heatmap_) {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_HEATMAP_ON);
  } else {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_HEATMAP_OFF);
  }
}

void MainWindow::DisplayInferenceCheckBoxToggled() {
  display_inference_.store(checkbox_display_inference_->isChecked());
  microdisplay_->SetDisplayInference(display_inference_);
  previewer_->SetDisplayInference(display_inference_);
  if (display_inference_) {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_INFERENCE_ON);
  } else {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_INFERENCE_OFF);
  }
}

void MainWindow::SnapshotButtonClicked() {
  auto dialog = std::make_unique<QInputDialog>(this);
  dialog->setOptions(QInputDialog::UsePlainTextEditForTextInput);
  dialog->setWindowTitle(kSnapshotCommentTitle);
  dialog->setLabelText(kSnapshotCommentMessage);
  dialog->setOkButtonText(kSnapshotConfirmLabel);
  const int ret = dialog->exec();
  if (ret) {
    if (absl::GetFlag(FLAGS_test_mode)) {
      looper_->TakeTestSnapshots(current_objective_, current_model_type_,
                                 brightness_slider_->value(),
                                 dialog->textValue().toUtf8().constData());
    } else {
      previewer_->TakeSnapshot(current_objective_, current_model_type_,
                               brightness_slider_->value(),
                               dialog->textValue().toUtf8().constData());
    }
    GetLogger().LogEvent(ArmEvent::CLICK_SNAPSHOT);
  }
  button_snapshot_->setChecked(false);
}

void MainWindow::ObjectiveSelected(QWidget* button) {
  absl::MutexLock unused_lock(&model_lock_);
  if (active_objective_button_ == button) {
    return;
  }
  if (active_objective_button_) {
    // Turn off the current selection.
    active_objective_button_->setChecked(false);
  }
  active_objective_button_ = dynamic_cast<QPushButton*>(button);
  active_objective_button_->setChecked(true);
  const auto objective = static_cast<image_processor::ObjectiveLensPower>(
      button->property(kObjectivePowerProperty).toInt());
  current_objective_ = objective;
  UpdateDisabledObjectives();
  looper_->SetObjectiveAndModelType(objective, current_model_type_);
  // Only log when a different objective is selected.
  GetLogger().LogEvent(ObjectiveToEvent(objective));
}

QPushButton* MainWindow::GetButtonForObjective(ObjectiveLensPower objective) {
  QPushButton* objective_button;
  switch (objective) {
    case ObjectiveLensPower::OBJECTIVE_2x:
      objective_button = button_2x_.get();
      break;
    case ObjectiveLensPower::OBJECTIVE_4x:
      objective_button = button_4x_.get();
      break;
    case ObjectiveLensPower::OBJECTIVE_10x:
      objective_button = button_10x_.get();
      break;
    case ObjectiveLensPower::OBJECTIVE_20x:
      objective_button = button_20x_.get();
      break;
    case ObjectiveLensPower::OBJECTIVE_40x:
      objective_button = button_40x_.get();
      break;
    default:
      objective_button = button_unknown_objective_.get();
      break;
  }
  return objective_button;
}

void MainWindow::SelectObjectiveForModel(ObjectiveLensPower objective) {
  auto* objective_button = GetButtonForObjective(objective);
  if (active_objective_button_) active_objective_button_->setChecked(false);
  objective_button->setChecked(true);
  active_objective_button_ = objective_button;
  current_objective_ = objective;
}

void MainWindow::ModelTypeSelected(QWidget* button) {
  absl::MutexLock unused_lock(&model_lock_);
  if (active_model_type_button_ == button) {
    // Currently selected button is clicked.
    active_model_type_button_->setChecked(true);
    return;
  }
  if (active_model_type_button_) {
    // Turn off the current selection.
    active_model_type_button_->setChecked(false);
  }
  if (!AutomaticObjectiveSwitchingIsOn()) {
    // We do not support 40x for gleason, so we default to 10x if it is
    // selected.
    button_40x_->setDisabled(button == button_gleason_.get());
    if (button == button_gleason_.get() &&
        active_objective_button_ == button_40x_.get()) {
      SelectObjectiveForModel(ObjectiveLensPower::OBJECTIVE_10x);
    }

    // We do not support 2x, 4x, 10x, or 20x models for mitotic.
    button_2x_->setDisabled(button == button_mitotic_.get());
    button_4x_->setDisabled(button == button_mitotic_.get());
    button_10x_->setDisabled(button == button_mitotic_.get());
    button_20x_->setDisabled(button == button_mitotic_.get());

    if (button == button_mitotic_.get()) {
      SelectObjectiveForModel(ObjectiveLensPower::OBJECTIVE_40x);
    }
  }
  active_model_type_button_ = dynamic_cast<QPushButton*>(button);
  const auto model_type = static_cast<image_processor::ModelType>(
      button->property(kModelTypeProperty).toInt());
  current_model_type_ = model_type;
  UpdateDisplayedModelClasses();
  looper_->SetObjectiveAndModelType(current_objective_, model_type);
  // Only log when a different model type is selected.
  GetLogger().LogEvent(ModelTypeToEvent(model_type));
}

void MainWindow::GleasonModelClassToggled(QCheckBox* checkbox) {
  absl::MutexLock unused_lock(&model_classes_lock_);
  const auto model_class = static_cast<image_processor::GleasonClasses>(
      checkbox->property(kModelClassProperty).toInt());
  if (checkbox->isChecked()) {
    positive_gleason_classes_.emplace(model_class);
  } else {
    positive_gleason_classes_.erase(model_class);
  }
  looper_->SetPositiveGleasonClasses(positive_gleason_classes_);
  GetLogger().LogEvent(ArmEvent::TOGGLE_GLEASON_CLASS);
}

void MainWindow::CervicalModelClassToggled(QCheckBox* checkbox) {
  absl::MutexLock unused_lock(&model_classes_lock_);
  const auto model_class = static_cast<image_processor::CervicalClasses>(
      checkbox->property(kModelClassProperty).toInt());
  if (checkbox->isChecked()) {
    positive_cervical_classes_.emplace(model_class);
  } else {
    positive_cervical_classes_.erase(model_class);
  }
  looper_->SetPositiveCervicalClasses(positive_cervical_classes_);
  GetLogger().LogEvent(ArmEvent::TOGGLE_CERVICAL_CLASS);
}

void MainWindow::BrightnessChanged(int new_brightness) {
  auto status = looper_->SetAutoExposureBrightness(new_brightness);
  if (!status.ok()) {
    QMessageBox error_msg_box;
    error_msg_box.setText(kAutoExposureBrightnessError);
    error_msg_box.exec();
    LOG(WARNING) << "Set auto exposure brightness error.";
  }
}

void MainWindow::DisplayWarning() {
  absl::MutexLock unused_lock(&warning_lock_);
  auto warning_popup = std::make_unique<QMessageBox>(this);
  warning_popup->setText(warning_message_.c_str());
  warning_popup->setIcon(QMessageBox::Warning);
  warning_popup->exec();
}

void MainWindow::InitializeAutomaticObjectiveSwitching() {
  objective_serial_ = serial::ObjectiveSerial::GetObjectiveSerial();
  if (objective_serial_->Initialized()) {
    const auto status = objective_serial_->StartListening();
    if (status.ok()) {
      const auto objective_or = objective_serial_->GetObjective();
      if (objective_or.ok()) {
        current_objective_ = *objective_or;
        objective_serial_->AddCallback([this](ObjectiveLensPower objective) {
          this->ObjectiveSelected(this->GetButtonForObjective(objective));
        });
      } else {
        LOG(WARNING) << "Error getting intial objective: "
                     << objective_or.status();
      }
    } else {
      LOG(WARNING) << "Objective failed to start listening: " << status;
    }
  }
}

bool MainWindow::AutomaticObjectiveSwitchingIsOn() {
  return objective_serial_ != nullptr && objective_serial_->IsListening();
}

void MainWindow::UpdateDisabledObjectives() {
  if (AutomaticObjectiveSwitchingIsOn()) {
    button_2x_->setDisabled(!button_2x_->isChecked());
    button_4x_->setDisabled(!button_4x_->isChecked());
    button_10x_->setDisabled(!button_10x_->isChecked());
    button_20x_->setDisabled(!button_20x_->isChecked());
    button_40x_->setDisabled(!button_40x_->isChecked());
    button_unknown_objective_->setDisabled(
        !button_unknown_objective_->isChecked());
  } else {
    button_unknown_objective_->setDisabled(true);
    button_unknown_objective_->setVisible(false);
  }
}

void MainWindow::EVRangeChanged(int new_value, bool is_min) {
  if (is_min) {
    looper_->SetEVMin(new_value);
  } else {
    looper_->SetEVMax(new_value);
  }
}

void MainWindow::EVStepsPerUnitSelected(int new_value, QWidget* button) {
  if (active_ev_steps_per_unit_button_ == button) {
    // Currently selected button is clicked.
    active_ev_steps_per_unit_button_->setChecked(true);
    return;
  }
  if (active_ev_steps_per_unit_button_) {
    // Turn off the current selection.
    active_ev_steps_per_unit_button_->setChecked(false);
  }
  active_ev_steps_per_unit_button_ = dynamic_cast<QPushButton*>(button);
  looper_->SetEVStepsPerUnit(new_value);
}

void MainWindow::DisplayCalibrationTargetCheckBoxToggled() {
  display_calibration_target_.store(
      checkbox_display_calibration_target_->isChecked());
  microdisplay_->SetDisplayCalibrationTarget(display_calibration_target_);
  previewer_->SetDisplayCalibrationTarget(display_calibration_target_);
  if (display_calibration_target_) {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_CALIBRATION_TARGET_ON);
  } else {
    GetLogger().LogEvent(ArmEvent::TOGGLE_DISPLAY_CALIBRATION_TARGET_OFF);
  }
}

void MainWindow::CalibrationMarginLeftChanged(int diff) {
  microdisplay_->AdjustMarginLeft(diff);
}

void MainWindow::CalibrationMarginTopChanged(int diff) {
  microdisplay_->AdjustMarginTop(diff);
}

void MainWindow::InitSignalMappers() {
  objective_signal_mapper_ = std::make_unique<QSignalMapper>();
  model_type_signal_mapper_ = std::make_unique<QSignalMapper>();
  connect(objective_signal_mapper_.get(), SIGNAL(mapped(QWidget*)), this,
          SLOT(ObjectiveSelected(QWidget*)));
  connect(model_type_signal_mapper_.get(), SIGNAL(mapped(QWidget*)), this,
          SLOT(ModelTypeSelected(QWidget*)));
}

void MainWindow::SetUpMainLayout() {
  main_layout_ = std::make_unique<QHBoxLayout>(this);
  this->setLayout(main_layout_.get());
  preview_layout_ = std::make_unique<QHBoxLayout>();
  controls_container_ = std::make_unique<QWidget>(this);
  controls_layout_ = std::make_unique<QVBoxLayout>();
  main_layout_->addWidget(controls_container_.get());
  main_layout_->addLayout(preview_layout_.get());
  controls_container_->setLayout(controls_layout_.get());
  controls_container_->setFixedWidth(kControlsWidth);
  previewer_ = std::make_unique<Previewer>();
  preview_layout_->addWidget(previewer_.get());
}

void MainWindow::SetUpControlPanel() {
  model_type_box_ = std::make_unique<QGroupBox>(kModelTypeLabel, this);
  objective_box_ = std::make_unique<QGroupBox>(kObjectiveLabel, this);
  rou_label_ = std::make_unique<QLabel>(this);

  rou_label_->setText(kResearchUseOnlyWarning);
  brightness_controls_box_ =
      std::make_unique<QGroupBox>(kBrightnessLabel, this);
  button_snapshot_ = absl::WrapUnique(CreateSnapshotButton());

  controls_layout_->addWidget(rou_label_.get());
  controls_layout_->addSpacing(kControlsSpacing);
  controls_layout_->addWidget(model_type_box_.get());
  controls_layout_->addSpacing(kControlsSpacing);
  controls_layout_->addWidget(objective_box_.get());
  controls_layout_->addSpacing(kControlsSpacing);
  controls_layout_->addWidget(brightness_controls_box_.get());
  controls_layout_->addSpacing(kControlsSpacing);
  if (absl::GetFlag(FLAGS_test_mode)) {
    test_controls_box_ = std::make_unique<QGroupBox>(kTestControlsLabel, this);
    controls_layout_->addWidget(test_controls_box_.get());
    controls_layout_->addSpacing(kControlsSpacing);
  }
  if (absl::GetFlag(FLAGS_calibration_mode)) {
    calibration_controls_box_ =
        std::make_unique<QGroupBox>(kCalibrationControlsLabel, this);
    controls_layout_->addWidget(calibration_controls_box_.get());
  } else {
    checkbox_display_heatmap_ =
        absl::WrapUnique(CreateDisplayHeatmapCheckBox());
    checkbox_display_inference_ =
        absl::WrapUnique(CreateDisplayInferenceCheckBox());
    controls_layout_->addWidget(checkbox_display_inference_.get());
    controls_layout_->addWidget(checkbox_display_heatmap_.get());
  }
  controls_layout_->addSpacing(kControlsSpacing);
  controls_layout_->addWidget(button_snapshot_.get());
  controls_layout_->addStretch(1);
}

void MainWindow::SetUpModelTypeControls() {
  // Model types
  button_lyna_ = absl::WrapUnique(CreateAndConnectButton(
      kLynaModelLabel, image_processor::ModelType::LYNA));
  button_gleason_ = absl::WrapUnique(CreateAndConnectButton(
      kGleasonModelLabel, image_processor::ModelType::GLEASON));
  button_mitotic_ = absl::WrapUnique(CreateAndConnectButton(
      kMitoticModelLabel, image_processor::ModelType::MITOTIC));
  button_cervical_ = absl::WrapUnique(CreateAndConnectButton(
      kCervicalModelLabel, image_processor::ModelType::CERVICAL));

  // Model classes
  checkbox_gleason_gp_3_ = absl::WrapUnique(CreateAndConnectModelClassCheckbox(
      kGleasonGp3, image_processor::GleasonClasses::GP_3));
  checkbox_gleason_gp_4_ = absl::WrapUnique(CreateAndConnectModelClassCheckbox(
      kGleasonGp4, image_processor::GleasonClasses::GP_4));
  checkbox_gleason_gp_5_ = absl::WrapUnique(CreateAndConnectModelClassCheckbox(
      kGleasonGp5, image_processor::GleasonClasses::GP_5));
  checkbox_cervical_cin_1_ =
      absl::WrapUnique(CreateAndConnectModelClassCheckbox(
          kCervicalCin1, image_processor::CervicalClasses::CIN_1));
  checkbox_cervical_cin_2_plus_ =
      absl::WrapUnique(CreateAndConnectModelClassCheckbox(
          kCervicalCin2Plus, image_processor::CervicalClasses::CIN_2_PLUS));

  model_type_layout_ = std::make_unique<QVBoxLayout>();
  model_type_layout_->addWidget(button_lyna_.get());
  model_type_layout_->addWidget(button_gleason_.get());
  model_type_layout_->addWidget(checkbox_gleason_gp_3_.get());
  model_type_layout_->addWidget(checkbox_gleason_gp_4_.get());
  model_type_layout_->addWidget(checkbox_gleason_gp_5_.get());
  model_type_layout_->addWidget(button_mitotic_.get());
  model_type_layout_->addWidget(button_cervical_.get());
  model_type_layout_->addWidget(checkbox_cervical_cin_1_.get());
  model_type_layout_->addWidget(checkbox_cervical_cin_2_plus_.get());

  model_type_box_->setLayout(model_type_layout_.get());
  model_type_box_->setStyleSheet(kControlGroupLabelStyle);
  UpdateDisplayedModelClasses();
}

void MainWindow::SetUpObjectiveControls() {
  button_2x_ = absl::WrapUnique(CreateAndConnectButton(
      "2x", image_processor::ObjectiveLensPower::OBJECTIVE_2x));
  button_4x_ = absl::WrapUnique(CreateAndConnectButton(
      "4x", image_processor::ObjectiveLensPower::OBJECTIVE_4x));
  button_10x_ = absl::WrapUnique(CreateAndConnectButton(
      "10x", image_processor::ObjectiveLensPower::OBJECTIVE_10x));
  button_20x_ = absl::WrapUnique(CreateAndConnectButton(
      "20x", image_processor::ObjectiveLensPower::OBJECTIVE_20x));
  button_40x_ = absl::WrapUnique(CreateAndConnectButton(
      "40x", image_processor::ObjectiveLensPower::OBJECTIVE_40x));
  button_unknown_objective_ = absl::WrapUnique(CreateAndConnectButton(
      "Unknown",
      image_processor::ObjectiveLensPower::UNSPECIFIED_OBJECTIVE_LENS_POWER));

  objective_layout_ = std::make_unique<QVBoxLayout>();
  objective_layout_->addWidget(button_2x_.get());
  objective_layout_->addWidget(button_4x_.get());
  objective_layout_->addWidget(button_10x_.get());
  objective_layout_->addWidget(button_20x_.get());
  objective_layout_->addWidget(button_40x_.get());
  objective_layout_->addWidget(button_unknown_objective_.get());

  objective_box_->setLayout(objective_layout_.get());
  objective_box_->setStyleSheet(kControlGroupLabelStyle);
}


void MainWindow::SetUpBrightnessControls() {
  brightness_slider_ = absl::WrapUnique(
      CreateBrightnessSlider(absl::GetFlag(FLAGS_initial_brightness)));
  brightness_spin_box_ = absl::WrapUnique(
      CreateBrightnessSpinBox(absl::GetFlag(FLAGS_initial_brightness)));

  brightness_controls_layout_ = std::make_unique<QHBoxLayout>();
  brightness_controls_layout_->addWidget(brightness_slider_.get());
  brightness_controls_layout_->addWidget(brightness_spin_box_.get());

  brightness_controls_box_->setLayout(brightness_controls_layout_.get());
  brightness_controls_box_->setStyleSheet(kControlGroupLabelStyle);
  ConnectBrightnessControls();
}

void MainWindow::UpdateDisplayedModelClasses() {
  const bool is_gleason =
      (current_model_type_ == image_processor::ModelType::GLEASON);
  const bool is_cervical =
      (current_model_type_ == image_processor::ModelType::CERVICAL);
  checkbox_gleason_gp_3_->setVisible(is_gleason);
  checkbox_gleason_gp_4_->setVisible(is_gleason);
  checkbox_gleason_gp_5_->setVisible(is_gleason);
  checkbox_cervical_cin_1_->setVisible(is_cervical);
  checkbox_cervical_cin_2_plus_->setVisible(is_cervical);
  // This show() is needed to trigger a window update event, which is needed
  // when showing widgets that start out as being hidden. Note that Qt's
  // documentation is incorrect in equating setVisible(true) with show(); see
  // the first comment under https://stackoverflow.com/a/12178039.
  model_type_box_->show();
}

void MainWindow::SetUpTestingControls() {
  ev_range_box_ = std::make_unique<QGroupBox>(kEVRangeLabel, this);
  ev_range_layout_ = std::make_unique<QHBoxLayout>();
  ev_min_spin_box_ = absl::WrapUnique(CreateEVMinSpinBox(kEVMin));
  ev_max_spin_box_ = absl::WrapUnique(CreateEVMaxSpinBox(kEVMax));
  ev_range_layout_->addWidget(ev_min_spin_box_.get());
  ev_range_layout_->addWidget(ev_max_spin_box_.get());
  ev_range_box_->setLayout(ev_range_layout_.get());

  ev_step_size_box_ = std::make_unique<QGroupBox>(kEVStepSizeLabel, this);
  ev_step_size_layout_ = std::make_unique<QHBoxLayout>();
  ev_step_size_box_->setLayout(ev_step_size_layout_.get());
  button_ev_1_4_ = absl::WrapUnique(CreateAndConnectEVButton(4, "1/4"));
  button_ev_1_3_ = absl::WrapUnique(CreateAndConnectEVButton(3, "1/3"));
  button_ev_1_2_ = absl::WrapUnique(CreateAndConnectEVButton(2, "1/2"));
  button_ev_1_1_ = absl::WrapUnique(CreateAndConnectEVButton(1, "1"));
  ev_step_size_layout_->addWidget(button_ev_1_4_.get());
  ev_step_size_layout_->addWidget(button_ev_1_3_.get());
  ev_step_size_layout_->addWidget(button_ev_1_2_.get());
  ev_step_size_layout_->addWidget(button_ev_1_1_.get());

  test_controls_layout_ = std::make_unique<QVBoxLayout>();
  test_controls_layout_->addWidget(ev_range_box_.get());
  test_controls_layout_->addWidget(ev_step_size_box_.get());
  test_controls_box_->setLayout(test_controls_layout_.get());
  test_controls_box_->setStyleSheet(kControlGroupLabelStyle);
}

void MainWindow::SetUpCalibrationControls() {
  checkbox_display_calibration_target_ =
      absl::WrapUnique(CreateDisplayCalibrationTargetCheckBox());
  calibration_margin_left_spin_box_ =
      absl::WrapUnique(CreateCalibrationMarginLeftSpinBox());
  calibration_margin_top_spin_box_ =
      absl::WrapUnique(CreateCalibrationMarginTopSpinBox());
  calibration_margin_left_label_ =
      std::make_unique<QLabel>(kCalibrationMarginLeftLabel);
  calibration_margin_top_label_ =
      std::make_unique<QLabel>(kCalibrationMarginTopLabel);

  calibration_controls_layout_ = std::make_unique<QVBoxLayout>();
  calibration_controls_layout_->addWidget(
      checkbox_display_calibration_target_.get());
  calibration_controls_layout_->addWidget(calibration_margin_left_label_.get());
  calibration_controls_layout_->addWidget(
      calibration_margin_left_spin_box_.get());
  calibration_controls_layout_->addWidget(calibration_margin_top_label_.get());
  calibration_controls_layout_->addWidget(
      calibration_margin_top_spin_box_.get());

  calibration_controls_box_->setLayout(calibration_controls_layout_.get());
  calibration_controls_box_->setStyleSheet(kControlGroupLabelStyle);
}

QPushButton* MainWindow::CreateAndConnectButton(
    const char* text, image_processor::ObjectiveLensPower power) {
  QPushButton* button = new QPushButton(text, this);
  SetToggleButtonStyle(button);
  button->setProperty(kObjectivePowerProperty,
                      QVariant(static_cast<int>(power)));
  // Forward click signal to signal mapper.
  connect(button, SIGNAL(clicked()), objective_signal_mapper_.get(),
          SLOT(map()));
  // Assign signal mapping with the button itself as a parameter.
  objective_signal_mapper_->setMapping(button, button);
  return button;
}

QPushButton* MainWindow::CreateAndConnectButton(
    const char* text, image_processor::ModelType model_type) {
  QPushButton* button = new QPushButton(text, this);
  SetToggleButtonStyle(button);
  button->setProperty(kModelTypeProperty,
                      QVariant(static_cast<int>(model_type)));
  // Forward click signal to signal mapper.
  connect(button, SIGNAL(clicked()), model_type_signal_mapper_.get(),
          SLOT(map()));
  // Assign signal mapping with the button itself as a parameter.
  model_type_signal_mapper_->setMapping(button, button);
  return button;
}

QCheckBox* MainWindow::CreateAndConnectModelClassCheckbox(
    const char* text, image_processor::GleasonClasses gleason_class) {
  QCheckBox* checkbox = new QCheckBox(text, this);
  checkbox->setChecked(positive_gleason_classes_.find(gleason_class) !=
                       positive_gleason_classes_.end());
  checkbox->setStyleSheet(kCheckBoxStyle);
  checkbox->setProperty(kModelClassProperty,
                        QVariant(static_cast<int>(gleason_class)));
  connect(checkbox, &QCheckBox::clicked, this,
          [this, checkbox]() { this->GleasonModelClassToggled(checkbox); });
  return checkbox;
}

QCheckBox* MainWindow::CreateAndConnectModelClassCheckbox(
    const char* text, image_processor::CervicalClasses cervical_class) {
  QCheckBox* checkbox = new QCheckBox(text, this);
  checkbox->setChecked(positive_cervical_classes_.find(cervical_class) !=
                       positive_cervical_classes_.end());
  checkbox->setStyleSheet(kCheckBoxStyle);
  checkbox->setProperty(kModelClassProperty,
                        QVariant(static_cast<int>(cervical_class)));
  connect(checkbox, &QCheckBox::clicked, this,
          [this, checkbox]() { this->CervicalModelClassToggled(checkbox); });
  return checkbox;
}

QCheckBox* MainWindow::CreateDisplayHeatmapCheckBox() {
  QCheckBox* checkbox = new QCheckBox(kDisplayHeatmapLabel, this);
  checkbox->setChecked(display_heatmap_);
  checkbox->setStyleSheet(kCheckBoxStyle);
  connect(checkbox, &QCheckBox::clicked, this,
          &MainWindow::DisplayHeatmapCheckBoxToggled);
  return checkbox;
}

QCheckBox* MainWindow::CreateDisplayInferenceCheckBox() {
  QCheckBox* checkbox = new QCheckBox(kDisplayInferenceLabel, this);
  checkbox->setChecked(display_inference_);
  checkbox->setStyleSheet(kCheckBoxStyle);
  connect(checkbox, &QCheckBox::clicked, this,
          &MainWindow::DisplayInferenceCheckBoxToggled);
  return checkbox;
}

QPushButton* MainWindow::CreateSnapshotButton() {
  QPushButton* button = new QPushButton(kSnapshotLabel, this);
  SetToggleButtonStyle(button);
  connect(button, SIGNAL(clicked()), this, SLOT(SnapshotButtonClicked()));
  return button;
}

QSlider* MainWindow::CreateBrightnessSlider(int initial_brightness) {
  QSlider* slider = new QSlider(Qt::Orientation::Horizontal, this);
  slider->setTickInterval(20);
  slider->setTickPosition(QSlider::TicksBelow);
  slider->setValue(initial_brightness);
  slider->setMinimum(0);
  slider->setMaximum(100);
  connect(slider, SIGNAL(valueChanged(int)), this,
          SLOT(BrightnessChanged(int)));
  connect(slider, &QSlider::sliderReleased, this,
          []() { GetLogger().LogEvent(ArmEvent::SET_BRIGHTNESS); });
  return slider;
}

QSpinBox* MainWindow::CreateBrightnessSpinBox(int initial_brightness) {
  QSpinBox* spin_box = new QSpinBox(this);
  spin_box->setRange(0, 100);
  spin_box->setValue(initial_brightness);
  spin_box->setSingleStep(1);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &MainWindow::BrightnessChanged);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          []() { GetLogger().LogEvent(ArmEvent::SET_BRIGHTNESS); });
  return spin_box;
}

void MainWindow::ConnectBrightnessControls() {
  connect(brightness_slider_.get(), &QSlider::valueChanged,
          brightness_spin_box_.get(), [this](int value) {
            brightness_spin_box_->blockSignals(true);
            brightness_spin_box_->setValue(value);
            brightness_spin_box_->blockSignals(false);
          });
  connect(brightness_spin_box_.get(),
          QOverload<int>::of(&QSpinBox::valueChanged), brightness_slider_.get(),
          [this](int value) {
            brightness_slider_->blockSignals(true);
            brightness_slider_->setValue(value);
            brightness_slider_->blockSignals(false);
          });
}

QSpinBox* MainWindow::CreateEVMinSpinBox(int initial_ev_min) {
  QSpinBox* spin_box = new QSpinBox(this);
  spin_box->setRange(kEVMin, 0);
  spin_box->setValue(initial_ev_min);
  spin_box->setSingleStep(1);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int value) { this->EVRangeChanged(value, /*is_min=*/true); });
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          []() { GetLogger().LogEvent(ArmEvent::SET_EV_MIN); });
  return spin_box;
}

QSpinBox* MainWindow::CreateEVMaxSpinBox(int initial_ev_max) {
  QSpinBox* spin_box = new QSpinBox(this);
  spin_box->setRange(0, kEVMax);
  spin_box->setValue(initial_ev_max);
  spin_box->setSingleStep(1);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int value) { this->EVRangeChanged(value, /*is_min=*/false); });
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          []() { GetLogger().LogEvent(ArmEvent::SET_EV_MAX); });
  return spin_box;
}

QPushButton* MainWindow::CreateAndConnectEVButton(int ev_steps_per_unit,
                                                  const char* text) {
  QPushButton* button = new QPushButton(text, this);
  SetToggleButtonStyle(button);
  if (ev_steps_per_unit == kInitialEVStepsPerUnit) {
    button->setChecked(true);
    active_ev_steps_per_unit_button_ = button;
  }
  connect(button, &QPushButton::clicked, this,
          [this, button, ev_steps_per_unit]() {
            this->EVStepsPerUnitSelected(ev_steps_per_unit, button);
            GetLogger().LogEvent(ArmEvent::SET_EV_STEP_SIZE);
          });
  return button;
}

QCheckBox* MainWindow::CreateDisplayCalibrationTargetCheckBox() {
  QCheckBox* checkbox = new QCheckBox(kDisplayCalibrationTargetLabel, this);
  checkbox->setChecked(display_calibration_target_);
  checkbox->setStyleSheet(kCheckBoxStyle);
  connect(checkbox, &QCheckBox::clicked, this,
          &MainWindow::DisplayCalibrationTargetCheckBoxToggled);
  return checkbox;
}

QSpinBox* MainWindow::CreateCalibrationMarginLeftSpinBox() {
  QSpinBox* spin_box = new QSpinBox(this);
  spin_box->setRange(-1 * kMarginMaxDiff, kMarginMaxDiff);
  spin_box->setValue(0);
  spin_box->setSingleStep(1);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int value) { this->CalibrationMarginLeftChanged(value); });
  return spin_box;
}

QSpinBox* MainWindow::CreateCalibrationMarginTopSpinBox() {
  QSpinBox* spin_box = new QSpinBox(this);
  spin_box->setRange(-1 * kMarginMaxDiff, kMarginMaxDiff);
  spin_box->setValue(0);
  spin_box->setSingleStep(1);
  connect(spin_box, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int value) { this->CalibrationMarginTopChanged(value); });
  return spin_box;
}

void MainWindow::SetToggleButtonStyle(QPushButton* button) {
  button->setCheckable(true);
  button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  button->setStyleSheet(kButtonStyle);
}

}  // namespace arm_app
