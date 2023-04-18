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
#ifndef PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MAIN_WINDOW_H_
#define PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MAIN_WINDOW_H_

#include <QBoxLayout>
#include <QCheckBox>
#include <QCloseEvent>
#include <QDockWidget>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalMapper>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>
#include <atomic>
#include <functional>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "arm_app/microdisplay.h"
#include "arm_app/previewer.h"
#include "image_processor/inferer.h"
#include "main_looper/looper.h"
#include "serial_interface/objective_serial.h"

namespace arm_app {

// Window to provide controls to ARM functionality.
class MainWindow : public QWidget {
  Q_OBJECT

 public:
  MainWindow(Microdisplay* microdisplay);

 signals:
  void SignalDisplayWarning();

 protected:
  void closeEvent(QCloseEvent* event) override;

 private slots:
  void DisplayHeatmapCheckBoxToggled();
  void DisplayInferenceCheckBoxToggled();
  void SnapshotButtonClicked();
  void ObjectiveSelected(QWidget* button);
  void ModelTypeSelected(QWidget* button);
  void BrightnessChanged(int new_brightness);
  void DisplayWarning();
  // Model classes handlers.
  void GleasonModelClassToggled(QCheckBox* checkbox);
  void CervicalModelClassToggled(QCheckBox* checkbox);
  // Testing mode handlers.
  void EVRangeChanged(int new_value, bool is_min);
  void EVStepsPerUnitSelected(int new_value, QWidget* button);
  // Calibration mode handlers.
  void DisplayCalibrationTargetCheckBoxToggled();
  void CalibrationMarginLeftChanged(int diff);
  void CalibrationMarginTopChanged(int diff);

 private:
  // Initializes signal mappers used to connect model type and objective
  // buttons. This needs to be called before connecting the buttons.
  void InitSignalMappers();

  // Utility functions for window layout.
  void SetUpMainLayout();
  void SetUpControlPanel();
  void SetUpModelTypeControls();
  void SetUpObjectiveControls();
  void SetUpBrightnessControls();
  void SetUpTestingControls();
  void SetUpCalibrationControls();

  // Utility functions for controls.
  QPushButton* CreateAndConnectButton(
      const char* text, image_processor::ObjectiveLensPower power);
  QPushButton* CreateAndConnectButton(const char* text,
                                      image_processor::ModelType model_type);
  QCheckBox* CreateDisplayHeatmapCheckBox();
  QCheckBox* CreateDisplayInferenceCheckBox();
  QPushButton* CreateSnapshotButton();
  QSlider* CreateBrightnessSlider(int initial_brightness);
  QSpinBox* CreateBrightnessSpinBox(int initial_brightness);
  // Synchronizes the brightness slider and spin box. Must be called after both
  // are created.
  void ConnectBrightnessControls();

  // Automatic objective switching.
  void InitializeAutomaticObjectiveSwitching();
  bool AutomaticObjectiveSwitchingIsOn();
  // Updates which objectives should be marked as disabled when automatic
  // objective switching is on.
  void UpdateDisabledObjectives();

  // Updates which model class checkboxes are shown.
  void UpdateDisplayedModelClasses();
  QCheckBox* CreateAndConnectModelClassCheckbox(
      const char* text, image_processor::GleasonClasses gleason_class);
  QCheckBox* CreateAndConnectModelClassCheckbox(
      const char* text, image_processor::CervicalClasses cervical_class);

  // Testing mode controls.
  QSpinBox* CreateEVMinSpinBox(int initial_ev_min);
  QSpinBox* CreateEVMaxSpinBox(int initial_ev_max);
  QPushButton* CreateAndConnectEVButton(int ev_steps_per_unit,
                                        const char* text);

  // Calibration mode controls.
  QCheckBox* CreateDisplayCalibrationTargetCheckBox();
  QSpinBox* CreateCalibrationMarginLeftSpinBox();
  QSpinBox* CreateCalibrationMarginTopSpinBox();

  void SetToggleButtonStyle(QPushButton* button);

  // Signal mappers for mapping button clicks to slot calls.
  std::unique_ptr<QSignalMapper> objective_signal_mapper_;
  std::unique_ptr<QSignalMapper> model_type_signal_mapper_;

  // Selects objective for a model type. This is used when we select default
  // objectives for a model type when the model type is selected while an
  // unsupported objective is checked.
  void SelectObjectiveForModel(image_processor::ObjectiveLensPower objective);
  QPushButton* GetButtonForObjective(
      image_processor::ObjectiveLensPower objective);

  // Owned UI elements
  std::unique_ptr<QHBoxLayout> main_layout_;

  std::unique_ptr<QHBoxLayout> preview_layout_;
  std::unique_ptr<Previewer> previewer_;

  std::unique_ptr<QWidget> controls_container_;
  std::unique_ptr<QVBoxLayout> controls_layout_;

  std::unique_ptr<QGroupBox> model_type_box_;
  std::unique_ptr<QVBoxLayout> model_type_layout_;
  std::unique_ptr<QPushButton> button_lyna_;
  std::unique_ptr<QPushButton> button_gleason_;
  std::unique_ptr<QPushButton> button_mitotic_;
  std::unique_ptr<QPushButton> button_cervical_;

  std::unique_ptr<QGroupBox> objective_box_;
  std::unique_ptr<QVBoxLayout> objective_layout_;
  std::unique_ptr<QLabel> rou_label_;
  std::unique_ptr<QPushButton> button_2x_;
  std::unique_ptr<QPushButton> button_4x_;
  std::unique_ptr<QPushButton> button_10x_;
  std::unique_ptr<QPushButton> button_20x_;
  std::unique_ptr<QPushButton> button_40x_;
  // Unknown objective button, only displayed in Automatic Objective Switching
  std::unique_ptr<QPushButton> button_unknown_objective_;

  std::unique_ptr<QGroupBox> brightness_controls_box_;
  std::unique_ptr<QHBoxLayout> brightness_controls_layout_;
  std::unique_ptr<QSlider> brightness_slider_;
  std::unique_ptr<QSpinBox> brightness_spin_box_;

  // Testing mode UI elements
  std::unique_ptr<QGroupBox> test_controls_box_;
  std::unique_ptr<QVBoxLayout> test_controls_layout_;
  std::unique_ptr<QGroupBox> ev_range_box_;
  std::unique_ptr<QHBoxLayout> ev_range_layout_;
  std::unique_ptr<QSpinBox> ev_max_spin_box_;
  std::unique_ptr<QSpinBox> ev_min_spin_box_;
  std::unique_ptr<QGroupBox> ev_step_size_box_;
  std::unique_ptr<QHBoxLayout> ev_step_size_layout_;
  std::unique_ptr<QPushButton> button_ev_1_4_;
  std::unique_ptr<QPushButton> button_ev_1_3_;
  std::unique_ptr<QPushButton> button_ev_1_2_;
  std::unique_ptr<QPushButton> button_ev_1_1_;

  // Model classes UI elements
  std::unique_ptr<QCheckBox> checkbox_gleason_gp_3_;
  std::unique_ptr<QCheckBox> checkbox_gleason_gp_4_;
  std::unique_ptr<QCheckBox> checkbox_gleason_gp_5_;
  std::unique_ptr<QCheckBox> checkbox_cervical_cin_1_;
  std::unique_ptr<QCheckBox> checkbox_cervical_cin_2_plus_;

  std::unique_ptr<QCheckBox> checkbox_display_heatmap_;
  std::unique_ptr<QCheckBox> checkbox_display_inference_;

  // Calibration mode UI elements
  std::unique_ptr<QGroupBox> calibration_controls_box_;
  std::unique_ptr<QVBoxLayout> calibration_controls_layout_;
  std::unique_ptr<QCheckBox> checkbox_display_calibration_target_;
  std::unique_ptr<QLabel> calibration_margin_left_label_;
  std::unique_ptr<QSpinBox> calibration_margin_left_spin_box_;
  std::unique_ptr<QLabel> calibration_margin_top_label_;
  std::unique_ptr<QSpinBox> calibration_margin_top_spin_box_;

  std::unique_ptr<QPushButton> button_snapshot_;

  // UI state
  QPushButton* active_objective_button_ = nullptr;
  QPushButton* active_model_type_button_ = nullptr;
  QPushButton* active_ev_steps_per_unit_button_ = nullptr;
  std::atomic_bool display_heatmap_{false};
  std::atomic_bool display_inference_{true};
  std::atomic_bool display_calibration_target_;

  // Warning message to be displayed. This can be updated by other threads and
  // is persisted here to be displayed by the GUI thread.
  absl::Mutex warning_lock_;
  std::string warning_message_;

  // Automatic objective switching
  std::shared_ptr<serial::ObjectiveSerial> objective_serial_;

  // Model state.
  absl::Mutex model_lock_;
  image_processor::ObjectiveLensPower current_objective_;
  image_processor::ModelType current_model_type_;
  // Initial model classes. Should match those used by the inferer.
  absl::Mutex model_classes_lock_;
  absl::flat_hash_set<image_processor::GleasonClasses>
      positive_gleason_classes_ = {image_processor::GleasonClasses::GP_3,
                                   image_processor::GleasonClasses::GP_4,
                                   image_processor::GleasonClasses::GP_5};
  absl::flat_hash_set<image_processor::CervicalClasses>
      positive_cervical_classes_ = {
          image_processor::CervicalClasses::CIN_2_PLUS};

  // Looper for running image captor, inference, and heatmap displaying.
  std::unique_ptr<main_looper::Looper> looper_;

  Microdisplay* microdisplay_;
};

}  // namespace arm_app

#endif  // PATHOLOGY_OFFLINE_AR_MICROSCOPE_ARM_APP_MAIN_WINDOW_H_
