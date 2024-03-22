"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the blood vessel wall segmentation tab of the GUI
########################################################################################################################

import os
import time
from threading import Event

import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QFileDialog,
    QDialog,
    QProgressBar,
    QTabWidget
)

from trpseg import DATA_DIRECTORY
from trpseg.trpseg_util.utility import get_file_list_from_directory, count_segmentation_pixels
from trpseg.GUI.parameter_picker import ThresholdPicker
from trpseg.GUI.file_picker import FilePicker
from trpseg.trpseg_util.z_normalization import z_stack_tissue_mean_normalization
from trpseg.segmentation.blood_vessel_wall_seg import blood_vessel_wall_segmentation, get_default_min_signal_intensity,\
    get_default_high_confidence_threshold, get_default_local_th_offset, get_default_rball_th, get_default_border_rf_model_path
from trpseg.GUI.gui_utility import InputOutputWidgetSingleInput, NormalizationWidget, show_error_message


maximalNeededSizePolicy = QSizePolicy()
maximalNeededSizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
maximalNeededSizePolicy.setVerticalPolicy(QSizePolicy.Policy.Maximum)


class VesselWallPage(QWidget):

    def __init__(self, parent_main_window):
        super().__init__(parent_main_window)

        self.parent_main_window = parent_main_window

        self.resolution = np.zeros(shape=(3,), dtype=np.float32)
        self.input_folder_path = ""
        self.input_file_list_c00 = []
        self.input_file_list_c01 = []
        self.numFilesC00 = 0
        self.numFilesC01 = 0

        self.normalize_folder_out = ""
        self.output_folder_path = ""

        pageLayout = QVBoxLayout()
        self.setLayout(pageLayout)

        # 1. Choose Input Directory-------------------------------------------------------------------------------------
        self.in_out_widget = InputOutputWidgetSingleInput()
        self.in_out_widget.input_dir_button.clicked.connect(self.chooseInputDir)
        self.in_out_widget.output_dir_button.clicked.connect(self.chooseOutputDir)

        pageLayout.addWidget(self.in_out_widget, Qt.AlignmentFlag.AlignTop)



        # 2. Vessel Segmentation Parameter Tab--------------------------------------------------------------------------

        self.min_signal_intensity = get_default_min_signal_intensity()
        self.high_conf_threshold = get_default_high_confidence_threshold()
        self.local_th_offset = 0  # is set in self.initialize_vesseg_parameters after resolution is set
        self.rball_th = 0  # is set in self.initialize_vesseg_parameters after resolution is set
        self.border_model_path = ""  # is set in self.initialize_vesseg_parameters after resolution is set

        self.remove_border = True
        self.remove_small_objects = True
        self.store_intermediate_results = True

        # Tab Widget with the Tabs ('Tissue Mean Normalization' , 'Blood Vessel Segmentation Parameters')
        bloodVesselWallTabs = QTabWidget()
        pageLayout.addWidget(bloodVesselWallTabs)

        vesseg_parameter_widget = QWidget(bloodVesselWallTabs)

        bloodVesselWallTabs.addTab(vesseg_parameter_widget, 'Blood Vessel Segmentation Parameters')
        #bloodVesselWallTabs.setCurrentIndex(1)

        parameter_layout = QVBoxLayout()
        parameter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        #parameter_layout.setContentsMargins(10, 10, 20, 10)

        vesseg_parameter_widget.setLayout(parameter_layout)

        # Choose min_signal_intensity
        min_signal_layout = QHBoxLayout()
        min_signal_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        min_signal_label = QLabel("Choose minimal vessel signal intensity:")
        self.min_signal_spin = QSpinBox()
        self.min_signal_spin.setMinimum(0)
        self.min_signal_spin.setMaximum(65535)
        self.min_signal_spin.setSingleStep(10)
        self.min_signal_spin.setValue(self.min_signal_intensity)
        self.min_signal_spin.valueChanged.connect(self.set_min_signal)

        self.min_signal_spin.setToolTip("<p>All pixels with a value lower than this, are assumed to NOT be blood vessels.</p>")

        self.min_signal_button = QPushButton("Use ThPicker")
        self.min_signal_button.setEnabled(False)
        self.min_signal_button.clicked.connect(self.use_min_signal_threshold_picker)

        min_signal_layout.addWidget(min_signal_label)
        min_signal_layout.addWidget(self.min_signal_spin)
        min_signal_layout.addWidget(self.min_signal_button)
        parameter_layout.addLayout(min_signal_layout)

        # Choose high_confidence_threshold
        high_conf_layout = QHBoxLayout()
        high_conf_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        high_conf_label = QLabel("Choose high confidence threshold:")
        self.high_conf_spin = QSpinBox()
        self.high_conf_spin.setMinimum(0)
        self.high_conf_spin.setMaximum(65535)
        self.high_conf_spin.setSingleStep(10)
        self.high_conf_spin.setValue(self.high_conf_threshold)
        self.high_conf_spin.valueChanged.connect(self.set_high_conf_th)

        self.high_conf_spin.setToolTip("<p>All pixels with a value larger than this, are assumed to be blood vessels. "
                                       "They will be directly add to the segmentation result.</p>")

        self.high_conf_button = QPushButton("Use ThPicker")
        self.high_conf_button.setEnabled(False)
        self.high_conf_button.clicked.connect(self.use_high_conf_threshold_picker)

        high_conf_layout.addWidget(high_conf_label)
        high_conf_layout.addWidget(self.high_conf_spin)
        high_conf_layout.addWidget(self.high_conf_button)
        parameter_layout.addLayout(high_conf_layout)

        # Choose local_th_offset
        local_th_offset_layout = QHBoxLayout()
        local_th_offset_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        local_th_offset_label = QLabel("Choose offset for local thresholding:")
        self.local_th_offset_spin = QSpinBox()
        self.local_th_offset_spin.setMinimum(-10000)
        self.local_th_offset_spin.setMaximum(10000)
        self.local_th_offset_spin.setSingleStep(10)
        self.local_th_offset_spin.setValue(self.local_th_offset)
        self.local_th_offset_spin.valueChanged.connect(self.set_local_th_offset)

        self.local_th_offset_spin.setToolTip("<p>Choose the offset for the local thresholding.</p>"
                                             "<p>The higher this value, the more objects (objects with lower intensity) will be segmented. "
                                             "However, also the more noise and unwanted objects get segmented.</p>")

        local_th_offset_layout.addWidget(local_th_offset_label)
        local_th_offset_layout.addWidget(self.local_th_offset_spin)
        parameter_layout.addLayout(local_th_offset_layout)

        # Choose rball_th (rolling ball threshold)
        rball_th_layout = QHBoxLayout()
        rball_th_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        rball_th_label = QLabel("Choose threshold for rolling ball filtered image:")
        self.rball_th_spin = QSpinBox()
        self.rball_th_spin.setMinimum(0)
        self.rball_th_spin.setMaximum(10000)
        self.rball_th_spin.setSingleStep(10)
        self.rball_th_spin.setValue(self.rball_th)
        self.rball_th_spin.valueChanged.connect(self.set_rball_th)

        self.rball_th_spin.setToolTip("<p>This is the threshold to determine what belongs to blood vessels in a rolling ball filtered image.</p>"
                                      "<p>The higher the threshold, the less pixels are assumed to be blood vessels.</p>")

        rball_th_layout.addWidget(rball_th_label)
        rball_th_layout.addWidget(self.rball_th_spin)
        parameter_layout.addLayout(rball_th_layout)

        #Choose whether to remove_border
        remove_border_layout = QHBoxLayout()
        remove_border_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.remove_border_chbox = QCheckBox()
        self.remove_border_chbox.setChecked(self.remove_border)
        self.remove_border_chbox.stateChanged.connect(self.remove_border_change)

        self.remove_border_chbox.setToolTip("<p>Determine whether to use an already trained random forest model to"
                                            "predict what pixels are potentially blood vessels.</p>"
                                            "<p>This is very useful to remove unwanted objects from the segmentation"
                                            " result, especially at the brain boundary. However, it can also remove "
                                            "parts in the segmentation result that are blood vessels and that"
                                            " you do not want to remove.</p>")

        remove_border_layout.addWidget(self.remove_border_chbox)
        remove_border_layout.addWidget(QLabel("Remove border"))

        #Choose random forest model
        self.remove_border_lineEdit = QLineEdit()
        self.remove_border_lineEdit.setEnabled(False)
        self.remove_border_lineEdit.setText(self.border_model_path)

        self.choose_model_button = QPushButton("Choose Model...")
        self.choose_model_button.setSizePolicy(maximalNeededSizePolicy)
        self.choose_model_button.clicked.connect(self.set_new_model_path)

        self.choose_model_button.setToolTip("<p>Choose file with .joblib extension that is an already trained random "
                                            "forest model.</p>")

        remove_border_layout.addWidget(self.remove_border_lineEdit, Qt.AlignmentFlag.AlignLeft)
        remove_border_layout.addWidget(self.choose_model_button, Qt.AlignmentFlag.AlignLeft)


        parameter_layout.addLayout(remove_border_layout)

        #Choose whether to remove small objects
        remove_small_objects_layout = QHBoxLayout()
        remove_small_objects_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.remove_small_objects_chbox = QCheckBox()
        self.remove_small_objects_chbox.setChecked(self.remove_small_objects)

        self.remove_small_objects_chbox.stateChanged.connect(self.remove_small_objects_change)

        self.remove_small_objects_chbox.setToolTip("<p>If checked, then too small objects will be removed from the segmentation result.</p>")

        remove_small_objects_layout.addWidget(self.remove_small_objects_chbox)
        remove_small_objects_layout.addWidget(QLabel("Remove small objects"))

        parameter_layout.addLayout(remove_small_objects_layout)

        #Choose whether to store intermediate results

        store_intermed_res_layout = QHBoxLayout()
        store_intermed_res_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.store_intermed_res_chbox = QCheckBox()
        self.store_intermed_res_chbox.setChecked(self.store_intermediate_results)

        self.store_intermed_res_chbox.stateChanged.connect(self.store_intermed_res_change)

        self.store_intermed_res_chbox.setToolTip("<p>Choose whether to store intermediate results. If this is enabled"
                                                 " then the following is additionally stored:</p>"
                                                 "<p>The local thresholding results.</p>"
                                                 "<p>The thresholded rolling ball filtered results.</p>")

        store_intermed_res_layout.addWidget(self.store_intermed_res_chbox)
        store_intermed_res_layout.addWidget(QLabel("Store intermediate images"))

        parameter_layout.addLayout(store_intermed_res_layout)


        # 3. Mean Tissue Normalization----------------------------------------------------------------------------------

        normalization_widget_placeholder = QWidget(bloodVesselWallTabs)

        bloodVesselWallTabs.addTab(normalization_widget_placeholder, 'Tissue Mean Normalization')

        self.normalization_widget = NormalizationWidget(normalization_widget_placeholder)

        self.normalization_widget.normalization_min_button.clicked.connect(self.use_normalization_min_threshold_picker)
        self.normalization_widget.normalization_max_button.clicked.connect(self.use_normalization_max_threshold_picker)
        self.normalization_widget.normalization_button.clicked.connect(self.perform_normalization)

        # 4. Button to start vessel wall segmentation----------------------------------------------------------------
        ves_seg_start_layout = QHBoxLayout()
        self.ves_seg_start_button = QPushButton("Start Segmentation")
        self.ves_seg_start_button.setEnabled(False)
        self.ves_seg_start_button.clicked.connect(self.start_ves_seg)

        self.ves_seg_start_button.setToolTip("<p>Start the vessel wall segmentation pipeline.</p>")

        ves_seg_start_layout.addWidget(self.ves_seg_start_button, Qt.AlignmentFlag.AlignHCenter)

        pageLayout.addLayout(ves_seg_start_layout)

        # 5. Button to start vessel wall quantification by object pixel count----------------------------------------------------------------
        ves_quantify_layout = QHBoxLayout()
        self.ves_quantify_start_button = QPushButton("Start Quantification by Object Pixel Count")
        self.ves_quantify_start_button.setEnabled(False)
        self.ves_quantify_start_button.clicked.connect(self.start_ves_quantify)

        self.ves_quantify_start_button.setToolTip("<p>Estimate the vessel wall volume from a given binary segmentation."
                                             " Statistics will be written to a txt file.</p>"
                                             "<p>The volume is computed as the number of object pixels"
                                                  " multiplied by the pixel volume.</p>")

        ves_quantify_layout.addWidget(self.ves_quantify_start_button, Qt.AlignmentFlag.AlignHCenter)

        pageLayout.addLayout(ves_quantify_layout)

    def use_normalization_min_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c00)

        if th_dlg.exec():
            self.normalization_widget.tissue_min = th_dlg.chosen_threshold
            self.normalization_widget.normalization_min_spin.setValue(self.normalization_widget.tissue_min)

        th_dlg.deleteLater()

    def use_normalization_max_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c00)

        if th_dlg.exec():
            self.normalization_widget.tissue_max = th_dlg.chosen_threshold
            self.normalization_widget.normalization_max_spin.setValue(self.normalization_widget.tissue_max)

        th_dlg.deleteLater()

    def set_resolution(self, resolution):
        self.resolution = resolution

    def initialize_vesseg_parameters(self):
        self.rball_th = get_default_rball_th(self.resolution)
        self.local_th_offset = get_default_local_th_offset(self.resolution)
        self.border_model_path = get_default_border_rf_model_path(self.resolution)
        self.remove_border = True

        self.rball_th_spin.setValue(self.rball_th)
        self.local_th_offset_spin.setValue(self.local_th_offset)
        self.remove_border_chbox.setChecked(self.remove_border)
        self.remove_border_lineEdit.setText(self.border_model_path)

    def set_min_signal(self, value):
        self.min_signal_intensity = value

    def set_high_conf_th(self, value):
        self.high_conf_threshold = value

    def set_local_th_offset(self, value):
        self.local_th_offset = value

    def set_rball_th(self, value):
        self.rball_th = value

    def set_new_model_path(self):
        #home = os.path.expanduser("~")
        home = DATA_DIRECTORY
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", home, "Joblib Files (*.joblib)")
        if file_path != '':
            self.border_model_path = file_path
            self.remove_border_lineEdit.setText(file_path)

    def remove_border_change(self, state):
        if state == 0:
            self.remove_border = False
        else:
            self.remove_border = True

    def remove_small_objects_change(self, state):
        if state == 0:
            self.remove_small_objects = False
        else:
            self.remove_small_objects = True

    def use_min_signal_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c00)

        if th_dlg.exec():
            self.min_signal_intensity = th_dlg.chosen_threshold
            self.min_signal_spin.setValue(self.min_signal_intensity)

        th_dlg.deleteLater()

    def use_high_conf_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c00)

        if th_dlg.exec():
            self.high_conf_threshold = th_dlg.chosen_threshold
            self.high_conf_spin.setValue(self.high_conf_threshold)

        th_dlg.deleteLater()

    def store_intermed_res_change(self, state):
        if state == 0:
            self.store_intermediate_results = False
        else:
            self.store_intermediate_results = True

    def perform_normalization(self):
        #Checks whether needed parameters have been set
        if self.normalization_widget.tissue_max == 0 or self.normalization_widget.tissue_min == 0:
            show_error_message(self, "A tissue min > 0 and a tissue max > 0 need to be specified!")
            return

        if self.normalization_widget.tissue_max <= self.normalization_widget.tissue_min:
            show_error_message(self, "For the normalization the tissue max needs to be larger than tissue min.")
            return

        timestr = time.strftime("%H%M%S_")

        out_folder_name = timestr + "normalized_images"
        self.normalize_folder_out = self.output_folder_path + "\\" + out_folder_name
        os.makedirs(self.normalize_folder_out, exist_ok=True)

        # Open progress dialog + start thread with parameters
        norm_dlg = NormalizationProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if norm_dlg.exec():
            print("Normalization successful")
            self.parent_main_window.show()
            self.set_input_dir_to_normalized()
        else:
            print("Normalization unsuccessful")
            self.parent_main_window.show()

        norm_dlg.deleteLater()

    def set_input_dir_to_normalized(self):
        self.input_folder_path = self.normalize_folder_out
        self.in_out_widget.input_dir_line.setText(self.normalize_folder_out)
        self.input_file_list_c00 = get_file_list_from_directory(self.input_folder_path, isTif=None, channel_one=None)

        self.numFilesC00 = len(self.input_file_list_c00)
        self.in_out_widget.numFiles_label.setText(f"Number of Images: {self.numFilesC00}")

    def chooseInputDir(self):
        file_picker = FilePicker()

        if file_picker.exec():
            if file_picker.input_folder_path != '':
                self.input_folder_path = file_picker.input_folder_path
                self.in_out_widget.input_dir_line.setText(self.input_folder_path)
                self.input_file_list_c00 = file_picker.input_file_paths
                self.numFilesC00 = len(self.input_file_list_c00)
                self.in_out_widget.numFiles_label.setText(f"Number of Images: {self.numFilesC00}")
        file_picker.deleteLater()

        self.checkEnableElements()

    def chooseOutputDir(self):
        home = os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", home)
        if directory != '':
            self.output_folder_path = directory
            self.in_out_widget.output_dir_line.setText(directory)

            self.checkEnableElements()

    def checkEnableElements(self):
        if self.numFilesC00 > 0 and self.output_folder_path != "":
            self.normalization_widget.normalization_min_button.setEnabled(True)
            self.normalization_widget.normalization_max_button.setEnabled(True)
            self.normalization_widget.normalization_button.setEnabled(True)
            self.min_signal_button.setEnabled(True)
            self.high_conf_button.setEnabled(True)
            self.ves_seg_start_button.setEnabled(True)
            self.ves_quantify_start_button.setEnabled(True)

    def start_ves_seg(self):
        seg_dlg = VesSegProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if seg_dlg.exec():
            print("Segmentation successful")
            self.parent_main_window.show()
        else:
            print("Segmentation unsuccessful")
            self.parent_main_window.show()

        seg_dlg.deleteLater()

    def start_ves_quantify(self):
        quantify_dlg = VesQuantifyProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if quantify_dlg.exec():
            print("Quantification successful")
            self.parent_main_window.show()
        else:
            print("Quantification unsuccessful")
            self.parent_main_window.show()

        quantify_dlg.deleteLater()

class NormalizationProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for tissue mean normalization of channel 0 image slices and allows to cancel it."""
    def __init__(self, parent: VesselWallPage):
        super().__init__(parent)

        self.setWindowTitle("Tissue Mean Normalization")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Starting Normalization ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_seg_button = QPushButton("Cancel")
        self.cancel_seg_button.clicked.connect(self.cancel_normalization)

        cancelLayout.addWidget(self.cancel_seg_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        # A thread to perform tissue mean normalization
        self.normalization_thread = QThread()

        self.worker = NormalizationWorker(parent.input_file_list_c00, parent.normalize_folder_out, parent.normalization_widget.tissue_min, parent.normalization_widget.tissue_max)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.normalization_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.normalization_thread.finished.connect(self.worker.deleteLater)

        self.normalization_thread.started.connect(self.worker.startNormalization)
        self.normalization_thread.start()

    def close_on_success(self):
        self.normalization_thread.quit()
        self.normalization_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_normalization(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.normalization_thread.quit()
        self.normalization_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_normalization()
        event.ignore()


class NormalizationWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_files_list, output_folder_path, tissue_min, tissue_max):

        super().__init__()

        self.canceled = Event()

        self.input_files_list = input_files_list
        self.output_folder_path = output_folder_path

        self.tissue_min = tissue_min
        self.tissue_max = tissue_max


    def startNormalization(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        z_stack_tissue_mean_normalization(self.input_files_list, self.output_folder_path, self.tissue_min, self.tissue_max,
                                          canceled=self.canceled, progress_status=self.progress_status)

        if not self.canceled.is_set():
            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()


    def stop(self):
        self.canceled.set()


class VesSegProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for blood vessel wall segmentation and allows to cancel it."""
    def __init__(self, parent: VesselWallPage):
        super().__init__(parent)

        self.setWindowTitle("Vessel Wall Segmentation")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Starting Vessel Wall Segmentation ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_seg_button = QPushButton("Cancel")
        self.cancel_seg_button.clicked.connect(self.cancel_segmentation)

        cancelLayout.addWidget(self.cancel_seg_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        # Thread to perform vessel wall segmentation
        self.segmentation_thread = QThread()

        self.worker = VesSegWorker(parent.input_file_list_c00, parent.output_folder_path, parent.resolution, parent.min_signal_intensity,
                                   parent.high_conf_threshold, parent.local_th_offset, parent.rball_th, parent.store_intermediate_results, parent.remove_border,
                                   parent.border_model_path, parent.remove_small_objects)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.segmentation_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.segmentation_thread.finished.connect(self.worker.deleteLater)

        self.segmentation_thread.started.connect(self.worker.startSegmentation)
        self.segmentation_thread.start()

    def close_on_success(self):
        self.segmentation_thread.quit()
        self.segmentation_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_segmentation(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.segmentation_thread.quit()
        self.segmentation_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_segmentation()
        event.ignore()


class VesSegWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_file_paths, output_folder_path, resolution, min_signal_intensity, high_confidence_threshold,
                 local_th_offset, rball_th, store_intermed_results, remove_border, border_rf_model_path, remove_small_objects):

        super().__init__()

        self.canceled = Event()

        self.input_file_paths = input_file_paths
        self.output_folder_path = output_folder_path

        self.resolution = resolution
        self.min_signal_intensity = min_signal_intensity
        self.high_confidence_threshold = high_confidence_threshold
        self.local_th_offset = local_th_offset
        self.rball_th = rball_th
        self.store_intermed_results = store_intermed_results
        self.remove_border = remove_border
        self.border_rf_model_path = border_rf_model_path
        self.remove_small_objects = remove_small_objects

    def startSegmentation(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        blood_vessel_wall_segmentation(self.input_file_paths, self.output_folder_path, self.resolution,
                                       self.min_signal_intensity, self.high_confidence_threshold,
                                       self.local_th_offset, self.rball_th,
                                       store_intermed_results=self.store_intermed_results,
                                       remove_border=self.remove_border,
                                       border_rf_model_path=self.border_rf_model_path,
                                       remove_small_objects=self.remove_small_objects, canceled=self.canceled,
                                       progress_status=self.progress_status)

        if not self.canceled.is_set():
            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()


    def stop(self):
        self.canceled.set()


class VesQuantifyProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for Blood Vasculature Volume quantification and allows to cancel it."""

    def __init__(self, parent: VesselWallPage):
        super().__init__(parent)

        self.setWindowTitle("Quantify Volume by Object Pixel Count")

        # Main Layout
        page_layout = QVBoxLayout()
        # page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        # Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0, 100)
        self.progBarText = QLabel("Starting Quantification ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        # Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_seg_button = QPushButton("Cancel")
        self.cancel_seg_button.clicked.connect(self.cancel_segmentation)

        cancelLayout.addWidget(self.cancel_seg_button)
        page_layout.addLayout(cancelLayout)

        # Worker Thread
        # Thread to perform vessel wall quantification
        self.quantification_thread = QThread()

        self.worker = QuantifyVolumeWorker(parent.input_folder_path, parent.output_folder_path, parent.resolution)

        # print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.quantification_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.quantification_thread.finished.connect(self.worker.deleteLater)

        self.quantification_thread.started.connect(self.worker.startVolumeQuantification)
        self.quantification_thread.start()

    def close_on_success(self):
        self.quantification_thread.quit()
        self.quantification_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_segmentation(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.quantification_thread.quit()
        self.quantification_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_segmentation()
        event.ignore()


class QuantifyVolumeWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_folder_path, output_folder_path, resolution):

        super().__init__()

        self.canceled = Event()

        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

        self.resolution = resolution

    def startVolumeQuantification(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        # compute total volume by summing up pixels physical sizes
        # and write to txt
        total_num_pixels = count_segmentation_pixels(self.input_folder_path, isTif=None, channel_one=None, canceled=self.canceled, progress_status=self.progress_status)
        total_volume = total_num_pixels * self.resolution[0] * self.resolution[1] * self.resolution[2]

        time_str = time.strftime("%H%M%S_")
        out_name = time_str + "total_vessel_wall_volume(from object pixel count).txt"
        volume_file_path = Path(self.output_folder_path) / out_name

        out_str = "Total fenestrated blood vessel wall volume (computed from summing up object pixel volumes) [" + u"\u03bc" + "m^3]: " + str(
            total_volume)

        with open(volume_file_path, 'w', encoding="utf-8") as f:
            f.write(out_str)

        if not self.canceled.is_set():
            self.progress_status.emit(["Finished...", 100])

        self.finished.emit()

    def stop(self):
        self.canceled.set()
