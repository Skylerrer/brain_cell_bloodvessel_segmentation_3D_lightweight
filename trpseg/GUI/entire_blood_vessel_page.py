"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the entire blood vessel segmentation tab of the GUI
########################################################################################################################

import os
import time
from threading import Event

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
    QFileDialog,
    QDialog,
    QProgressBar,
    QTabWidget
)

import numpy as np
from pathlib import Path

from trpseg import DATA_DIRECTORY
from trpseg.trpseg_util.utility import get_file_list_from_directory, read_image, count_segmentation_pixels
from trpseg.segmentation.blood_vessel_NN_seg import network_predict_and_postprocess, get_default_NN_model_path
from trpseg.trpseg_util.graph_generation import perform_graph_generation
from trpseg.GUI.gui_utility import InputOutputWidgetSingleInput, show_error_message
from trpseg.GUI.file_picker import FilePicker

maximalNeededSizePolicy = QSizePolicy()
maximalNeededSizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
maximalNeededSizePolicy.setVerticalPolicy(QSizePolicy.Policy.Maximum)


class FilledVesselPage(QWidget):

    def __init__(self, parent_main_window):
        super().__init__(parent_main_window)

        self.parent_main_window = parent_main_window

        self.resolution = np.zeros(shape=(3,), dtype=np.float32)
        self.input_folder_path = ""
        self.model_path = get_default_NN_model_path()
        self.input_file_list_c00 = []
        self.input_file_list_c01 = []
        self.numFilesC00 = 0
        self.numFilesC01 = 0

        self.output_folder_path = ""
        self.new_input_folder_for_graph = ""

        self.do_post_remSmall = True
        self.do_post_fillHoles = True
        self.do_post_opening = True

        self.save_graph = True

        pageLayout = QVBoxLayout()
        self.setLayout(pageLayout)

        # 1. Choose Input/Output Directory------------------------------------------------------------------------------
        self.in_out_widget = InputOutputWidgetSingleInput()
        self.in_out_widget.input_dir_button.clicked.connect(self.chooseInputDir)
        self.in_out_widget.output_dir_button.clicked.connect(self.chooseOutputDir)

        pageLayout.addWidget(self.in_out_widget, Qt.AlignmentFlag.AlignTop)

        # 2. Choose deep neural network model---------------------------------------------------------------------------
        select_model_layout = QHBoxLayout()
        select_model_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        model_label = QLabel("Select Model Directory:")
        self.model_lineEdit = QLineEdit()
        self.model_lineEdit.setEnabled(False)
        self.model_lineEdit.setText(self.model_path)

        self.choose_model_button = QPushButton("Choose Model...")
        self.choose_model_button.setSizePolicy(maximalNeededSizePolicy)
        self.choose_model_button.clicked.connect(self.set_new_model_path)

        self.choose_model_button.setToolTip("<p>Choose a folder in which a trained neural network is stored.</p>"
                                            "<p>The model should previously have been stored via tensorflow.keras.Model save method.</p>")

        select_model_layout.addWidget(model_label)
        select_model_layout.addWidget(self.model_lineEdit)
        select_model_layout.addWidget(self.choose_model_button)

        pageLayout.addLayout(select_model_layout)

        # 3. Parameter Selection----------------------------------------------------------------------------------------
        parameterTabs = QTabWidget()
        pageLayout.addWidget(parameterTabs)

        dnn_page = QWidget(parameterTabs)
        dnn_layout = QVBoxLayout()
        dnn_page.setLayout(dnn_layout)

        parameterTabs.addTab(dnn_page, 'Neural Network Parameters')

        graph_page = QWidget(parameterTabs)
        graph_layout = QVBoxLayout()
        graph_page.setLayout(graph_layout)

        parameterTabs.addTab(graph_page, 'Graph Generation Parameters')


        #Parameters for DNN Prediction
        #Remove Small Regions
        post_remSmall_layout = QHBoxLayout()
        post_remSmall_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_remSmall_chbox = QCheckBox()
        self.post_remSmall_chbox.setChecked(self.do_post_remSmall)

        self.post_remSmall_chbox.stateChanged.connect(self.post_remSmall_change)

        self.post_remSmall_chbox.setToolTip("<p>If checked, then too small objects will be removed from the segmentation result.</p>")

        post_remSmall_layout.addWidget(self.post_remSmall_chbox)
        post_remSmall_layout.addWidget(QLabel("Postprocess (remove small objects)"))

        dnn_layout.addLayout(post_remSmall_layout)

        # Fill Holes
        post_fillHoles_layout = QHBoxLayout()
        post_fillHoles_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_fillHoles_chbox = QCheckBox()
        self.post_fillHoles_chbox.setChecked(self.do_post_fillHoles)

        self.post_fillHoles_chbox.stateChanged.connect(self.post_fillHoles_change)

        self.post_fillHoles_chbox.setToolTip("<p>If checked, then black holes inside segmented regions will be filled.</p>"
                                             "<p>This is done in 2D on every image slice separately.</p>")

        post_fillHoles_layout.addWidget(self.post_fillHoles_chbox)
        post_fillHoles_layout.addWidget(QLabel("Postprocess (fill holes)"))

        dnn_layout.addLayout(post_fillHoles_layout)

        # Opening
        post_opening_layout = QHBoxLayout()
        post_opening_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_opening_chbox = QCheckBox()
        self.post_opening_chbox.setChecked(self.do_post_opening)

        self.post_opening_chbox.stateChanged.connect(self.post_opening_change)

        self.post_opening_chbox.setToolTip("<p>If checked, then 3D morphological opening will be performed as postprocessing. "
                                           "This further removes unwanted small objects and smoothes the segmentation result a bit.</p>")

        post_opening_layout.addWidget(self.post_opening_chbox)
        post_opening_layout.addWidget(QLabel("Postprocess (opening)"))

        dnn_layout.addLayout(post_opening_layout)


        #Parameters for Graph Generation

        #save graph
        save_graph_layout = QHBoxLayout()
        save_graph_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.save_graph_chbox = QCheckBox()
        self.save_graph_chbox.setChecked(self.save_graph)

        self.save_graph_chbox.stateChanged.connect(self.save_graph_change)

        self.save_graph_chbox.setToolTip("<p>If checked, then the graph object that will be generated from the filled "
                                         "blood vessel segmentation will be stored via python pickle.</p>"
                                         "<p>The stored igraph.Graph object can be loaded and inspected by your own python scripts.</p>")

        save_graph_layout.addWidget(self.save_graph_chbox)
        save_graph_layout.addWidget(QLabel("Save graph"))

        graph_layout.addLayout(save_graph_layout)

        # 4. Buttons to start workers-----------------------------------------------------------------------------------
        # Button to start Deep Neural Network Prediction
        seg_start_layout = QHBoxLayout()
        self.seg_start_button = QPushButton("Start Segmentation")
        self.seg_start_button.setEnabled(False)
        self.seg_start_button.clicked.connect(self.start_network_predict)

        self.seg_start_button.setToolTip("<p>Start the filled blood vessel segmentation process.</p>"
                                         "<p>A deep neural network will be used to predict what belongs to blood "
                                         "vessels. The segmentation result will be postprocessed.</p>")

        seg_start_layout.addWidget(self.seg_start_button, Qt.AlignmentFlag.AlignHCenter)

        pageLayout.addLayout(seg_start_layout)



        # Button to start Quantification via pixel count
        pixel_count_quantify_layout = QHBoxLayout()
        self.pixel_count_quantify_start_button = QPushButton("Start Quantification by Object Pixel Count")
        self.pixel_count_quantify_start_button.setEnabled(False)
        self.pixel_count_quantify_start_button.clicked.connect(self.start_pixel_count_quantification)

        self.pixel_count_quantify_start_button.setToolTip("<p>Estimate the volume of blood vasculature in a given binary blood vessel segmentation."
                                                          " Statistics will be written to a txt file.</p>"
                                           "<p>The volume is computed as the number of object pixels multiplied by the pixel volume.</p>")

        pixel_count_quantify_layout.addWidget(self.pixel_count_quantify_start_button, Qt.AlignmentFlag.AlignHCenter)

        pageLayout.addLayout(pixel_count_quantify_layout)


        # Button to start Graph Generation
        graph_start_layout = QHBoxLayout()
        self.graph_start_button = QPushButton("Start Quantification via Graph Generation")
        self.graph_start_button.setEnabled(False)
        self.graph_start_button.clicked.connect(self.start_graph_generation)

        self.graph_start_button.setToolTip("<p>Generate a graph from a binary blood vessel segmentation result.</p>"
                                           "<p>Statistics of the graph will be written to a txt file. Furthermore, "
                                           "the generated igraph.Graph object is stored if the 'Save graph' checkbox is checked.</p>")

        graph_start_layout.addWidget(self.graph_start_button, Qt.AlignmentFlag.AlignHCenter)

        pageLayout.addLayout(graph_start_layout)


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
            self.seg_start_button.setEnabled(True)
            self.pixel_count_quantify_start_button.setEnabled(True)
            self.graph_start_button.setEnabled(True)

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_new_model_path(self):
        #home = os.path.expanduser("~")
        home = os.path.join(DATA_DIRECTORY, "NN_models")
        directory = QFileDialog.getExistingDirectory(self, "Select Model Directory", home)
        if directory != '':
            self.model_path = directory
            self.model_lineEdit.setText(directory)

    def set_new_input_folder_for_graph(self, folder_path):
        self.new_input_folder_for_graph = folder_path

    def set_input_dir_to_network_out(self):
        if self.new_input_folder_for_graph != "":
            self.input_folder_path = self.new_input_folder_for_graph
            self.in_out_widget.input_dir_line.setText(self.input_folder_path)
            self.input_file_list_c00 = get_file_list_from_directory(self.input_folder_path, isTif=None, channel_one=None)

            self.numFilesC00 = len(self.input_file_list_c00)
            self.in_out_widget.numFiles_label.setText(f"Number of Images: {self.numFilesC00}")

    def post_remSmall_change(self, state):
        if state == 0:
            self.do_post_remSmall = False
        else:
            self.do_post_remSmall = True

    def post_fillHoles_change(self, state):
        if state == 0:
            self.do_post_fillHoles = False
        else:
            self.do_post_fillHoles = True

    def post_opening_change(self, state):
        if state == 0:
            self.do_post_opening = False
        else:
            self.do_post_opening = True

    def save_graph_change(self, state):
        if state == 0:
            self.save_graph = False
        else:
            self.save_graph = True

    def start_network_predict(self):
        # Open progress dialog + start thread with parameters
        dlg = NetworkProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Segmentation successful")
            self.parent_main_window.show()
            self.set_input_dir_to_network_out()
        else:
            print("Segmentation unsuccessful")
            self.parent_main_window.show()

        dlg.deleteLater()

    def start_graph_generation(self):
        # Open progress dialog + start thread with parameters
        if read_image(self.input_file_list_c00[0]).dtype != np.uint8:
            show_error_message(self, "Graph Generation needs 8bit binary images as input.")
            return

        dlg = GraphProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Graph generation successful")
            self.parent_main_window.show()
        else:
            print("Graph generation unsuccessful")
            self.parent_main_window.show()

        dlg.deleteLater()

    def start_pixel_count_quantification(self):
        # Open progress dialog + start thread with parameters
        if read_image(self.input_file_list_c00[0]).dtype != np.uint8:
            show_error_message(self, "Quantification by pixel count needs 8bit binary images as input.")
            return

        dlg = QuantifyVolumeProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Volume quantification successful")
            self.parent_main_window.show()
        else:
            print("Volume quantification unsuccessful")
            self.parent_main_window.show()

        dlg.deleteLater()



class NetworkProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread neural network prediction + postprocessing and allows to cancel it."""
    def __init__(self, parent: FilledVesselPage):
        super().__init__(parent)

        self.setWindowTitle("Network Prediction")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Starting Filled Vessel Segmentation ...")

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
        # Thread to perform filled vessel segmentation
        self.segmentation_thread = QThread()

        self.worker = NetworkWorker(parent.input_file_list_c00, parent.output_folder_path, parent.resolution, parent.model_path,
                                   parent.do_post_remSmall, parent.do_post_fillHoles, parent.do_post_opening)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.segmentation_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.output_ready.connect(parent.set_new_input_folder_for_graph) #returns in which folder the final output from the network is located
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


class NetworkWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    output_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_file_paths, output_folder_path, resolution, model_path, remove_small_objects, fill_holes, do_opening):

        super().__init__()

        self.canceled = Event()

        self.input_file_paths = input_file_paths
        self.output_folder_path = output_folder_path

        self.resolution = resolution
        self.model_path = model_path
        self.remove_small_objects = remove_small_objects
        self.fill_holes = fill_holes
        self.do_opening = do_opening

    def startSegmentation(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        result_folder_path = network_predict_and_postprocess(self.input_file_paths, self.output_folder_path, self.model_path, self.resolution,
                                                             self.remove_small_objects, self.fill_holes, self.do_opening,
                                                             canceled=self.canceled, progress_status=self.progress_status)

        if not self.canceled.is_set():
            self.output_ready.emit(result_folder_path)

            self.progress_status.emit(["Finished...", 100])

        self.finished.emit()


    def stop(self):
        self.canceled.set()



class QuantifyVolumeProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for Blood Vasculature Volume quantification and allows to cancel it."""

    def __init__(self, parent: FilledVesselPage):
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
        # Thread to perform quantification
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
        out_name = time_str + "total_vessel_volume(from object pixel count).txt"
        volume_file_path = Path(self.output_folder_path) / out_name

        out_str = "Total fenestrated blood vessel volume (computed from summing up pixel volumes) [" + u"\u03bc" + "m^3]: " + str(
            total_volume)

        with open(volume_file_path, 'w', encoding="utf-8") as f:
            f.write(out_str)

        if not self.canceled.is_set():
            self.progress_status.emit(["Finished...", 100])

        self.finished.emit()

    def stop(self):
        self.canceled.set()


class GraphProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for generating a graph from the filled vessel segmentation and allows to cancel it."""
    def __init__(self, parent: FilledVesselPage):
        super().__init__(parent)

        self.setWindowTitle("Graph Generation")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Starting Graph Generation ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_seg_button = QPushButton("Cancel")
        self.cancel_seg_button.clicked.connect(self.cancel_generation)

        cancelLayout.addWidget(self.cancel_seg_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        # Thread to perform graph generation
        self.graph_thread = QThread()

        self.worker = GraphWorker(parent.input_folder_path, parent.output_folder_path, parent.resolution, parent.save_graph)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.graph_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.graph_thread.finished.connect(self.worker.deleteLater)

        self.graph_thread.started.connect(self.worker.startGraphGeneration)
        self.graph_thread.start()

    def close_on_success(self):
        self.graph_thread.quit()
        self.graph_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_generation(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.graph_thread.quit()
        self.graph_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_generation()
        event.ignore()


class GraphWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_folder_path, output_folder_path, resolution, save_graph):

        super().__init__()

        self.canceled = Event()

        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

        self.resolution = resolution
        self.save_graph = save_graph

    def startGraphGeneration(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        perform_graph_generation(self.input_folder_path, self.output_folder_path, self.resolution, self.save_graph, self.canceled, self.progress_status)

        if not self.canceled.is_set():
            self.progress_status.emit(["Finished...", 100])

        self.finished.emit()

    def stop(self):
        self.canceled.set()
