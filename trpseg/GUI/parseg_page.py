"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the pars tuberalis cells segmentation tab for the GUI
# It calls code for segmenting the cells as well as for counting the cells.
########################################################################################################################

import os
import time
from threading import Event

import numpy as np
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QComboBox,
    QDialog,
    QProgressBar,
    QTabWidget
)

from trpseg.trpseg_util.utility import get_file_list_from_directory, write_counting_stats_to_txt,\
    count_labels_3D_memeff, read_image
from trpseg.GUI.parameter_picker import ThresholdPickerGaussianSmoothed, ThresholdPicker
from trpseg.GUI.file_picker import FilePicker
from trpseg.segmentation.pars_tuberalis_cells_seg import segment_pars_tuberalis_cells, get_default_min_pars_cell_size,\
     get_remove_smaller_z_length_from_resolution, get_closing_radius_from_resolution, get_median_radius_from_resolution,\
     watershed_split_memeff, get_default_avg_pars_cell_size, get_default_final_pseg_output_name, get_default_brain_tissue_sigma,\
     estimateParsCellCounts
from trpseg.trpseg_util.z_normalization import z_stack_tissue_mean_normalization
from trpseg.GUI.gui_utility import InputOutputWidgetTwoInputs, InputOutputWidgetSingleInput, NormalizationWidget, show_error_message


maximalNeededSizePolicy = QSizePolicy()
maximalNeededSizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
maximalNeededSizePolicy.setVerticalPolicy(QSizePolicy.Policy.Maximum)


class ParsSegPage(QWidget):

    def __init__(self, parent_main_window):
        super().__init__(parent_main_window)

        self.parent_main_window = parent_main_window

        self.resolution = np.zeros(shape=(3,), dtype=np.float32)
        self.input_folder_path_blood = ""
        self.input_folder_path_pars = ""
        self.input_file_list_c00 = []
        self.input_file_list_c01 = []
        self.numFilesC00 = 0
        self.numFilesC01 = 0

        self.output_folder_path = ""
        self.store_intermediate = True

        self.pseg_prepend_str = time.strftime("%H%M%S_")
        self.final_pseg_folder_out = ""


        pageLayout = QVBoxLayout()
        pageLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(pageLayout)

        segmentationTabs = QTabWidget()
        pageLayout.addWidget(segmentationTabs)

        # Parseg Page Layout
        parseg_page_widget = QWidget()
        parseg_page_layout = QVBoxLayout()
        parseg_page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        parseg_page_widget.setLayout(parseg_page_layout)
        #pageLayout.setSpacing(0)

        segmentationTabs.addTab(parseg_page_widget, 'Cells Analysis')

        # 1. Choose Input/Output Directory------------------------------------------------------------------------------
        self.in_out_widget = InputOutputWidgetTwoInputs()
        self.in_out_widget.setContentsMargins(0,0,0,20)
        self.in_out_widget.input_dir_button_blood.clicked.connect(self.chooseInputDirBlood)
        self.in_out_widget.input_dir_button_pars.clicked.connect(self.chooseInputDirPars)
        self.in_out_widget.output_dir_button.clicked.connect(self.chooseOutputDir)

        parseg_page_layout.addWidget(self.in_out_widget, Qt.AlignmentFlag.AlignTop)


        # 2. Choose Parameters------------------------------------------------------------------------------------------

        #Basic Parameters
        self.remove_by_distance = False
        self.max_dist = 0
        self.brain_tissue_channel_one = False
        self.brain_tissue_threshold = 0
        self.pars_threshold = 0

        #Advanced automatically selected Parameters (They are set to default values after the resolution is set on the settings page)
        # Are initialized by self.initialize_advanced_parameters
        self.brain_tissue_sigma = None
        self.remove_smaller_than = None
        self.remove_smaller_z_width = None
        self.pars_median_radius = None
        self.closing_radius = None

        #Cell Counting Parameters
        self.avg_cell_size = None



        parameterTabs = QTabWidget()
        parseg_page_layout.addWidget(parameterTabs)

        #Segmentation Parameters
        advanced_para_page = QWidget(parameterTabs)
        parameter_Layout = QVBoxLayout()
        advanced_para_page.setLayout(parameter_Layout)

        parameterTabs.addTab(advanced_para_page, 'Segmentation Parameters')


        #The code that is commented out is not needed any longer since these parameters are automatically selected
        """
        #Brain Tissue Sigma
        brain_sigma_layout = QHBoxLayout()
        brain_sigma_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        brain_tissue_sigma_label = QLabel("Choose smoothing sigma for brain tissue detection:")

        self.brain_tissue_sigma_spinbox = QSpinBox()
        self.brain_tissue_sigma_spinbox.setMinimum(1)
        self.brain_tissue_sigma_spinbox.setMaximum(100)
        self.brain_tissue_sigma_spinbox.setSingleStep(1)
        self.brain_tissue_sigma_spinbox.valueChanged.connect(self.set_brain_sigma)

        self.brain_tissue_sigma_spinbox.setToolTip("<p>Choose the standard deviation (sigma) of the gaussian filter"
                                                   "that will be applied to the images for brain tissue detection.</p>"
                                                   "<p>Only needed if 'Remove by distance' checkbox is checked.</p>")

        brain_sigma_layout.addWidget(brain_tissue_sigma_label)
        brain_sigma_layout.addWidget(self.brain_tissue_sigma_spinbox)
        parameter_Layout.addLayout(brain_sigma_layout)

        #Median Radius
        median_radius_layout = QHBoxLayout()
        median_radius_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        median_radius_label = QLabel("Choose median filter radius [px]:")

        self.median_radius_spinbox = QSpinBox()
        self.median_radius_spinbox.setMinimum(1)
        self.median_radius_spinbox.setMaximum(100)
        self.median_radius_spinbox.setSingleStep(1)
        self.median_radius_spinbox.valueChanged.connect(self.set_median_radius)

        self.median_radius_spinbox.setToolTip("<p>Set the radius (in pixels) for the median filter that will be applied"
                                              " as a preprocessing step before segmenting the pars tuberalis cells.</p>")

        median_radius_layout.addWidget(median_radius_label)
        median_radius_layout.addWidget(self.median_radius_spinbox)
        parameter_Layout.addLayout(median_radius_layout)

        #Remove_smaller_than
        remove_smaller_volume_layout = QHBoxLayout()
        remove_smaller_volume_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_smaller_volume_label = QLabel("Choose minimal cell volume [" + "\u03bc" +"m" + "\u00b3]:")

        self.remove_smaller_volume_spinbox = QSpinBox()
        self.remove_smaller_volume_spinbox.setMinimum(1)
        self.remove_smaller_volume_spinbox.setMaximum(100000)
        self.remove_smaller_volume_spinbox.setSingleStep(10)
        self.remove_smaller_volume_spinbox.valueChanged.connect(self.set_smaller_than)

        self.remove_smaller_volume_spinbox.setToolTip("<p>In the segmentation result, objects that are smaller than "
                                                      "this value will be removed during postprocessing.</p>")

        remove_smaller_volume_layout.addWidget(remove_smaller_volume_label)
        remove_smaller_volume_layout.addWidget(self.remove_smaller_volume_spinbox)
        parameter_Layout.addLayout(remove_smaller_volume_layout)

        #Remove_smaller_z_width
        remove_smaller_z_layout = QHBoxLayout()
        remove_smaller_z_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_smaller_z_label = QLabel("Choose cells minimal allowed z expansion [px]:")

        self.remove_smaller_z_spinbox = QSpinBox()
        self.remove_smaller_z_spinbox.setMinimum(1)
        self.remove_smaller_z_spinbox.setMaximum(1000)
        self.remove_smaller_z_spinbox.setSingleStep(1)
        self.remove_smaller_z_spinbox.valueChanged.connect(self.set_smaller_z)

        self.remove_smaller_z_spinbox.setToolTip("<p>In the segmentation result, objects that expand over less than "
                                                 "this number of z slices (pixels in z direction) will be removed "
                                                 "during postprocessing</p>")

        remove_smaller_z_layout.addWidget(remove_smaller_z_label)
        remove_smaller_z_layout.addWidget(self.remove_smaller_z_spinbox)
        parameter_Layout.addLayout(remove_smaller_z_layout)

        #Closing_radius
        closing_radius_layout = QHBoxLayout()
        closing_radius_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        closing_radius_label = QLabel("Choose postprocessing closing radius [px]:")

        self.closing_radius_spinbox = QSpinBox()
        self.closing_radius_spinbox.setMinimum(1)
        self.closing_radius_spinbox.setMaximum(1000)
        self.closing_radius_spinbox.setSingleStep(1)
        self.closing_radius_spinbox.valueChanged.connect(self.set_closing_radius)

        self.closing_radius_spinbox.setToolTip("<p>Choose the radius of the 3D morphological closing operation which "
                                               "will be performed during postprocessing.</p>"
                                               "<p>The radius has to be given in pixels in the x-y image plane."
                                               "The radius in z-direction will be determined automatically.</p>"
                                               "<p>The puporse of closing is to fill small black holes in the segmentation result. "
                                               "The larger the radius, the larger are the holes that will be filled. "
                                               "But also the more the original segmentation result will be altered.</p>")

        closing_radius_layout.addWidget(closing_radius_label)
        closing_radius_layout.addWidget(self.closing_radius_spinbox)
        parameter_Layout.addLayout(closing_radius_layout)
        """


        # Select global threshold for pars tuberalis cells segmentation----------------------------------------------
        pars_th_layout = QHBoxLayout()
        #pars_th_layout.setContentsMargins(10, 30, 10, 30)
        pars_th_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        pars_th_label = QLabel("Choose threshold to detect pars tuberalis cells:")

        self.pars_th_spin = QSpinBox()
        self.pars_th_spin.setMinimum(0)
        self.pars_th_spin.setMaximum(65535)
        self.pars_th_spin.setSingleStep(10)
        self.pars_th_spin.setEnabled(False)
        self.pars_th_spin.valueChanged.connect(self.set_pars_th)

        self.pars_th_spin.setToolTip(
            "<p>Every pixel with intensity value higher than this threshold is considered to belong to the cells "
            "you want to segment. The result from this global thresholding will be further"
            " postprocessed.</p>")

        self.pars_th_button = QPushButton("Use ThPicker")
        self.pars_th_button.setEnabled(False)
        self.pars_th_button.clicked.connect(self.use_pars_th_picker)

        pars_th_layout.addWidget(pars_th_label)
        pars_th_layout.addWidget(self.pars_th_spin)
        pars_th_layout.addWidget(self.pars_th_button)

        parameter_Layout.addLayout(pars_th_layout)


        ## Remove by distance
        dist_page = QWidget()
        distanceLayout = QHBoxLayout()
        dist_page.setLayout(distanceLayout)

        parameter_Layout.addWidget(dist_page, alignment=Qt.AlignmentFlag.AlignLeft)

        distanceLayout.setContentsMargins(0,20,0,20)

        # checkBox remove_by_distance
        check_dist_layout = QHBoxLayout()
        check_dist_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.remove_by_distance_chbox = QCheckBox()
        self.remove_by_distance_chbox.setChecked(False)
        self.remove_by_distance_chbox.setEnabled(False)
        self.remove_by_distance_chbox.setToolTip("<p>This determines whether objects that are farther away from the "
                                                 "outer brain boundary than a specified max distance should be removed in the segmentation result.</p>"
                                                 "<p>The distance will be considered in 3D.</p>")

        self.remove_by_distance_chbox.stateChanged.connect(self.remove_by_distance_change)

        check_dist_layout.addWidget(self.remove_by_distance_chbox)
        check_dist_layout.addWidget(QLabel("Remove cells by distance"))

        distanceLayout.addLayout(check_dist_layout)

        # Distance Parameter Layout
        dist_parameter_layout = QVBoxLayout()
        dist_parameter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Choose max_distance
        max_dist_layout = QHBoxLayout()
        max_dist_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        max_dist_layout.addWidget(QLabel("Maximal allowed distance from brain boundary [" + "\u03bc" + "m]:"))

        self.distSpinBox = QDoubleSpinBox()
        self.distSpinBox.setMinimum(0)
        self.distSpinBox.setMaximum(10000)
        self.distSpinBox.setDecimals(1)
        self.distSpinBox.setSingleStep(10)
        self.distSpinBox.setEnabled(False)
        self.distSpinBox.valueChanged.connect(self.set_max_dist)

        self.distSpinBox.setToolTip(
            "<p>Set the maximal allowed distance from the outer brain boundary in " + "\u03bc" + "m.</p>"
                                                                                                 "<p>The distance will be considered in 3D.</p>")

        max_dist_layout.addWidget(self.distSpinBox)

        dist_parameter_layout.addLayout(max_dist_layout)

        # Choose BrainTissue Channel
        brain_channel_layout = QHBoxLayout()
        brain_channel_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        b_ch_label = QLabel("Choose channel to use for brain tissue detection:")
        self.b_ch_cbox = QComboBox()
        self.b_ch_cbox.addItems(["Blood Channel", "Pars Channel"])

        self.b_ch_cbox.setCurrentIndex(0)
        self.b_ch_cbox.setEnabled(False)
        self.b_ch_cbox.currentIndexChanged.connect(self.set_brain_channel)

        self.b_ch_cbox.setToolTip(
            "<p>Choose the channel in which the brain tissue has the highest contrast to the ventricle and the outside of the brain.</p>"
            "<p>Brain tissue segmentation is necessary to determine distances to the outer brain boundary.</p>")

        brain_channel_layout.addWidget(b_ch_label)
        brain_channel_layout.addWidget(self.b_ch_cbox)

        dist_parameter_layout.addLayout(brain_channel_layout)

        # Choose BrainTissue threshold
        brain_th_layout = QHBoxLayout()
        brain_th_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        brain_th_label = QLabel("Choose threshold to detect brain tissue:")
        self.brain_th_spin = QSpinBox()
        self.brain_th_spin.setMinimum(0)
        self.brain_th_spin.setMaximum(65535)
        self.brain_th_spin.setSingleStep(10)
        self.brain_th_spin.setEnabled(False)
        self.brain_th_spin.valueChanged.connect(self.set_brain_th)

        self.brain_th_spin.setToolTip(
            "<p>Choose threshold that includes the brain tissue but not the outside of the brain.</p>")

        self.brain_th_button = QPushButton("Use ThPicker")
        self.brain_th_button.setEnabled(False)
        self.brain_th_button.clicked.connect(self.use_brain_th_picker)

        brain_th_layout.addWidget(brain_th_label)
        brain_th_layout.addWidget(self.brain_th_spin)
        brain_th_layout.addWidget(self.brain_th_button)
        dist_parameter_layout.addLayout(brain_th_layout)

        distanceLayout.addLayout(dist_parameter_layout)


        # Store intermediate results
        store_intermed_layout = QHBoxLayout()
        store_intermed_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.store_intermed_chbox = QCheckBox()
        self.store_intermed_chbox.setChecked(self.store_intermediate)

        self.store_intermed_chbox.stateChanged.connect(self.store_intermed_change)

        store_intermed_layout.addWidget(self.store_intermed_chbox)
        store_intermed_layout.addWidget(QLabel("Store intermediate images"))

        self.store_intermed_chbox.setToolTip("<p>If this is checked then intermediate results from the segmentation "
                                             "pipeline will stored. This includes:</p>"
                                             "<p>The brain tissue segmentation result if objects are removed by distance</p>"
                                             "<p>The distance map if objects are removed by distance</p>"
                                             "<p>The segmentation result after global thresholding and the possible distance dependent object removal.</p>"
                                             "<p>The results after small objects have been removed.</p>")

        parameter_Layout.addLayout(store_intermed_layout)


        ## Cell Counting Parameters
        cellcount_para_page = QWidget(parameterTabs)
        cellcount_para_layout = QVBoxLayout()
        cellcount_para_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        cellcount_para_page.setLayout(cellcount_para_layout)

        parameterTabs.addTab(cellcount_para_page, 'Cell Counting Parameters')

        # Determine cell count by dividing through average cell size

        #Allow to choose average cell size
        avg_cell_size_layout = QHBoxLayout()
        avg_cell_size_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        avg_cell_size_label = QLabel("Average cell size [" + "\u03bc" + "m" + "\u00b3]:")

        self.avg_cell_size_spinbox = QSpinBox()
        self.avg_cell_size_spinbox.setMinimum(1)
        self.avg_cell_size_spinbox.setMaximum(100000)
        self.avg_cell_size_spinbox.setSingleStep(10)
        self.avg_cell_size_spinbox.valueChanged.connect(self.set_average_cell_size)

        self.avg_cell_size_spinbox.setToolTip("<p>Set the average cell size in cubic micrometers.</p>"
                                              "<p>It will be used to count cells. First the total cells volume is computed"
                                              " as the number of object pixels multiplied by the pixel volume. The result"
                                              " is divided by the average cell size to get the cell count.</p>")

        avg_cell_size_layout.addWidget(avg_cell_size_label)
        avg_cell_size_layout.addWidget(self.avg_cell_size_spinbox)
        cellcount_para_layout.addLayout(avg_cell_size_layout)

        #Initialize Cell Counting Parameters
        self.initialize_cellcount_parameters()




        ## Mean Tissue Normalization
        self.normalize_folder_out = ""

        normalization_widget_placeholder = QWidget(parameterTabs)

        parameterTabs.addTab(normalization_widget_placeholder, 'Tissue Mean Normalization')

        self.normalization_widget = NormalizationWidget(normalization_widget_placeholder)

        self.normalization_widget.normalization_min_button.clicked.connect(self.use_normalization_min_threshold_picker)
        self.normalization_widget.normalization_max_button.clicked.connect(self.use_normalization_max_threshold_picker)
        self.normalization_widget.normalization_button.clicked.connect(self.perform_normalization)


        # 3. Button to start pars tuberalis cell segmentation-----------------------------------------------------------
        pars_seg_start_layout = QHBoxLayout()
        self.pars_seg_start_button = QPushButton("Start Segmentation")
        self.pars_seg_start_button.setEnabled(False)
        self.pars_seg_start_button.clicked.connect(self.start_pars_seg)

        self.pars_seg_start_button.setToolTip("<p>Start the pars tuberalis cells segmentation pipeline.</p>")

        pars_seg_start_layout.addWidget(self.pars_seg_start_button, Qt.AlignmentFlag.AlignHCenter)

        parseg_page_layout.addLayout(pars_seg_start_layout)

        # 4. Button to count cells--------------------------------------------------------------------------------------
        count_cells_start_layout = QHBoxLayout()
        self.count_cells_start_button = QPushButton("Estimate Cell Count")
        self.count_cells_start_button.setEnabled(False)
        self.count_cells_start_button.clicked.connect(self.start_count_cells)

        self.count_cells_start_button.setToolTip("<p>Start estimating the cell count given a binary segmentation as input."
                                                 " Statistics will be written to a txt file.</p>"
                                                 " <p>The cell count will be determined by dividing"
                                                 " the total cells volume (number object pixels * pixel volume) by"
                                                 " the average cell size.</p>")

        count_cells_start_layout.addWidget(self.count_cells_start_button, Qt.AlignmentFlag.AlignHCenter)
        parseg_page_layout.addLayout(count_cells_start_layout)

        # 5. Watershed Splitting Cells Page-----------------------------------------------------------------------------
        self.watershed_page_widget = WatershedWidget(self)
        segmentationTabs.addTab(self.watershed_page_widget, 'Watershed Cell Splitting')


    def start_pars_seg(self):
        #Check whether needed parameters have been set by user
        if self.remove_by_distance:
            if self.max_dist == 0 or self.brain_tissue_threshold == 0:
                show_error_message(self, "You need to specify a max distance and the brain tissue threshold!")
                return

        if self.pars_threshold == 0:
            show_error_message(self, "You need to specify a threshold value for pars tuberalis cells segmentation!")
            return

        #Determine how folders will be named
        self.pseg_prepend_str = time.strftime("%H%M%S_")

        final_out_folder_name = self.pseg_prepend_str + get_default_final_pseg_output_name()
        self.final_pseg_folder_out = str(self.output_folder_path) + "\\" + final_out_folder_name

        #Open progress dialog + start thread with parameters
        seg_dlg = ParsProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if seg_dlg.exec():
            print("Segmentation successful")
            self.parent_main_window.show()
            self.set_input_dir_to_result_folder()
        else:
            print("Segmentation unsuccessful")
            self.parent_main_window.show()
        seg_dlg.deleteLater()

    def start_count_cells(self):
        if read_image(self.input_file_list_c01[0]).dtype != np.uint8:
            show_error_message(self, "Counting Cells only works on 8bit binary images!")
            return

        dlg = CountCellsProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Counting successful")
            self.parent_main_window.show()
        else:
            print("Counting unsuccessful")
            self.parent_main_window.show()
        dlg.deleteLater()

    def set_input_dir_to_result_folder(self):
        self.input_folder_path_pars = self.final_pseg_folder_out
        self.in_out_widget.input_dir_line_pars.setText(str(self.final_pseg_folder_out))
        self.input_file_list_c01 = get_file_list_from_directory(self.input_folder_path_pars, isTif=None, channel_one=None)

        self.numFilesC01 = len(self.input_file_list_c01)
        self.in_out_widget.numFilesBlood_label.setText(f"Number of Images (Pars): {self.numFilesC01}")

    def initialize_advanced_parameters(self):
        self.brain_tissue_sigma = get_default_brain_tissue_sigma()
        self.remove_smaller_than = get_default_min_pars_cell_size()
        self.remove_smaller_z_width = get_remove_smaller_z_length_from_resolution(self.resolution)
        self.pars_median_radius = get_median_radius_from_resolution(self.resolution)
        self.closing_radius = get_closing_radius_from_resolution(self.resolution)
        self.store_intermediate = True

        self.store_intermed_chbox.setChecked(self.store_intermediate)
        #Not needed anymore because computed automatically and less parameters better to not overwhelm user
        #self.brain_tissue_sigma_spinbox.setValue(self.brain_tissue_sigma)
        #self.median_radius_spinbox.setValue(self.pars_median_radius)
        #self.remove_smaller_volume_spinbox.setValue(self.remove_smaller_than)
        #self.remove_smaller_z_spinbox.setValue(self.remove_smaller_z_width)
        #self.closing_radius_spinbox.setValue(self.closing_radius)

    def initialize_cellcount_parameters(self):
        self.avg_cell_size = get_default_avg_pars_cell_size()

        self.avg_cell_size_spinbox.setValue(self.avg_cell_size)

    def use_normalization_min_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c01)

        if th_dlg.exec():
            self.normalization_widget.tissue_min = th_dlg.chosen_threshold
            self.normalization_widget.normalization_min_spin.setValue(self.normalization_widget.tissue_min)

        th_dlg.deleteLater()

    def use_normalization_max_threshold_picker(self):
        th_dlg = ThresholdPicker()

        th_dlg.set_image_paths(self.input_file_list_c01)

        if th_dlg.exec():
            self.normalization_widget.tissue_max = th_dlg.chosen_threshold
            self.normalization_widget.normalization_max_spin.setValue(self.normalization_widget.tissue_max)

        th_dlg.deleteLater()

    def perform_normalization(self):
        # Checks whether needed parameters have been set
        if self.normalization_widget.tissue_max == 0 or self.normalization_widget.tissue_min == 0:
            show_error_message(self, "A tissue min > 0 and a tissue max > 0 need to be specified!")
            return

        if self.normalization_widget.tissue_max <= self.normalization_widget.tissue_min:
            show_error_message(self, "For normalization the tissue max needs to be larger than tissue min.")
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
        self.input_folder_path_pars = self.normalize_folder_out
        self.in_out_widget.input_dir_line_pars.setText(self.normalize_folder_out)
        self.input_file_list_c01 = get_file_list_from_directory(self.input_folder_path_pars, isTif=None, channel_one=None)

        self.numFilesC01 = len(self.input_file_list_c01)
        self.in_out_widget.numFilesPars_label.setText(f"Number of Images (Pars): {self.numFilesC01}")

    def store_intermed_change(self, state):
        if state == 0:
            self.store_intermediate = False
        else:
            self.store_intermediate = True

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.watershed_page_widget.set_resolution(resolution)

    def set_brain_sigma(self, value):
        self.brain_tissue_sigma = value

    def set_median_radius(self, value):
        self.pars_median_radius = value

    def set_smaller_than(self, value):
        self.remove_smaller_than = value

    def set_smaller_z(self, value):
        self.remove_smaller_z_width = value

    def set_closing_radius(self, value):
        self.closing_radius = value

    def use_pars_th_picker(self):
        #th_dlg = ThresholdPickerMedianSmoothed(self.pars_median_radius)
        #We use here gaussian instead of median because median is slower and we only want to choose a coarse threshold
        th_dlg = ThresholdPickerGaussianSmoothed(2)

        th_dlg.set_image_paths(self.input_file_list_c01)

        if th_dlg.exec():
            self.pars_threshold = th_dlg.chosen_threshold
            self.pars_th_spin.setValue(self.pars_threshold)

        th_dlg.deleteLater()

    def use_brain_th_picker(self):
        th_dlg = ThresholdPickerGaussianSmoothed(self.brain_tissue_sigma)
        if self.brain_tissue_channel_one:
            th_dlg.set_image_paths(self.input_file_list_c01)
        else:
            th_dlg.set_image_paths(self.input_file_list_c00)

        if th_dlg.exec():
            self.brain_tissue_threshold = th_dlg.chosen_threshold
            self.brain_th_spin.setValue(self.brain_tissue_threshold)

        th_dlg.deleteLater()

    def set_pars_th(self, value):
        self.pars_threshold = value

        if self.numFilesC00 > 0 and self.numFilesC01 > 0 and self.output_folder_path != "":
            self.pars_seg_start_button.setEnabled(True)

    def set_brain_th(self, value):
        self.brain_tissue_threshold = value

    def set_brain_channel(self, index):
        if index == 0:
            self.brain_tissue_channel_one = False
        else:
            self.brain_tissue_channel_one = True

    def set_max_dist(self, val):
        self.max_dist = val

    def remove_by_distance_change(self, state):
        if state == 0:
            self.remove_by_distance = False
            self.toggle_distance_params(False)
        else:
            self.remove_by_distance = True
            self.toggle_distance_params(True)

    def toggle_distance_params(self, enable):
        self.distSpinBox.setEnabled(enable)
        self.b_ch_cbox.setEnabled(enable)
        self.brain_th_spin.setEnabled(enable)
        self.brain_th_button.setEnabled(enable)


    def set_average_cell_size(self, value):
        self.avg_cell_size = value

    def chooseInputDirBlood(self):
        file_picker = FilePicker()

        if file_picker.exec():
            if file_picker.input_folder_path != '':
                self.input_folder_path_blood = file_picker.input_folder_path
                self.in_out_widget.input_dir_line_blood.setText(self.input_folder_path_blood)
                self.input_file_list_c00 = file_picker.input_file_paths
                self.numFilesC00 = len(self.input_file_list_c00)
                self.in_out_widget.numFilesBlood_label.setText(f"Number of Images (Blood): {self.numFilesC00}")
        file_picker.deleteLater()

        self.checkEnableElements()

    def chooseInputDirPars(self):
        file_picker = FilePicker()

        if file_picker.exec():
            if file_picker.input_folder_path != '':
                self.input_folder_path_pars = file_picker.input_folder_path
                self.in_out_widget.input_dir_line_pars.setText(self.input_folder_path_pars)
                self.input_file_list_c01 = file_picker.input_file_paths
                self.numFilesC01 = len(self.input_file_list_c01)
                self.in_out_widget.numFilesPars_label.setText(f"Number of Images (Pars): {self.numFilesC01}")
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
        if self.numFilesC00 > 0 and self.numFilesC01 > 0 and self.output_folder_path != "":
            self.remove_by_distance_chbox.setEnabled(True)
            self.pars_th_spin.setEnabled(True)
            self.pars_th_button.setEnabled(True)
            self.pars_seg_start_button.setEnabled(True)

        if self.numFilesC01 > 0 and self.output_folder_path != "":
            self.count_cells_start_button.setEnabled(True)
            self.normalization_widget.normalization_button.setEnabled(True)
            self.normalization_widget.normalization_min_button.setEnabled(True)
            self.normalization_widget.normalization_max_button.setEnabled(True)


class NormalizationProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for tissue mean normalization of channel 1 image slices and allows to cancel it."""
    def __init__(self, parent: ParsSegPage):
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
        # Thread to perform tissue mean normalization
        self.normalization_thread = QThread()

        self.worker = NormalizationWorker(parent.input_file_list_c01, parent.normalize_folder_out, parent.normalization_widget.tissue_min, parent.normalization_widget.tissue_max)

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


class ParsProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for pars tuberalis cells segmentaion + postprocessing and allows to cancel it."""
    def __init__(self, parent: ParsSegPage):
        super().__init__(parent)

        self.setWindowTitle("Segment Pars Tuberalis Cells")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Starting Segmentation ...")

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
        # Thread to perform pars tuberalis cells segmentation
        self.seg_thread = QThread()

        self.worker = ParSegWorker(parent.input_file_list_c01, parent.input_file_list_c00, parent.output_folder_path, parent.resolution,
                                   parent.pars_threshold, parent.remove_by_distance, parent.remove_smaller_than,
                                   parent.remove_smaller_z_width, parent.max_dist,
                                   parent.brain_tissue_sigma, parent.brain_tissue_threshold, parent.brain_tissue_channel_one,
                                   parent.pars_median_radius, parent.closing_radius, parent.store_intermediate, parent.pseg_prepend_str)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.seg_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.seg_thread.finished.connect(self.worker.deleteLater)

        self.seg_thread.started.connect(self.worker.startSegmentation)
        self.seg_thread.start()

    def close_on_success(self):
        self.seg_thread.quit()
        self.seg_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_segmentation(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.seg_thread.quit()
        self.seg_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_segmentation()
        event.ignore()


class ParSegWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_file_paths_pars, input_file_paths_blood, output_folder_path, resolution, pars_threshold,
                 remove_by_distance, remove_smaller_than, remove_smaller_z_width, max_dist, brain_tissue_sigma,
                 brain_tissue_threshold, brain_channel_one, pars_median_radius, closing_radius,
                 store_intermediate, prepend_to_folder_name):

        super().__init__()

        self.canceled = Event()

        self.input_file_paths_pars = input_file_paths_pars
        self.input_file_paths_blood = input_file_paths_blood
        self.output_folder_path = output_folder_path
        self.resolution = resolution
        self.pars_threshold = pars_threshold
        self.remove_by_distance = remove_by_distance
        self.max_dist = max_dist
        self.brain_tissue_threshold = brain_tissue_threshold
        self.brain_channel_one = brain_channel_one
        self.del_intermediate = not store_intermediate

        #Advanced Parameters (can be determined automatically but can also be chosen by user)
        self.brain_tissue_sigma = brain_tissue_sigma
        self.remove_smaller_than = remove_smaller_than
        self.remove_smaller_z_width = remove_smaller_z_width
        self.median_radius = pars_median_radius
        self.closing_radius = closing_radius

        self.prepend_to_folder_name = prepend_to_folder_name

    def startSegmentation(self):
        segment_pars_tuberalis_cells(self.input_file_paths_pars, self.input_file_paths_blood, self.output_folder_path, self.resolution,
                                     self.pars_threshold, self.remove_by_distance, self.brain_tissue_threshold,
                                     remove_smaller_than=self.remove_smaller_than,
                                     remove_smaller_z_length=self.remove_smaller_z_width, max_dist=self.max_dist,
                                     brain_region_sigma=self.brain_tissue_sigma,
                                     brain_channel_one=self.brain_channel_one, median_radius=self.median_radius,
                                     closing_radius=self.closing_radius, delete_intermed_results=self.del_intermediate,
                                     prepend_to_folder_name=self.prepend_to_folder_name, canceled=self.canceled,
                                     progress_status=self.progress_status)

        if not self.canceled.is_set():
            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()

    def stop(self):
        self.canceled.set()


class CountCellsProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for counting the pars tuberalis cells and allows to cancel it."""
    def __init__(self, parent: ParsSegPage):
        super().__init__(parent)

        self.setWindowTitle("Count Pars Tuberalis Cells")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0, 100)
        self.progBarText = QLabel("Start Counting ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_counting)

        cancelLayout.addWidget(self.cancel_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        self.thread = QThread()

        self.worker = CountCellsWorker(parent.input_folder_path_pars, parent.output_folder_path, parent.resolution, parent.avg_cell_size)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.thread.finished.connect(self.worker.deleteLater)

        self.thread.started.connect(self.worker.startCounting)
        self.thread.start()

    def close_on_success(self):
        self.thread.quit()
        self.thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_counting(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_counting()
        event.ignore()


class CountCellsWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_folder_path, output_folder_path, resolution, average_cell_size):

        super().__init__()

        self.canceled = Event()

        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.file_output_path = output_folder_path + "\\" + "cell_counting_results.txt"
        self.resolution = resolution
        self.average_cell_size = average_cell_size


    def startCounting(self):


        num_cells = estimateParsCellCounts(self.input_folder_path, self.resolution, self.average_cell_size, canceled=self.canceled, progress_status=self.progress_status)
        self.file_output_path = self.output_folder_path + "\\" + "estimated_cell_count_results.txt"

        if not self.canceled.is_set():
            progress = 95
            self.progress_status.emit(["Writing results to txt...", progress])

            write_counting_stats_to_txt(self.file_output_path, num_cells)

            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()

    def stop(self):
        self.canceled.set()


class WatershedWidget(QWidget):
    """The subtab for performing watershed transformation."""
    def __init__(self, parent: ParsSegPage):
        super().__init__()

        self.parent_main_window = parent.parent_main_window

        self.input_folder_path = ""
        self.input_file_list_c00 = []
        self.input_file_list_c01 = []
        self.numFilesC00 = 0
        self.numFilesC01 = 0

        self.output_folder_path = ""
        self.watershed_folder_out = ""

        self.resolution = np.zeros(shape=(3,), dtype=np.float32)


        watershed_page_layout = QVBoxLayout()
        watershed_page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(watershed_page_layout)

        # 1. Choose Input Directory-------------------------------------------------------------------------------------
        self.in_out_widget = InputOutputWidgetSingleInput()
        self.in_out_widget.input_dir_button.clicked.connect(self.chooseInputDir)
        self.in_out_widget.output_dir_button.clicked.connect(self.chooseOutputDir)

        watershed_page_layout.addWidget(self.in_out_widget, Qt.AlignmentFlag.AlignTop)


        # 2. Button to start watershed cell splitting-------------------------------------------------------------------
        split_cells_layout = QHBoxLayout()
        self.split_cells_button = QPushButton("Start Splitting Cells")
        self.split_cells_button.setEnabled(False)
        self.split_cells_button.clicked.connect(self.start_watershed)

        self.split_cells_button.setToolTip("<p>Use watershed segmentation on a given 3D binary segmentation to "
                                           "split clustered pars tuberalis cells.</p>"
                                           "<p>Important!: Only useful if the pars tuberalis cells are already pretty "
                                           "well isolated and touch only slightly. Also it will be assumed that the "
                                           "cells have a roundish shape.</p>")

        split_cells_layout.addWidget(self.split_cells_button, Qt.AlignmentFlag.AlignHCenter)

        watershed_page_layout.addLayout(split_cells_layout)

        # 3. Button to count labels from watershed result---------------------------------------------------------------
        """ # Currently this is commented out because it will normally not be used within the segmenation pipeline.
        # If you want to count labels then use the method count_labels_3D_memeff(...) directly
        count_labels_start_layout = QHBoxLayout()
        self.count_labels_start_button = QPushButton("Count Cell Labels")
        self.count_labels_start_button.setEnabled(False)
        self.count_labels_start_button.clicked.connect(self.start_count_labels)

        self.count_labels_start_button.setToolTip("<p>Use this to count the number of integer labels given a labeled image stack as input.</p>"
                                                  "<p>In a labeled image stack, different objects are assigned different integers.</p>"
                                                  "<p>The resulting count will be written to a txt file.</p>")
        count_labels_start_layout.addWidget(self.count_labels_start_button, Qt.AlignmentFlag.AlignHCenter)
        watershed_page_layout.addLayout(count_labels_start_layout)
        """

    def chooseInputDir(self):
        file_picker = FilePicker()

        if file_picker.exec():
            if file_picker.input_folder_path != '':
                self.input_folder_path = file_picker.input_folder_path
                self.in_out_widget.input_dir_line.setText(self.input_folder_path)
                self.input_file_list_c01 = file_picker.input_file_paths
                self.numFilesC01 = len(self.input_file_list_c01)
                self.in_out_widget.numFiles_label.setText(f"Number of Images: {self.numFilesC01}")
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
        if self.numFilesC01 > 0 and self.output_folder_path != "":
            self.split_cells_button.setEnabled(True)
            #This button is not any longer included in GUI (see above commented out code:self.count_labels_start_button)
            #self.count_labels_start_button.setEnabled(True)

    def set_input_dir_to_result_folder(self):
        self.input_folder_path = self.watershed_folder_out
        self.in_out_widget.input_dir_line.setText(self.watershed_folder_out)
        self.input_file_list_c01 = get_file_list_from_directory(self.input_folder_path, isTif=None, channel_one=None)

        self.numFilesC01 = len(self.input_file_list_c01)
        self.in_out_widget.numFiles_label.setText(f"Number of Images: {self.numFilesC01}")

    def set_resolution(self, resolution):
        self.resolution = resolution

    def start_watershed(self):
        # Open progress dialog + start thread

        timestr = time.strftime("%H%M%S_")

        out_folder_name = timestr + "watershed_result"
        self.watershed_folder_out = self.output_folder_path + "\\" + out_folder_name
        self.watershed_folder_out = self.watershed_folder_out

        dlg = WatershedProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Watershed Splitting successful")
            self.parent_main_window.show()
            self.set_input_dir_to_result_folder()
        else:
            print("Watershed Splitting unsuccessful")
            self.parent_main_window.show()
        dlg.deleteLater()

    def start_count_labels(self):
        dlg = CountLabelsProgressDialog(self)

        if self.parent_main_window.hide_main_window_when_processing:
            self.parent_main_window.hide()

        if dlg.exec():
            print("Counting labels successful")
            self.parent_main_window.show()
        else:
            print("Counting labels unsuccessful")
            self.parent_main_window.show()
        dlg.deleteLater()


class WatershedProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for performing watershed transformation and allows to cancel it."""
    def __init__(self, parent: WatershedWidget):
        super().__init__(parent)

        self.setWindowTitle("Watershed Segmentation")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0, 100)
        self.progBarText = QLabel("Starting Watershed Splitting ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_watershed)

        cancelLayout.addWidget(self.cancel_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        # Thread to perform pars tuberalis cells segmentation
        self.water_thread = QThread()

        self.worker = WatershedWorker(parent.input_folder_path, parent.output_folder_path, parent.watershed_folder_out, parent.resolution)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.water_thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.water_thread.finished.connect(self.worker.deleteLater)

        self.water_thread.started.connect(self.worker.start_splitting)
        self.water_thread.start()

    def close_on_success(self):
        self.water_thread.quit()
        self.water_thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_watershed(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.water_thread.quit()
        self.water_thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_watershed()
        event.ignore()


class WatershedWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_folder_path, main_output_folder_path, watershed_output_folder_path, resolution):

        super().__init__()

        self.canceled = Event()

        self.input_folder_path = input_folder_path
        self.watershed_folder_path = watershed_output_folder_path
        self.file_output_path = main_output_folder_path + "\\" + "watershed_cell_counting_results.txt"

        self.resolution = resolution


    def start_splitting(self):
        #print('Thread ID: %s' % int(QThread.currentThreadId()))
        watershed_stats = watershed_split_memeff(self.input_folder_path, self.watershed_folder_path, self.resolution,
                                                 canceled=self.canceled, progress_status=self.progress_status)

        if not self.canceled.is_set():
            num_labels, max_label, labels_list = watershed_stats
            write_counting_stats_to_txt(self.file_output_path, num_labels, labels=labels_list)

            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()


    def stop(self):
        self.canceled.set()


class CountLabelsProgressDialog(QDialog):
    """Progress Dialog that starts a new Thread for counting the number of labels in a labeled image stack and allows to cancel it."""
    def __init__(self, parent: WatershedWidget):
        super().__init__(parent)

        self.setWindowTitle("Count Cell Labels")

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(page_layout)

        #Progress Bar
        pBarLayout = QVBoxLayout()
        self.progBar = QProgressBar()
        self.progBar.setRange(0,100)
        self.progBarText = QLabel("Start Counting ...")

        pBarLayout.addWidget(self.progBar)
        pBarLayout.addWidget(self.progBarText)
        page_layout.addLayout(pBarLayout)

        #Cancel Button
        cancelLayout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_counting)

        cancelLayout.addWidget(self.cancel_button)
        page_layout.addLayout(cancelLayout)

        #Worker Thread
        self.thread = QThread()

        self.worker = CountLabelsWorker(parent.input_folder_path, parent.output_folder_path)

        #print('Main thread ID: %s' % int(QThread.currentThreadId()))
        self.worker.moveToThread(self.thread)

        self.worker.progress_status.connect(self.update_progress_bar)
        self.worker.finished.connect(self.close_on_success)

        self.thread.finished.connect(self.worker.deleteLater)

        self.thread.started.connect(self.worker.startCounting)
        self.thread.start()

    def close_on_success(self):
        self.thread.quit()
        self.thread.wait()
        self.accept()

    def update_progress_bar(self, status):
        self.progBar.setValue(status[1])
        if status[0] != "":
            self.progBarText.setText(status[0])

    def cancel_counting(self):
        self.progBarText.setText("Canceling...")
        self.progBarText.repaint()
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.reject()

    def closeEvent(self, event):
        self.cancel_counting()
        event.ignore()


class CountLabelsWorker(QObject):

    progress_status = pyqtSignal(list)  # ['Status Text', status percentage between 0-100]

    finished = pyqtSignal()

    def __init__(self, input_folder_path, output_folder_path):

        super().__init__()

        self.canceled = Event()

        self.input_folder_path = input_folder_path
        self.file_output_path = output_folder_path + "\\" + "watershed_cell_counting_results.txt"


    def startCounting(self):

        label_stats = count_labels_3D_memeff(self.input_folder_path, return_labels=True, canceled=self.canceled, progress_status=self.progress_status)

        if not self.canceled.is_set():
            progress = 95
            self.progress_status.emit(["Writing results to txt...", progress])

            num_labels, labels = label_stats
            write_counting_stats_to_txt(self.file_output_path, num_labels, labels=labels)

            progress = 100
            self.progress_status.emit(["Finished...", progress])

        self.finished.emit()

    def stop(self):
        self.canceled.set()
