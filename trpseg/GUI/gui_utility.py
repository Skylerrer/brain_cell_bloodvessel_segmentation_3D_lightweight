"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements utility functions for the GUI
########################################################################################################################

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox
)

from trpseg.GUI.file_picker import FilePicker


class InputOutputWidgetSingleInput(QWidget):

    def __init__(self):
        super().__init__()

        in_out_layout = QVBoxLayout()
        in_out_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(in_out_layout)

        # 1. Choose Input Directory--------------------------------------------------------------------------------------
        input_layout = QVBoxLayout()
        input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        input_layout.setContentsMargins(0, 0, 0, 0)

        dir_layout = QHBoxLayout()

        input_dir_label = QLabel("Choose Input Directory:")

        self.input_dir_line = QLineEdit()
        self.input_dir_line.setEnabled(False)
        # input_dir_line.setMinimumSize(300,10)

        self.input_dir_button = QPushButton("Select Directory ...")
        self.input_dir_button.setToolTip("<p>Select the folder in which the image stack slices that "
                                         "you want to process are stored.</p>")

        dir_layout.addWidget(input_dir_label)
        dir_layout.addWidget(self.input_dir_line)
        dir_layout.addWidget(self.input_dir_button)

        # Display numFiles
        display_filenum = QHBoxLayout()
        display_filenum.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.numFiles_label = QLabel("Number of Images:")
        # self.numFiles_label.setMinimumSize(250,1)
        # self.numFiles_label.setStyleSheet(stylesheets.help)
        display_filenum.addWidget(self.numFiles_label)
        # display_filenum.addStretch(0)

        input_layout.addLayout(dir_layout)
        input_layout.addLayout(display_filenum)
        # input_layout.addStretch(0)

        in_out_layout.addLayout(input_layout, Qt.AlignmentFlag.AlignTop)

        # 2.Choose Output Directory--------------------------------------------------------------------------------------
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        output_dir_label = QLabel("Choose Output Directory:")
        # output_dir_label.setStyleSheet(stylesheets.help)
        self.output_dir_line = QLineEdit()
        self.output_dir_line.setEnabled(False)
        # output_dir_line.setMinimumSize(300,10)

        # self.output_dir_line.setStyleSheet(stylesheets.help)
        self.output_dir_button = QPushButton("Select Directory ...")
        self.output_dir_button.setToolTip("<p>Select the folder where the results should be "
                                          "stored.</p>")
        # output_dir_button.setStyleSheet(stylesheets.help)
        output_layout.addWidget(output_dir_label)
        output_layout.addWidget(self.output_dir_line)
        output_layout.addWidget(self.output_dir_button)

        in_out_layout.addLayout(output_layout, Qt.AlignmentFlag.AlignTop)


class InputOutputWidgetTwoInputs(QWidget):

    def __init__(self):
        super().__init__()

        in_out_layout = QVBoxLayout()
        in_out_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(in_out_layout)

        # 1. Choose Input Directory--------------------------------------------------------------------------------------
        input_layout = QVBoxLayout()
        input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        input_layout.setContentsMargins(0, 0, 0, 0)

        #Input for blood files
        dir_layout1 = QHBoxLayout()

        input_dir_label_blood = QLabel("Choose Input Directory (Blood):")

        self.input_dir_line_blood = QLineEdit()
        self.input_dir_line_blood.setEnabled(False)
        # input_dir_line.setMinimumSize(300,10)

        self.input_dir_button_blood = QPushButton("Select Directory ...")
        self.input_dir_button_blood.setToolTip("<p>Select the folder in which the blood image stack slices that "
                                         "you want to process are stored.</p>")

        dir_layout1.addWidget(input_dir_label_blood)
        dir_layout1.addWidget(self.input_dir_line_blood)
        dir_layout1.addWidget(self.input_dir_button_blood)

        # Input for Pars files
        dir_layout2 = QHBoxLayout()

        input_dir_label_pars = QLabel("Choose Input Directory (Pars):")

        self.input_dir_line_pars = QLineEdit()
        self.input_dir_line_pars.setEnabled(False)
        # input_dir_line.setMinimumSize(300,10)

        self.input_dir_button_pars = QPushButton("Select Directory ...")
        self.input_dir_button_pars.setToolTip("<p>Select the folder in which the pars image stack slices that "
                                               "you want to process are stored.</p>")

        dir_layout2.addWidget(input_dir_label_pars)
        dir_layout2.addWidget(self.input_dir_line_pars)
        dir_layout2.addWidget(self.input_dir_button_pars)


        # Display numFiles
        display_filenum = QHBoxLayout()
        display_filenum.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.numFilesC00_label = QLabel("Number of Images (Blood):")
        # self.numFilesC00_label.setMinimumSize(250,1)
        self.numFilesC01_label = QLabel("Number of Images (Pars):")
        # self.numFilesC01_label.setMinimumSize(250, 1)
        # self.numFilesC00_label.setStyleSheet(stylesheets.help)
        # self.numFilesC01_label.setStyleSheet(stylesheets.help)
        display_filenum.addWidget(self.numFilesC00_label)
        display_filenum.addWidget(self.numFilesC01_label)
        # display_filenum.addStretch(0)

        input_layout.addLayout(dir_layout1)
        input_layout.addLayout(dir_layout2)
        input_layout.addLayout(display_filenum)
        # input_layout.addStretch(0)

        in_out_layout.addLayout(input_layout, Qt.AlignmentFlag.AlignTop)

        # 2.Choose Output Directory--------------------------------------------------------------------------------------
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        output_dir_label = QLabel("Choose Output Directory:")
        # output_dir_label.setStyleSheet(stylesheets.help)
        self.output_dir_line = QLineEdit()
        self.output_dir_line.setEnabled(False)
        # output_dir_line.setMinimumSize(300,10)

        # self.output_dir_line.setStyleSheet(stylesheets.help)
        self.output_dir_button = QPushButton("Select Directory ...")
        self.output_dir_button.setToolTip("<p>Select the folder where the results should be "
                                          "stored.</p>")
        # output_dir_button.setStyleSheet(stylesheets.help)
        output_layout.addWidget(output_dir_label)
        output_layout.addWidget(self.output_dir_line)
        output_layout.addWidget(self.output_dir_button)

        in_out_layout.addLayout(output_layout, Qt.AlignmentFlag.AlignTop)

class NormalizationWidget(QWidget):
    def __init__(self, parent_widget: QWidget):
        super().__init__()

        self.tissue_min = 0
        self.tissue_max = 0

        # Normalization Layout
        normalization_layout = QHBoxLayout()
        normalization_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # normalization_layout.setContentsMargins(10,20,10,10)
        parent_widget.setLayout(normalization_layout)

        # Min Spinbox
        normalization_min_layout = QHBoxLayout()
        normalization_label_min = QLabel("Min Tissue Value:")
        self.normalization_min_spin = QSpinBox()
        self.normalization_min_spin.setMinimum(0)
        self.normalization_min_spin.setMaximum(65535)
        self.normalization_min_spin.setSingleStep(10)
        self.normalization_min_spin.valueChanged.connect(self.set_tissue_min)

        self.normalization_min_spin.setToolTip("<p>Choose the minimal intensity of brain tissue that should be"
            " considered in the normalization factor estimation.</p><p>Choose this value so that the ventricle and the"
            " region outside of the brain are not included.</p>")

        self.normalization_min_button = QPushButton("Use ThPicker")
        self.normalization_min_button.setEnabled(False)
        # When you use this widget you have to connect a slot to the following signal
        #self.normalization_min_button.clicked.connect(...)

        normalization_min_layout.addWidget(normalization_label_min)
        normalization_min_layout.addWidget(self.normalization_min_spin)
        normalization_min_layout.addWidget(self.normalization_min_button)

        # Max Spinbox
        normalization_max_layout = QHBoxLayout()
        normalization_max_layout.setContentsMargins(10, 10, 50, 10)
        normalization_label_max = QLabel("Max Tissue Value:")
        self.normalization_max_spin = QSpinBox()
        self.normalization_max_spin.setMinimum(0)
        self.normalization_max_spin.setMaximum(65535)
        self.normalization_max_spin.setSingleStep(10)
        self.normalization_max_spin.valueChanged.connect(self.set_tissue_max)

        self.normalization_max_spin.setToolTip("<p>Choose the maximal intensity of brain tissue that should be "
            "considered in the normalization factor estimation.</p><p>Choose this value so that high intensities from"
            " fluorescently marked cells are above this threshold and other brain tissue is below.</p>")

        self.normalization_max_button = QPushButton("Use ThPicker")
        self.normalization_max_button.setEnabled(False)
        # When you use this widget you have to connect a slot to the following signal
        #self.normalization_max_button.clicked.connect(...)

        normalization_max_layout.addWidget(normalization_label_max)
        normalization_max_layout.addWidget(self.normalization_max_spin)
        normalization_max_layout.addWidget(self.normalization_max_button)

        # Normalize Button
        self.normalization_button = QPushButton("Normalize")
        self.normalization_button.setEnabled(False)
        # When you use this widget you have to connect a slot to the following signal
        #self.normalization_button.clicked.connect(...)

        normalization_layout.addLayout(normalization_min_layout)
        normalization_layout.addLayout(normalization_max_layout)
        normalization_layout.addWidget(self.normalization_button)

    def set_tissue_min(self, value):
        self.tissue_min = value

    def set_tissue_max(self, value):
        self.tissue_max = value


def show_error_message(parent, error_string):
    """Show a message box with a specified error string."""

    pressed_button = QMessageBox.critical(parent, "Error", error_string, buttons=QMessageBox.StandardButton.Discard,
                                          defaultButton=QMessageBox.StandardButton.Discard)
