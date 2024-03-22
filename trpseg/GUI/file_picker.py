"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the File Picker Widgets for the GUI
# This widget can be used to specify a filter that selects only specific files
########################################################################################################################

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QDialog,
    QFileDialog
)

from trpseg.trpseg_util.utility import get_file_list_from_directory_filter

class FilePicker(QDialog):
    """FilePicker base class"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("File Picker")
        self.input_file_paths = None
        self.filter_text = None
        self.file_ending_filter_text = None
        self.input_folder_path = ""

        #Main Layout
        page_layout = QVBoxLayout()
        #page_layout.setContentsMargins(0,0,0,0)
        self.setLayout(page_layout)

        #Filter Layout
        filter_layout = QVBoxLayout()
        filter_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        filter_label = QLabel("Filter:")
        self.filter_input_line = QLineEdit()
        self.filter_input_line.setEnabled(True)

        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input_line)

        self.filter_input_line.editingFinished.connect(self.set_filter)

        self.filter_input_line.setToolTip("<p>You can specify a string that needs to be contained within the file names.</p>"
                                           "<p>Example: C00, C01, _c4_</p>")

        #File Ending Layout

        file_ending_layout = QVBoxLayout()
        file_ending_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        file_ending_label = QLabel("File Ending:")
        self.file_ending_input_line = QLineEdit()
        self.file_ending_input_line.setEnabled(True)

        file_ending_layout.addWidget(file_ending_label)
        file_ending_layout.addWidget(self.file_ending_input_line)

        self.file_ending_input_line.editingFinished.connect(self.set_file_ending_filter)

        self.file_ending_input_line.setToolTip(
            "<p>You can specify a the file extension you are interested in. It does not matter whether you use a . ('dot') or not.</p>"
            "<p>Example: .tif , .png, tiff</p>")

        #Directory Layout
        dir_layout = QVBoxLayout()
        dir_layout_sub = QHBoxLayout()
        input_dir_label = QLabel("Choose Input Directory:")

        self.input_dir_line = QLineEdit()
        self.input_dir_line.setEnabled(False)
        # input_dir_line.setMinimumSize(300,10)

        self.input_dir_button = QPushButton("Select Directory ...")
        self.input_dir_button.setToolTip("<p>Select the folder in which the image stack slices that "
                                         "you want to process are stored.</p>")

        self.input_dir_button.clicked.connect(self.chooseDir)

        dir_layout_sub.addWidget(self.input_dir_line)
        dir_layout_sub.addWidget(self.input_dir_button)

        dir_layout.addWidget(input_dir_label)
        dir_layout.addLayout(dir_layout_sub)

        okay_button = QPushButton("Okay")
        okay_button.clicked.connect(self.finish_file_picker)

        page_layout.addLayout(filter_layout)
        page_layout.addLayout(file_ending_layout)
        page_layout.addLayout(dir_layout)
        page_layout.addWidget(okay_button)


    def finish_file_picker(self):

        self.input_file_paths = get_file_list_from_directory_filter(self.input_folder_path,
                                                                    self.file_ending_filter_text, self.filter_text)
        if self.input_file_paths is not None:
            self.accept()
        else:
            self.reject()

    def chooseDir(self):
        home = os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", home)
        if directory != '':
            self.input_folder_path = directory
            self.input_dir_line.setText(directory)


    def set_filter(self):
        self.filter_text = self.filter_input_line.text()

    def set_file_ending_filter(self):
        self.file_ending_filter_text = self.file_ending_input_line.text()
        