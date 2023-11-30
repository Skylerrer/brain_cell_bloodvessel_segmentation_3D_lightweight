"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements settings tab for the GUI
# Here the resolution has to be specified. It will be passed on to the ParsSegPage, VesselWallPage, FilledVesselPage
########################################################################################################################

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
    QDoubleSpinBox,
    QCheckBox
)


maximalNeededSizePolicy = QSizePolicy()
maximalNeededSizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
maximalNeededSizePolicy.setVerticalPolicy(QSizePolicy.Policy.Maximum)


class SettingsPage(QWidget):

    def __init__(self, parent_main_window):
        super().__init__(parent_main_window)

        # Reference to the main_window of the GUI
        # The main window passes the resolution on to the other GUI parts
        self.parent_main_window = parent_main_window

        largerFont = self.font()
        largerFont.setPointSize(14)

        #Settings Page Layout
        pageLayout = QVBoxLayout()
        self.setLayout(pageLayout)

        #Explanation Text
        exLayout = QHBoxLayout()
        exText = QLabel("To continue, please enter your image resolution:")
        exText.setFont(largerFont)
        exLayout.addWidget(exText)
        exLayout.setContentsMargins(0,50,0,0)
        pageLayout.addLayout(exLayout)

        # Anisotropic Resolution Input
        resolutionLayout = QGridLayout()
        resolutionLayout.setContentsMargins(0,0,0,0)
        pageLayout.addLayout(resolutionLayout)

        x = QLabel("X")
        y = QLabel("Y")
        z = QLabel("Z")

        self.x_res = 0.0
        self.y_res = 0.0
        self.z_res = 0.0

        #x.setSizePolicy(p)
        #y.setSizePolicy(p)
        #z.setSizePolicy(p)
        resolutionLayout.addWidget(x, 0, 1)
        resolutionLayout.addWidget(y, 0, 2)
        resolutionLayout.addWidget(z, 0, 3)

        #axisLabelLayout.addStretch(5)

        voxelSizeLabel = QLabel("VoxelSize [" + u"\u03bc" +"m]:")
        voxelSizeLabel.setSizePolicy(maximalNeededSizePolicy)
        resolutionLayout.addWidget(voxelSizeLabel, 1, 0)

        # SpinBox to insert resolution of X direction
        self.xSpinBox = QDoubleSpinBox()
        self.xSpinBox.setMinimum(0)
        self.xSpinBox.setMaximum(20)
        self.xSpinBox.setDecimals(5)
        self.xSpinBox.setSingleStep(0.01)

        # SpinBox to insert resolution of Y direction
        self.ySpinBox = QDoubleSpinBox()
        self.ySpinBox.setMinimum(0)
        self.ySpinBox.setMaximum(20)
        self.ySpinBox.setDecimals(5)
        self.ySpinBox.setSingleStep(0.01)

        # SpinBox to insert resolution of Z direction
        self.zSpinBox = QDoubleSpinBox()
        self.zSpinBox.setMinimum(0)
        self.zSpinBox.setMaximum(20)
        self.zSpinBox.setDecimals(5)
        self.zSpinBox.setSingleStep(0.01)

        #Button that sets the resolution and calls the method that passes the resolution on to the other GUI parts
        self.set_res_button = QPushButton("Set")
        self.set_res_button.clicked.connect(self.after_resolution_inserted)

        resolutionLayout.addWidget(self.xSpinBox, 1, 1)
        resolutionLayout.addWidget(self.ySpinBox, 1, 2)
        resolutionLayout.addWidget(self.zSpinBox, 1, 3)
        resolutionLayout.addWidget(self.set_res_button, 1, 4)

        #Hide main window settings
        hide_window_layout = QHBoxLayout()
        hide_window_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hide_window_layout.setContentsMargins(0, 10, 0, 0)


        self.hide_main_window_checkbox = QCheckBox()
        self.hide_main_window_checkbox.setChecked(self.parent_main_window.hide_main_window_when_processing)
        self.hide_main_window_checkbox.stateChanged.connect(self.hide_main_window_val_change)
        self.hide_main_window_checkbox.setToolTip("<p>If this is checked, then the main window gets hidden when long "
                                                  "computations are executed in another thread. Instead only a small "
                                                  "dialog window is shown.</p>"
                                                  "<p>If this is unchecked, then the main window keeps being visible "
                                                  "in addition to the dialog window. The main window cannot be moved "
                                                  "while the progress dialog is open!</p>")
        hide_window_layout.addWidget(self.hide_main_window_checkbox)
        hide_window_layout.addWidget(QLabel("Hide the main window during performing long computations"))

        pageLayout.addLayout(hide_window_layout)
        pageLayout.addStretch(0)


    def after_resolution_inserted(self):

        self.x_res = self.xSpinBox.value()
        self.y_res = self.ySpinBox.value()
        self.z_res = self.zSpinBox.value()

        if self.resolutionComplete():

            self.set_res_button.setEnabled(False)
            self.xSpinBox.setEnabled(False)
            self.ySpinBox.setEnabled(False)
            self.zSpinBox.setEnabled(False)

            self.parent_main_window.enable_main_menu()
            self.parent_main_window.pass_on_resolution(self.getResolution())
            self.parent_main_window.parSegPage.initialize_advanced_parameters()
            self.parent_main_window.vesselWallPage.initialize_vesseg_parameters()

    def resolutionComplete(self):
        if self.x_res > 0.0 and self.y_res > 0.0 and self.z_res > 0.0:
            return True
        return False

    def getResolution(self):
        return np.asarray([self.z_res, self.y_res, self.x_res], dtype=np.float32)

    def hide_main_window_val_change(self, state):
        if state == 0:
            self.parent_main_window.hide_main_window_when_processing = False
        else:
            self.parent_main_window.hide_main_window_when_processing = True
