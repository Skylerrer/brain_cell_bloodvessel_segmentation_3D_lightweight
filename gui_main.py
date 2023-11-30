"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the main method of the GUI. Execute this script to start the GUI for pars tuberalis cell and
# blood vessel segmentation + quantification.
########################################################################################################################

import sys
import traceback

from PyQt6.QtWidgets import QApplication

from trpseg.GUI.main_window import MainWindow


def main():

    try:
        app = QApplication(sys.argv)

        window = MainWindow()
        window.show()

        app.exec()

    except:
        traceback.print_exc()


if __name__ == "__main__":
    main()
