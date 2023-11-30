"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the main window for the GUI
# It includes the page tabs: SettingsPage, ParsSegPage, VesselWallPage, FilledVesselPage
########################################################################################################################

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QTabBar,
    QWidget,
    QVBoxLayout,
    QStackedLayout
)

from trpseg.GUI import stylesheets
from trpseg.GUI.settings_page import SettingsPage
from trpseg.GUI.parseg_page import ParsSegPage
from trpseg.GUI.blood_vessel_wall_page import VesselWallPage
from trpseg.GUI.entire_blood_vessel_page import FilledVesselPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRPAnalyzer")

        #Set central widget that includes everything
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.main_layout = QVBoxLayout(self.centralWidget)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0,0,0,0)

        #Top Menu
        self.top_main_menu = TopMenu()
        self.main_layout.addWidget(self.top_main_menu)

        # Hide main window when processing
        self.hide_main_window_when_processing = True

        #Stacked Pages
        self.pagesLayout = QStackedLayout()
        self.settingsPage = SettingsPage(self)
        self.parSegPage = ParsSegPage(self)
        self.vesselWallPage = VesselWallPage(self)
        self.filledVesPage = FilledVesselPage(self)

        self.pagesLayout.addWidget(self.settingsPage)
        self.pagesLayout.addWidget(self.parSegPage)
        self.pagesLayout.addWidget(self.vesselWallPage)
        self.pagesLayout.addWidget(self.filledVesPage)

        self.main_layout.addLayout(self.pagesLayout)

        #Connect QTabBar with stacked pages
        self.top_main_menu.currentChanged.connect(self.pagesLayout.setCurrentIndex)

        #Close the window
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(exit_action)
        file_menu.addSeparator()

    def enable_main_menu(self):
        self.top_main_menu.setTabEnabled(1, True)
        self.top_main_menu.setTabEnabled(2, True)
        self.top_main_menu.setTabEnabled(3, True)

    def pass_on_resolution(self, resolution):
        self.parSegPage.set_resolution(resolution)
        self.vesselWallPage.set_resolution(resolution)
        self.filledVesPage.set_resolution(resolution)


class TopMenu(QTabBar):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(stylesheets.main_menu)
        self.addTab("Settings")
        self.addTab("Pars Tuberalis Cells \n Analysis")
        self.addTab("Blood Vessel Walls \n Analysis")
        self.addTab("Entire Blood Vessel \n Analysis")
        self.setCurrentIndex(0)
        self.setTabEnabled(1, False)
        self.setTabEnabled(2, False)
        self.setTabEnabled(3, False)
