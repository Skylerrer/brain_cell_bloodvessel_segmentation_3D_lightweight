"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the Threshold Picker Widgets for the GUI
# These widgets are able to show images together with a red overlay representing the thresholding result on the image
########################################################################################################################

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QDialog,
    QSizePolicy,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QSlider
)
from scipy import ndimage
from skimage import exposure
import numpy as np
from skimage.morphology import disk
from skimage.filters import median

from trpseg.trpseg_util.utility import read_image


class ThresholdPicker(QDialog):
    """ThresholdPicker base class"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Threshold Picker")
        self.img_paths = None
        self.current_img_idx = -1
        self.current_img = None
        self.current_displayed_img = None
        self.current_th_image = None
        self.alpha = 0.5 # alpha of the red threshold overlay
        self.chosen_threshold = -1

        self.current_max_intensity = 65535
        self.displayed_max_intensity = 65535

        #Main Layout
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0,0,0,0)
        self.setLayout(page_layout)
        #View Image
        self.img_placeholder = QLabel()
        page_layout.addWidget(self.img_placeholder)
        #self.img_placeholder.setMinimumSize(512,512)
        #self.img_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        parameter_layout = QHBoxLayout()
        #allow to choose which image to show
        choose_img_layout = QHBoxLayout()
        choose_img_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        choose_label = QLabel("Displayed Image:")
        self.choose_spin = QSpinBox()
        self.choose_spin.setMinimum(0)
        self.choose_spin.setMaximum(0)
        self.choose_spin.setSingleStep(1)
        self.choose_spin.valueChanged.connect(self.change_current_orig_image)

        choose_img_layout.addWidget(choose_label)
        choose_img_layout.addWidget(self.choose_spin)

        parameter_layout.addLayout(choose_img_layout)

        page_layout.addLayout(parameter_layout)


        #then allow brightness change
        brightness_layout = QHBoxLayout()
        brightness_layout.setContentsMargins(20,-1,-1,-1)
        brightness_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        b_label = QLabel("Max Intensity:")
        b_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.bval_label = QLabel("65535")
        self.bval_label.setFixedWidth(30)

        self.b_slider = QSlider(Qt.Orientation.Horizontal)
        self.b_slider.setMaximumWidth(100)
        self.b_slider.setMinimum(1)
        self.b_slider.setMaximum(self.displayed_max_intensity)
        self.b_slider.setSingleStep(1)
        self.b_slider.setTracking(False)
        self.b_slider.valueChanged.connect(self.change_max_intensity)

        brightness_layout.addWidget(b_label, Qt.AlignmentFlag.AlignLeft)
        brightness_layout.addWidget(self.bval_label, Qt.AlignmentFlag.AlignLeft)
        brightness_layout.addWidget(self.b_slider, Qt.AlignmentFlag.AlignLeft)

        parameter_layout.addLayout(brightness_layout)

        #then allow to enter Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.setContentsMargins(-1,10,-1,-1)
        threshold_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        th_label = QLabel("Set Threshold:")
        th_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.th_spin = QSpinBox()
        self.th_spin.setMinimum(0)
        self.th_spin.setMaximum(65535)
        self.th_spin.setSingleStep(10)
        self.th_spin.valueChanged.connect(self.update_thresholded_image)

        th_reset = QPushButton("Reset")
        th_reset.clicked.connect(self.reset_th)

        th_done_button = QPushButton("Done")
        th_done_button.clicked.connect(self.finish_th)

        threshold_layout.addWidget(th_label)
        threshold_layout.addWidget(self.th_spin)
        threshold_layout.addWidget(th_reset)
        threshold_layout.addWidget(th_done_button)

        page_layout.addLayout(threshold_layout)


    def finish_th(self):
        pass
        if self.chosen_threshold != -1:
            self.accept()
        else:
            self.reject()

    def reset_th(self):
        self.update_displayed_image()

    def update_displayed_image(self, th_img=False):
        if th_img:
            bytesPerLine = 3 * self.current_displayed_img.shape[1] #BytesPerLine is number of pixels per row times the number of bytes per pixel (here RGB -> 3Byte)
            qimg = QImage(self.current_th_image, self.current_img.shape[1], self.current_img.shape[0], bytesPerLine, QImage.Format.Format_RGB888)
        else:
            bytesPerLine = self.current_displayed_img.shape[1]
            qimg = QImage(self.current_displayed_img, self.current_displayed_img.shape[1], self.current_displayed_img.shape[0], bytesPerLine, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.img_placeholder.setPixmap(pixmap.scaled(768, 768, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def update_thresholded_image(self):
        th_value = self.th_spin.value()
        self.chosen_threshold = th_value

        th_mask = (self.current_img > th_value)
        beta = 1.0 - self.alpha

        # Overlay red color onto th_mask pixels
        red_overlay_val = int(self.alpha * 255)
        img = self.current_displayed_img.copy()
        overlay_these_vals = (img[th_mask] * beta).astype(np.uint8)
        red_overlay = overlay_these_vals + red_overlay_val
        red_overlay = red_overlay.astype(np.uint8)

        img[th_mask] = overlay_these_vals
        red_overlay_img = img.copy()
        red_overlay_img[th_mask] = red_overlay

        #Displayed image from 8bit grayscale to (3*8bit) rgb
        th_image = np.stack((red_overlay_img, img, img), axis=-1)
        self.current_th_image = th_image
        self.update_displayed_image(th_img=True)


    def change_max_intensity(self, value):
        self.displayed_max_intensity = value
        self.current_displayed_img = self.displayed_img_from_current()
        self.update_displayed_image()
        self.bval_label.setText(str(value))


    def change_current_orig_image(self,value):

        self.current_img = read_image(self.img_paths[value])

        #Set new max intensity
        self.current_max_intensity = np.max(self.current_img)
        self.displayed_max_intensity = self.current_max_intensity

        # Set Brightness Slider
        self.b_slider.setMaximum(self.current_max_intensity - 1)
        self.b_slider.setValue(self.current_max_intensity - 1)
        self.bval_label.setText(str(self.current_max_intensity - 1))

        self.current_displayed_img = self.displayed_img_from_current()
        self.update_displayed_image()

    def displayed_img_from_current(self):

        img = self.current_img
        if self.current_max_intensity != self.displayed_max_intensity:
            img = exposure.rescale_intensity(img, in_range=(0,self.displayed_max_intensity), out_range=(0,self.current_max_intensity)).astype(np.uint16)

        img = exposure.rescale_intensity(img, in_range=(0, self.current_max_intensity), out_range=(0, 255)).astype(np.uint8)

        return img

    def set_image_paths(self, paths):
        self.img_paths = paths
        self.numImages = len(paths)

        self.current_img_idx = int(self.numImages//2)

        if self.numImages > 0:
            # Set choose image spinbox
            self.choose_spin.setMaximum(self.numImages - 1)
            self.choose_spin.setValue(self.current_img_idx)

            self.current_img = read_image(self.img_paths[self.current_img_idx])

            self.current_max_intensity = np.max(self.current_img)
            self.displayed_max_intensity = self.current_max_intensity

            #Set Brightness Slider
            self.b_slider.setMaximum(self.current_max_intensity-1)
            self.b_slider.setValue(self.current_max_intensity-1)
            self.bval_label.setText(str(self.current_max_intensity-1))

            #Create 8bit image for display
            self.current_displayed_img = self.displayed_img_from_current()

            self.update_displayed_image()


class ThresholdPickerGaussianSmoothed(ThresholdPicker):
    """ThresholdPicker that visualizes thesholding result on gaussian smoothed image."""
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.smoothed_img = None


    def update_thresholded_image(self):
        th_value = self.th_spin.value()
        self.chosen_threshold = th_value

        if self.smoothed_img is None:
            self.smoothed_img = ndimage.gaussian_filter(self.current_img, self.sigma, truncate=3)

        th_mask = (self.smoothed_img > th_value)
        betha = 1.0 - self.alpha

        # Overlay red color onto th_mask pixels
        red_overlay_val = int(self.alpha * 255)
        img = self.current_displayed_img.copy()

        overlay_these_vals = (img[th_mask] * betha).astype(np.uint8)
        red_overlay = overlay_these_vals + red_overlay_val
        red_overlay = red_overlay.astype(np.uint8)

        img[th_mask] = overlay_these_vals
        red_overlay_img = img.copy()
        red_overlay_img[th_mask] = red_overlay

        #Displayed image from 8bit grayscale to (3*8bit) rgb
        th_image = np.stack((red_overlay_img, img, img), axis=-1)
        self.current_th_image = th_image
        self.update_displayed_image(th_img=True)

    def change_current_orig_image(self, value):
        super().change_current_orig_image(value)
        self.smoothed_img = None


class ThresholdPickerMedianSmoothed(ThresholdPicker):
    """ThresholdPicker that visualizes thesholding result on median smoothed image.
    This is relatively slow for large median_radius.
    """
    def __init__(self, median_radius):
        super().__init__()
        self.radius = median_radius
        self.footprint = disk(median_radius)
        self.median_img = None


    def update_thresholded_image(self):
        th_value = self.th_spin.value()
        self.chosen_threshold = th_value

        if self.median_img is None:
            self.median_img = median(self.current_img, self.footprint)

        th_mask = (self.median_img > th_value)
        betha = 1.0 - self.alpha

        # Overlay red color onto th_mask pixels
        red_overlay_val = int(self.alpha * 255)
        img = self.current_displayed_img.copy()

        overlay_these_vals = (img[th_mask] * betha).astype(np.uint8)
        red_overlay = overlay_these_vals + red_overlay_val
        red_overlay = red_overlay.astype(np.uint8)

        img[th_mask] = overlay_these_vals
        red_overlay_img = img.copy()
        red_overlay_img[th_mask] = red_overlay

        #Displayed image from 8bit grayscale to (3*8bit) rgb
        th_image = np.stack((red_overlay_img, img, img), axis=-1)
        self.current_th_image = th_image
        self.update_displayed_image(th_img=True)

    def change_current_orig_image(self, value):
        super().change_current_orig_image(value)
        self.median_img = None
