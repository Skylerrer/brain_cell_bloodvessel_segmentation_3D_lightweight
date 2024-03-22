"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements the segmentation pipeline to segment the inside of the brain.
# It is used to make it possible to compute the distance for every pixel inside of the brain to the outer brain boundary.
########################################################################################################################

import os
from threading import Event

from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.measure import label

from trpseg.trpseg_util.utility import read_image, save_image, get_file_list_from_directory


def segment_brain_region(input_file_paths, output_folder_path, sigma, threshold, canceled=Event()):
    """
    Segment the brain region in images. Simply a global thresholding on the smoothed images is used to get the segmentation result.

    Parameters
    ----------
    input_file_paths : List[str], List[pathlib.Path]
        The paths to the files containing the 3D image stack slices.
    output_folder_path: str, pathlib.Path
        The path to the folder where the output images are stored
    sigma : int, float
        The sigma (standard deviation) of the gaussian smoothing filter.
    threshold : int, float
        The global threshold for brain tissue detection. Pixels in the smoothed image with a value greater than threshold
        are set to 255, others to 0
    canceled : threading.Event, optional
        An event object that allows to cancel the process

    Notes
    ----------
    This algorithm assumes that the upper left pixel in all 2D image slices is background and not brain tissue!
    The basic idea is:
    1. perform 2D gaussian smoothing with high sigma (=standard deviation)
    2. perform global thresholding. Pixels with value greater than the threshold will be assumed to be brain tissue
       and get value 1; other pixels get value 0
    3. set 0 pixels to 1 if they are inside of the brain (this adds dark regions inside of the brain to the brain region segmentation).
       Every 0 pixel that is not connected to the 0 region to which the upper left pixel belongs gets assigned value 1
    """

    input_files_list = input_file_paths

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    i = 0
    for file in input_files_list:

        if canceled.is_set():
            return

        print("Segment Brain Region from Image ", i)
        i += 1

        img = read_image(file)

        smooth_img = ndimage.gaussian_filter(img, sigma=sigma, truncate=3)
        smooth_img = smooth_img.astype(np.uint16)

        th_img = (smooth_img > threshold)*1

        labels = label(th_img, connectivity=2, background=-1)  # background=-1 important. So the outer dark region has different labels than dark regions inside brain (e.g. ventricle)

        outer_label = labels[0][0]  # IMPORTANT: it is currently assumed that the pixel in the upper left corner labels[0][0] belongs to th outside of the brain
        fg = (labels != outer_label)*1

        fg = fg*255

        out_name = Path(file.name)
        out_name = out_name.with_suffix(".png")
        output_path = output_folder / out_name

        save_image(fg, output_path, np.uint8)
