"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a method to perform brightness normalization across the z axis of an image stack.
# Brightness can change across different 2D image slices and adjustment could be necessary.
########################################################################################################################

import os
from threading import Event

from pathlib import Path
import numpy as np

from trpseg.trpseg_util.utility import read_image, save_image, get_file_list_from_directory


def z_stack_tissue_mean_normalization(input_folder_path_list, output_folder_path, min_tissue, max_tissue, tissue_mean=None,
                                      canceled=Event(), progress_status=None):
    """Normalize the brain tissue mean of all images in input_folder to the provided or automatically determined wanted tissue mean.

    Parameters
    ----------
    input_folder_path_list: List[str], List[pathlib.Path]
        List of file paths to the image slices to be normalized.
    output_folder_path: str, pathlib.Path
        The path to the folder where the normalized output images are stored
    min_tissue : int
        The minimal pixel intensity value considered for normalization.
        The value should ideally be chosen as the darkest brain tissue pixel intensity
        that is higher than the brightest pixel of the ventricle and the background.
    max_tissue : int
        The maximal pixel intensity value considered for normalization.
        This value should ideally be chosen as the brightest brain tissue pixel intensity
        excluding the marked pars tuberalis cells / tanycytes and blood vessels
    tissue_mean : int, float, optional
        The wanted tissue mean to which all images are normalized.
        If None, then the average tissue mean over the z_stack is automatically approximated.
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status


    Notes
    -------
    The normalization factor is determined by only looking at the brain tissue pixels. The brain tissue pixels are
    pixels with intensity values in the interval (min_tissue, max_tissue).

    The images are then multiplied by the factor (wanted_tissue_mean/current_image_tissue_mean).

    IMPORTANT: If you cannot find a good interval (min_tissue, max_tissue) that excludes the background,
    the ventricle and the high signal pixels (pars tuberalis cells/ tanycytes / blood vessels) then this method
    can not be used!
    """

    progress = 0

    input_files_list = input_folder_path_list

    numFiles = len(input_files_list)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    if tissue_mean is None:

        if progress_status is not None:
            progress_status.emit(["Determining tissue mean...", progress])

        tissue_mean, all_means, consider_for_normalization = auto_detect_tissue_mean(input_files_list, min_tissue, max_tissue)
        print("Detected Tissue Mean: " + str(tissue_mean))

    if canceled.is_set():
        return

    if progress_status is not None:
        progress_status.emit(["Normalizing images...", progress])

    for i in range(0, numFiles):

        if canceled.is_set():
            return

        progress = int((i/numFiles)*100)

        img = read_image(input_files_list[i])

        if progress_status is not None:
            progress_status.emit(["", progress])

        # be sure that there are not only very few pixels of tissue that could mess up the result
        if consider_for_normalization[i]:
            current_mean = all_means[i]

            img = img.astype(np.float64)
            img = img * (tissue_mean/current_mean)
            img = np.clip(img, 0, 65535)
            img = img.astype(np.uint16)

        else:
            print("No normalization for image: ", i)

        out_name = input_files_list[i].name
        out_path = output_folder / out_name
        save_image(img, out_path, np.uint16)

    if progress_status is not None:
        progress_status.emit(["Finished...", 100])


def auto_detect_tissue_mean(input_file_list, min_tissue, max_tissue):
    """Automatically determine the global tissue mean of an image stack. Additionally, return the mean of each slice of the image stack.

    Parameters
    ----------
    input_file_list : list of pathlib.Path
        The paths to the image stack slices.
    min_tissue : int
        The minimal pixel intensity value considered for determining the tissue mean.
    max_tissue : int
        The maximal pixel intensity value considered for determining the tissue mean.

    Returns
    -------
    global_tissue_mean : float
        The approximated global tissue mean
    all_means : list of float
        The tissue mean for all z-slices.
    consider_for_normalization: list of bool
        Specifies for every slice whether it has more than 100000 tissue pixels. Only then the slice is considered for normalization.

    Notes
    -------
    This algorithm looks at the image slices of an image stack and determines the mean brain tissue intensity in the stack.
    """

    num_files = len(input_file_list)

    num_taken_means = 0
    global_tissue_mean = 0
    all_means = []
    consider_for_normalization = []
    i = 0

    while i < num_files:
        img = read_image(input_file_list[i])

        th1 = img > min_tissue  # e.g. 380
        th2 = img < max_tissue  # e.g. 900
        th = np.logical_and(th1, th2)
        tissue_region = img[th]

        current_mean = np.mean(tissue_region)
        all_means.append(current_mean)

        #be sure that there are not only very few tissue pixels that could mess up the result
        if len(tissue_region) >= 100000:
            global_tissue_mean += current_mean
            num_taken_means += 1
            consider_for_normalization.append(True)
        else:
            consider_for_normalization.append(False)

        i += 1

    if num_taken_means == 0:
        raise RuntimeError("You need at least one image slice with more than 100000 pixels that belong to tissue according to your provided tissue min and max!")

    if num_files == 0:
        raise RuntimeError("You have to provide a file list with at least one image slice!")

    global_tissue_mean = global_tissue_mean / num_taken_means

    return global_tissue_mean, all_means, consider_for_normalization


#Normalize according to maximum pixel intensity. If maximum is greater than wanted_max then multiply by wanted_max/found_max
#Does only attentuate too high signals. Does not increase too low signals. Normally we dont want to change too much of the signal.
#And in our case we just wanted to attentuate too strong signal at the start of the image stack
def max_normalization(input_folder, output_folder, wanted_max, isTif=True, channel_one=False):
    """Normalize the image slices according to the 99.75% quantile.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the image slices to be normalized.
    output_folder_path: str, pathlib.Path
        The path to the folder where the normalized output images are stored
    wanted_max : int
        The wanted 99.75% quantile value.
    isTif : bool, optional
        Determines, which input images inside input_folder_path are considered.
        True, means that images have file type .tif
        False, means that images have file type .png
        None, means that all file types are considered
    channel_one : bool, optional
        Determines, which channel from the input images to use.
        True, means that only images containing the string "_C01_" are considered.
        False, means that only images containing the string "_C00_" are considered.
        None, means that all channels are considered.

    Notes
    -------
    Some image slices especially at the beginning and end of some image stacks had way higher maximal intensity values
    than the other image slices.
    This normalization method can be used to attentuate too high signals.
    """

    input_files_list = get_file_list_from_directory(input_folder, isTif=True, channel_one=channel_one)

    numFiles = len(input_files_list)

    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, numFiles):
        img = read_image(input_files_list[i])

        # TODO maybe higher quantile or only upper quantile of Pixels that are signal Pixels -> intensity > 900 or similar
        upperMax = np.quantile(img, 0.9975)  #0.25% of 2048*2048Pixel -> border to brightest 10486Pixel

        if upperMax > wanted_max:
            img = img.astype(np.float64)
            factor = wanted_max / upperMax
            print("Factor: ", factor)
            img = img * factor
            img = np.clip(img, 0, 65535)
            img = img.astype(np.uint16)

            upperMax = np.quantile(img, 0.9975)
            lowerMin = np.quantile(img, 0.0025)
            #print("Slice: " + str(i) + " | new Upper: " + str(upperMax) + " | new Lower: " + str(lowerMin))

        else:
            print("No signal normalization for image: ", i)

        out_name = input_files_list[i].name
        out_path = output_folder / out_name
        save_image(img, out_path, np.uint16)


#First normalize image stack according to mean and afterwards normalize according to max_normalization (attentuate too high signals). See also above.
def mean_and_max_normalize_z_stack_test(input_folder, output_folder, tissue_mean=580, tissue_min=380, tissue_max=900, signalQuantileMax=2000, maxSignal=2000, channel_one=False):

    input_files_list = get_file_list_from_directory(input_folder, isTif=True, channel_one=channel_one)

    numFiles = len(input_files_list)

    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, numFiles):
        img = read_image(input_files_list[i])
        th1 = img > tissue_min
        th2 = img < tissue_max
        th = np.logical_and(th1,th2)
        tissue_region = img[th]
        img = img.astype(np.float64)

        if len(tissue_region) > 100000:
            current_mean = np.mean(tissue_region)
            factor = tissue_mean/current_mean
            #print("Image: ", i, "| Factor: ", factor)
            img = img * (tissue_mean/current_mean)
            img = np.clip(img, 0, 65535)

        else:
            print("No mean normalization possible for image: ", i)

        #TODO maybe higher quantile or only upper quantile of Pixels that are signal Pixels -> intensity > 900 or similar
        upperMax = np.quantile(img, 0.9975)  # 0.25% of 2048*2048Pixel -> border to brightest 10486Pixel

        if upperMax > maxSignal:
            factor = signalQuantileMax / upperMax
            print("Max Normalization Factor: ", factor)
            img = img * factor
            img = np.clip(img, 0, 65535)
            img = img.astype(np.uint16)


        else:
            print("No max normalization for image: ", i)

        out_name = input_files_list[i].name
        out_path = output_folder / out_name
        save_image(img, out_path, np.uint16)
