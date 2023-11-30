"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a method to segment the 3rd ventricle.
# It asserts that in last image slice in the image stack there is a ventricle visible.
# If there is no ventricle there, then this algorithm does not work!
########################################################################################################################

import os

from pathlib import Path
import numpy as np
from skimage.filters import gaussian
from skimage.measure import label

from trpseg.trpseg_util.utility import read_image, save_image, count_image_boundary_coords,\
    get_file_list_from_directory, get_img_shape_from_filepath


def segment_ventricle_th(input_folder_path, output_folder_path, sigma, threshold, isTif=True, channel_one=False):
    """ Segment the ventricle by using thresholding and propagation of the ventricle location across z slices.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices from which the ventricle should be segmented.
    output_folder_path : str, pathlib.Path
        The path to the folder where the binary output images are stored.
    sigma : int, float
        The sigma (standard deviation) of the gaussian smoothing filter.
    threshold : int, float
        The global threshold used for distinguishing brain tissue from background area and ventricle area.
        It should be chosen as the minimal tissue pixel intensity value, but not lower than the highest ventricle pixel intensity value.
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
    ----------
    This algorithm asserts that in last image slice in the image stack there is a ventricle visible.
    If there is no ventricle there, then this algorithm does not work!
    Start at the last image in stack because there ventricle can normally be segmented really easy via global thresholding
    Then goes down the stack starting from this last image and also uses global thresholding for the other images
    However, only the possible ventricle regions that are in 6 neighborhood of ventricle in previous slice (z+1) are kept
    This allows to segment ventricle even if ventricle is splitted in two or more parts.
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    # input_files_list = input_files_list[0:333]  # select certain z coord files

    img_height, img_width = get_img_shape_from_filepath(input_files_list[0])

    current_img_idx = len(input_files_list) - 1
    last_img_idx = current_img_idx

    old_seg = []

    while current_img_idx >= 0:

        img_path = input_files_list[current_img_idx]
        img = read_image(img_path)
        img = gaussian(img, sigma=sigma, truncate=3)
        img = img * 65535
        img = img.astype(np.uint16)

        th_img = (img > threshold) * 1

        # Now the label_img has tissue as background (label 0) and outer_area and ventricle and unwanted holes with low intensity values as other labeled regions
        label_img = label(th_img, connectivity=2, background=1)

        # Could be optimized if necessary
        # remove label that has most pixels at image boundary (This is the outer dark area)
        labels = sorted(np.unique(label_img))
        labels = np.array(labels)
        label_connections_to_img_boundary = [0]  # initialize label 0 with 0 connections -> is tissue and does not matter
        start = 1  # dont include label 0 -> tissue

        for i in range(start, len(labels)):
            l = labels[i]
            n = count_image_boundary_coords(label_img, l)
            label_connections_to_img_boundary.append(n)

        outer_reg_label = np.argmax(label_connections_to_img_boundary)
        label_img[label_img == outer_reg_label] = 0


        # In first processed image only keep largest dark area (I previously removed the outer dark area (set to label 0) -> so largest dark area is ventricle)
        # largest dark area is second largest label because largest is 0 (background + tissue)
        if current_img_idx == last_img_idx:
            vol_labels = []
            for l in labels:
                vol_labels.append(np.sum(label_img == l))

            # second largest label is ventricle / largest is background
            sorted_l = np.argsort(vol_labels)
            ventricle_label = sorted_l[-2]
            seg = (label_img == ventricle_label) * 255
            old_seg = seg
        else:
            for i in range(1, len(labels)):
                l = labels[i]
                li_img = (label_img == l)
                coords = np.where(li_img)
                keep_region = has_coord_in_region(coords, old_seg, img_height, img_width)
                if not keep_region:
                    label_img[label_img == l] = 0

            seg = (label_img != 0) * 255
            old_seg = seg

        out_name = Path("ventricle_" + img_path.name)
        out_name = out_name.with_suffix(".png")
        output_path = output_folder / out_name
        save_image(seg, output_path, np.uint8)
        print("Ventricle Segmentation. Images left: ", current_img_idx)
        current_img_idx -= 1


def has_coord_in_region(coords, old_seg, img_height, img_width):
    """ If a coordinate in coords has value 255 in old_seg then return True else False"""

    img = np.zeros((img_height, img_width), np.uint8)
    img[coords] = 255

    six_connected = np.logical_and(img, old_seg)
    return np.any(six_connected)


def main():
    pass


if __name__ == "__main__":
    main()
