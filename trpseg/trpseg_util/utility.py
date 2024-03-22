"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file includes a lot of different utility methods for input output, 3D memory efficient image filters,
# segmentation or cell counting.
########################################################################################################################

import os
import math
import time
import shutil
import json
from itertools import combinations_with_replacement
from threading import Event

from PIL import Image
import igraph as ig
import numpy as np
from pathlib import Path
from tifffile import imwrite
from skimage.restoration import rolling_ball
from skimage.util.dtype import img_as_float32
from skimage import feature
from skimage.filters import median, gaussian
from skimage.morphology import disk, binary_dilation, binary_erosion
import skimage.morphology
from skimage.measure import label
from scipy import ndimage

#IMPORTANT
#We process and store 3D images with the following convention
#They are 3D numpy arrays that we index like imgStack[z][y][x] or imgStack[z,y,x]
#Explanation of z y x directions:
#z:
# z is the direction that goes from top to bottom in an image stack. -> Goes through the different 2D images
# e.g.: z=0 is 2D image slice Z0000 and z=100 is 2D image slice Z0100
#y:
# y is the vertical direction given a 2D image slice. So y indexes the rows starting from the top of the 2D image down to the bottom.
# e.g.: imgStack[0][2][:] is the third row of the first 2D image slice Z0000
#x:
# x is the horizontal direction given a 2D image slice. So x indexes the columns starting from the left of the 2D image to the right.
# e.g.: imgStack[0][:][2] is the third column of the first 2D image slice Z0000

#Some methods take a resolution parameter as input
#The resolution is assumed to be given in the order [z, y, x]
#e.g. [2.0, 0.325, 0.325] -> 2.0 micrometers in z direction / 0.325 micrometer in y and x direction


#Input/Output#####################IO####################################################################################
def read_image_stack(path):
    """Read an image stack that is stored as one file into a 3D numpy array.

    Parameters
    ----------
    path : str, pathlib.Path
        The path to the image stack file.

    Returns
    -------
    images : np.ndarray
        The image stack as a 3D numpy array.
        Axis 0 is z axis: axis across the 2D slices
        Axis 1 is y axis: vertical axis in a given 2D slice
        Axis 2 is x axis: horizontal axis in a given 2D Slice
    """

    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    img.close()
    images = np.array(images)
    return images


def read_image(path):
    """Read a single image into a numpy array.

    Parameters
    ----------
    path : str, pathlib.Path
        The path to the image file.

    Returns
    -------
    out : np.ndarray
        The image as 2D numpy array.
        Axis 0 is y axis: vertical axis in the given 2D image
        Axis 1 is x axis: horizontal axis in the given 2D image
    """

    img = Image.open(path)
    out = np.array(img)
    img.close()
    return out


def get_img_shape_from_filepath(img_path):
    """ Get the image height and width from a given image file path.
    Returns tuple (image_height, img_width).

    Parameters
    ----------
    img_path : str, pathlib.Path
        The path to the image file.

    Returns
    -------
    The shape of the image as 2-tuple (image_height, img_width)
    """

    img = read_image(img_path)
    return img.shape

def get_stack_shape_from_pathlist(input_file_paths):
    """Get the shape of an image stack from a list of paths of 2D image slices.

    Parameters
    ----------
    input_file_paths : list of str or list of pathlib.Path
        The paths to the image stack slices.

    Returns
    -------
    The shape of the image stack as ndarray with content [number_image_slices, image_height, img_width]
    """

    z_length = len(input_file_paths)

    img = read_image(input_file_paths[0])

    y_length, x_length = img.shape

    return np.asarray([z_length, y_length, x_length])


def read_img_list(input_list, dtype, normalize8bit=False):
    """Read the images from a list of file paths into a 3D numpy array.

    Parameters
    ----------
    input_list : list of str or list of pathlib.Path
        A list of file paths to 2D images.
    dtype : np.dtype.type
        The datatype of the read images.
        For 8-bit images use e.g. np.uint8
        For 16-bit images use e.g np.uint16
        For 32-bit images use e.g. np.float32
        For 64-bit images use e.g. np.float64
    normalize8bit : bool
        Determines whether to divide the pixel values by 255.
        This can be used when one wants to read binary images that are stored with the two values 0 and 255.

    Returns
    -------
    img : np.ndarray
        The image stack as a 3D numpy array.
        Axis 0 is z axis: axis across the 2D slices
        Axis 1 is y axis: vertical axis in a given 2D slice
        Axis 2 is x axis: horizontal axis in a given 2D Slice
    """

    img_height, img_width = get_img_shape_from_filepath(input_list[0])
    num_z_slices = len(input_list)

    img = np.empty(shape=(num_z_slices, img_height, img_width), dtype=dtype)
    i = 0
    for file_path in input_list:
        curr_img = read_image(file_path)
        if normalize8bit:
            curr_img = curr_img / 255

        curr_img = curr_img.astype(dtype)
        img[i, :, :] = curr_img

        i += 1
    print("Image Block Stacked.")
    return img


#asserts that compression is only True when path is a tif
def save_image(image, path, dtype, compression=False):
    """Save an image at the given path and with the given data type.

    Parameters
    ----------
    image : np.ndarray
        The image that should be stored as a file on disk.
    path : str, pathlib.Path
        The path where the image file is stored.
    dtype : np.dtype.type
        The data type under which the images are stored.
        For 8-bit images use e.g. np.uint8
        For 16-bit images use e.g np.uint16
        For 32-bit images use e.g. np.float32
        For 64-bit images use e.g. np.float64
    compression : bool
        Determines whether to compress the images on disk.
        Set this only to True if you save images with a .tif file extension!
    """

    img_save = Image.fromarray(image.astype(dtype))
    if compression:
        img_save.save(path, compression='tiff_lzw')
    else:
        img_save.save(path)
    img_save.close()


def save_img_stack_slices(imgStack, out_paths, dtype, compression=False):
    """Save the slices of an image stack at the given file paths under the given datatype.

    Parameters
    ----------
    imgStack : np.ndarray
        3D numpy array that should be stored in image files on disk.
    out_paths : list of str or list of pathlib.Path
        A list of file paths where the 2D image slices are stored.
    dtype : np.dtype.type
        The data type under which the image slices are stored.
    compression : bool
        Determines whether to compress the images on disk.
        Set this only to True if you save images with a .tif file extension!
    """

    for i in range(len(imgStack)):
        img = imgStack[i]
        save_image(img, out_paths[i], dtype, compression=compression)


def store_several_images_in_one_file(input_folder_path, output_folder_path, filename, shape, dtype=np.uint8, isTif=False, channel_one=False):
    """Store the image slices from the given input folder in a single image stack file

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the 2D images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the output image stack is stored
    filename : str
        The filename of the outputted image stack.
    shape : 3-tuple, array of shape (3,)
        The shape of the image stack
    dtype : np.dtype.type
        The data type under which the image stack is stored.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    out = Path(output_folder_path) / filename

    mmap = np.memmap(out, dtype=dtype, mode='w+', shape=shape)

    for i in range(len(input_files_list)):

        img = read_image(input_files_list[i])
        mmap[i] = img

        if i%200:
            mmap.flush()


#returns list of Path objects, each representing a file path
def get_file_list_from_directory(directory, isTif=None, channel_one=None):
    """Get filepaths of files from within the given directory. File paths are sorted lexicographically.

    Files inside of folders inside the given directory are not considered.

    Parameters
    ----------
    directory : str, pathlib.Path
        The path to the folder from which filepaths are extracted.
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

    Returns
    -------
    input_files_list : list of str, list of pathlib.Path
        Sorted list of file paths (sorted lexicographically)
    """

    input_folder = Path(directory)

    if channel_one is None:
        if isTif is None:
            input_list = input_folder.glob("*")
            input_list = [file_path for file_path in input_list if not file_path.is_dir()]
        elif isTif:
            input_list = input_folder.glob("*.tif")
        else:
            input_list = input_folder.glob("*.png")
    elif channel_one:
        if isTif is None:
            input_list = input_folder.glob("*_C01_*")
            input_list = [file_path for file_path in input_list if not file_path.is_dir()]
        elif isTif:
            input_list = input_folder.glob("*_C01_*.tif")
        else:
            input_list = input_folder.glob("*_C01_*.png")
    else:
        if isTif is None:
            input_list = input_folder.glob("*_C00_*")
            input_list = [file_path for file_path in input_list if not file_path.is_dir()]
        elif isTif:
            input_list = input_folder.glob("*_C00_*.tif")
        else:
            input_list = input_folder.glob("*_C00_*.png")

    if isTif is None:
        input_files_list = input_list
    else:
        input_files_list = [file_path for file_path in input_list]

    input_files_list = sorted(input_files_list)

    return input_files_list


def get_file_list_from_directory_filter(directory, file_ending=None, channel_filter=None):
    """Get filepaths of files from within the given directory. File paths are sorted lexicographically.

        Files inside of folders inside the given directory are not considered.
        A file ending and a channel_filter can be specified.

        Parameters
        ----------
        directory : str, pathlib.Path
            The path to the folder from which filepaths are extracted.
        file_ending : str, optional
            Determines, which input images inside input_folder_path are considered.
            Only files with the given file ending are considered.
            If None, all file endings are considered.
        channel_filter : str, optional
            Only files that contain that string are considered.
            This allows to for example to filter for certain channels (e.g. "C00" or "C01")
            None, means that all files are considered.

        Returns
        -------
        input_files_list : list of str, list of pathlib.Path
            Sorted list of file paths (sorted lexicographically)
        """

    input_folder = Path(directory)

    #make sure file_ending starts with a dot
    if file_ending is not None and len(file_ending) > 0:
        if file_ending[0] != ".":
            file_ending = "." + file_ending

    if file_ending is not None and channel_filter is not None:  #file_ending is not None / channel_filter is not None
        filter_str = "*" + channel_filter + "*" + file_ending
        input_list = input_folder.glob(filter_str)
        input_list = [file_path for file_path in input_list]
    elif file_ending is not None:                               #file_ending is not None / channel_filter is None
        filter_str = "*" + file_ending
        input_list = input_folder.glob(filter_str)
        input_list = [file_path for file_path in input_list]
    elif channel_filter is not None:                            #file_ending is None / channel_filter is not None
        filter_str = "*" + channel_filter + "*"
        input_list = input_folder.glob(filter_str)
        input_list = [file_path for file_path in input_list if not file_path.is_dir()]
    else:                                                       #file_ending is None / channel_filte is None
        input_list = input_folder.glob("*")
        input_list = [file_path for file_path in input_list if not file_path.is_dir()]


    input_files_list = sorted(input_list)

    return input_files_list


def png_to_tif(input_folder_path, output_folder_path, dtype, channel_one=None):
    """ Write png files from input_folder_path to tif files in output_folder_path under given datatype

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the 2D png images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the corresponding 2D tif images are stored
    dtype : np.dtype.type
        The data type under which the image stack is stored.
    channel_one : bool, optional
        Determines, which channel from the input images to use.
        True, means that only images containing the string "_C01_" are considered.
        False, means that only images containing the string "_C00_" are considered.
        None, means that all channels are considered.
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=False, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    for file in input_files_list:
        i += 1
        print("File: ", i)
        img = read_image(file)
        out_name = Path(file.name)
        out_name = out_name.with_suffix(".tif")
        output_path = output_folder / out_name
        save_image(img, output_path, dtype=dtype)


def make_binary_images_visible(input_folder_path, output_folder_path, isTif=False, channel_one=None):
    """Multiply images inside a folder by 255.

    This can be used to make binary images that only have values 0 and 1 better visualizable.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing 2D images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the corresponding output images are stored.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif, channel_one)

    numFiles = len(input_files_list)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, numFiles):
        img = read_image(input_files_list[i])

        img *= 255

        out_path = input_files_list[i].name
        out_path = output_folder / out_path
        save_image(img, out_path, np.uint8)


def create_one_file_substack(input_folder_path, out_path, x_offset: int, y_offset: int, z_offset: int, x_length: int, y_length: int, z_length: int, isTif=True, channel_one=False):
    """Create a possibly cropped single image stack file from several 2D image slices."""

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)
    z_end = z_offset + z_length
    input_files_list = input_files_list[z_offset:z_end]

    imstack = np.empty((z_length, y_length, x_length), np.uint8)

    x_end = x_offset + x_length
    y_end = y_offset + y_length

    i = 0
    for file_path in input_files_list:
        img = read_image(file_path)
        img = img[y_offset:y_end, x_offset:x_end]
        imstack[i, :, :] = img
        i += 1

    imwrite(out_path, imstack)
########################################################################################################################

#Mask Operations########################################################################################################
def substract_mask(input_folder_path, mask_folder_path, output_folder_path, isTif=False, channel_one=None, store_as_tif=False):
    """Given masks for every image in input_folder_path, set the images to 0 where their masks have a value different from 0.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the images from which the masks should be substracted.
    mask_folder_path :str, pathlib.Path
        The path to the folder containing the mask images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the corresponding output images are stored
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
    store_as_tif : bool
        If True, then store output images as tif files
        If False, then store output images as png files
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    mask_input_files_list = get_file_list_from_directory(mask_folder_path, isTif=False, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    #input_files_list = input_files_list[240:340]
    #mask_input_files_list = mask_input_files_list[240:340]
    numFiles = len(input_files_list)
    numMasks = len(mask_input_files_list)

    if numMasks != numFiles:
        raise Exception("The number of masks does not equal the number of input files!")

    for i in range(0, numFiles):
        print("Substract Mask: ", i)
        img = read_image(input_files_list[i])
        mask = read_image(mask_input_files_list[i])
        mask = (mask != 0)
        if store_as_tif:
            file_out_name = Path(input_files_list[i].name)
            file_out_name = file_out_name.with_suffix(".tif")
        else:
            file_out_name = Path(input_files_list[i].name)
            file_out_name = file_out_name.with_suffix(".png")

        out_path = output_folder / file_out_name
        img[mask] = 0

        save_image(img, out_path, dtype=np.uint8)


def substract_multiple_masks(input_folder_path, mask_folder_paths, output_folder_path, isTif=True, channel_one=True):
    """Given several masks for every image in input_folder_path, set the images to 0 where their masks have a value different from 0.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the images from which the masks should be substracted.
    mask_folder_paths :list of str, list of pathlib.Path
        The paths to the possibly multiple folders containing the mask images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the corresponding output images are stored
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    file_paths = [[] for _ in range(len(mask_folder_paths))]

    i = 0
    for mask_folder_path in mask_folder_paths:

        mask_input_files_list = get_file_list_from_directory(mask_folder_path, isTif=False)
        file_paths[i] = mask_input_files_list
        i += 1

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    numFiles = len(input_files_list)

    for i in range(0, len(file_paths)):

        numMasks = len(file_paths[i])

        if numMasks != numFiles:
            raise Exception("The number of masks does not equal the number of input files!")

    for i in range(0, numFiles):
        print("Substract Masks: ", i)
        img = read_image(input_files_list[i])
        #initialize final mask
        finalmask = read_image(file_paths[0][i])
        finalmask = (finalmask != 0)
        for j in range(1, len(file_paths)):
            mask = read_image(file_paths[j][i])
            mask = (mask != 0)
            finalmask = np.any([finalmask, mask], axis=0)

        file_out_name = input_files_list[i].name
        out_path = output_folder / file_out_name
        img[finalmask] = 4

        save_image(img, out_path, dtype=np.uint16)


def multiply_masks_by_original(input_folder_path, mask_folder_path, output_folder_path, isTif=True, channel_one=True):
    """Multiply the images given in input_folder_path with masks given in mask_folder_path.

    This method sets the images to 0 where their corresponding masks are 0.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the images which should be multiplied by mask.
    mask_folder_path :str, pathlib.Path
        The path to the folder containing the mask images.
    output_folder_path: str, pathlib.Path
        The path to the folder where the corresponding output images are stored
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    mask_input_files_list = get_file_list_from_directory(mask_folder_path, isTif=False)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    # input_files_list = input_files_list[252:253]
    # mask_input_files_list = mask_input_files_list[252:253]

    numFiles = len(input_files_list)
    numMasks = len(mask_input_files_list)

    if numFiles != numMasks:
        raise RuntimeError("The number of input files must equal the number of masks!")

    for i in range(0, numFiles):
        mask = read_image(mask_input_files_list[i])
        mask = (mask != 0) * 1
        input_img = read_image(input_files_list[i])
        output_img = mask*input_img
        out_name = Path(mask_input_files_list[i].name)
        out_name = out_name.with_suffix(".ome.tif")
        out_path = output_folder / out_name
        save_image(output_img, out_path, dtype=np.uint16)


# Write images that are completely black. All values 0.
# Writes as many black images as are in input folder. Black images have same shape as first image in input folder
def get_empty_masks(input_folder_path, output_folder_path, isTif=True, channel_one=True):
    """Write images that are completely black (all values 0).

    As many black images as are in input folder are created. Black images have same shape as first image in input folder

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing image stack slices from which shape of black image stack slices is infered.
    output_folder_path: str, pathlib.Path
        The path to the folder where the black images should be stored.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    helpimg = read_image(input_files_list[0])

    mask = np.zeros_like(helpimg, dtype=np.uint8)

    for i in range(0, len(input_files_list)):
        out_name = Path(input_files_list[i].name)
        out_name = out_name.with_suffix(".png")
        output_path = output_folder / out_name
        save_image(mask, output_path, dtype=np.uint8)

########################################################################################################################

#subtack_processing#####################################################################################################

def get_z_splits(number_images, max_images_per_split, radius):
    """Returns list of intervals, each consisting of maximal max_images_per_split.


    Parameters
    ----------
    number_images : int
        The number of images in the image stack for which splits should be determined.
    max_images_per_split : int
        The maximum number of images in one split/interval.
    radius : int
        The max radius of any filter that you use to process the substacks/splits. If you process an image stack in splits
        and use 3D image filters that take their neighborhood in a given radius in account, then you need some overlap to prevent boundary artefacts.
        The overlap used will be 2*radius.

    Returns
    -------
    splits : list of list of 2 ints
        The determined intervals of z slices in which the substacks can be processed. Includes a possible overlap.

    Notes
    ----------
    This method is useful to determine how to process image stacks block-wise with overlap.

    For example one uses a gaussian filter with radius 2 and wants to process an image stack with 30 image slices blockwise:
    Let's say max_images_per_split = 10, then this method returns:
    [0,10), [6,16), [12,22) [18,28), [24, 30]
    One first processes the first 3-dimensional image stack consisting of slices [0,10) and stores slices [0,8)
    Next, process [6,16) and store [8, 14) and so on.

    2*radius needs to be smaller than max_images_per_split

    Important! The minimal size of the last interval is 2*radius+1.
    If radius is 0 then this could cause problems because last split could include one single image slice and is therefore not 3D
    Solution: We add some images from previous split to last one
    """

    if number_images < 10:
        raise RuntimeError("Currently no image stacks with less than 10 images are allowed!")
    if max_images_per_split < 10:
        raise RuntimeError("Currently no splits with less than 10 images are allowed!")
    if max_images_per_split - 2*radius <= 1:
        raise RuntimeError("2*Radius must be smaller than max_images_per_split!")

    splits = []
    n = 0
    while n < number_images:
        start = n
        n += max_images_per_split
        if n <= number_images:
            end = n
        else:
            end = number_images

        if end != number_images:
            n = n-2*radius

        splits.append([start, end])

    #If last interval has less than 5 elements then add some from the previous interval and remove from previous.
    if splits[-1][1] - splits[-1][0] < 5:
        splits[-2][1] -= 5
        splits[-1][0] -= 5

    return splits


#file_names is only filenames of the files in the given start end interval
def store_substack_excluding_radius(substack, start, end, radius, number_images, file_names, output_folder, dtype, file_extension):
    """Store a substack excluding as many z-slices at the start and end of the stack as specified by radius.

    Parameters
    ----------
    substack : np.ndarray
        The substack that should be stored.
    start : int
        The z-coordinate of the start of the substack. start is included
    end : int
        The z-coordinate of the end of the substack. end is exluded
    radius : int
        The number of z-slices at the start and end of the image stack that will not be stored.
    number_images : int
        The total number of 2D images (z slices) in the complete stack from which the substack is a part of.
    file_names : list of pathlib.Path
        The file paths of the z slices in the substack.
    output_folder : str, pathlib.Path
        The path to the output folder where the substack slices should be stored.
    dtype : np.dtype.type
        The datatype of the output images.
    file_extension : str
        The file extension of the output images with point (e.g. ".png").
    """

    output_folder = Path(output_folder)

    shift = start  # if we substract shift from start and end then we have [start:end] = [start-start:end-start] -> to index substack

    if start == 0 and end == number_images:
        # store [start:end]
        pass
    elif start == 0:
        # store [0:end-radius]
        end = end - radius
    elif end == number_images:
        # store [start+radius:]
        start = start + radius
    else:
        # store [start+radius:end-radius]
        start = start + radius
        end = end - radius

    z_start = start - shift
    z_end = end - shift

    curr_slice = start
    for z in range(z_start, z_end):
        #print("Store substack slice: ", curr_slice)
        curr_slice += 1
        img = substack[z, :, :]
        out_name = Path(file_names[z].name)
        out_name = out_name.with_suffix(file_extension)
        out_path = output_folder / out_name
        save_image(img, out_path, dtype=dtype)

########################################################################################################################


#Neighborhood###########################################################################################################
# Assumes coord in form zyx
# Returns 2D neighbors but always prepends the same z to the returned coordinates
def get_neighbors_2D_with_z(coord, width=2048, height=2048, exclude_center=True):
    """Get the neighboring 3D coordinates of the 3D coordinate "coord".
    Although, we work here with 3D coordinates we only look for 2D neighbors. We do not change the z-coordinate.


    Parameters
    ----------
    coord : 3-tuple of int, list of three int
        The 3D coordinate for which neighbors in xy plane should be determined.
        Order of axes is [z, y, x]
    width : int
        The width of the 2D image from which the coordinate is.
        Only neighbors with a x coordinate lower than width are returned.
    height : int
        The height of the 2D image from which the coordinate is.
        Only neighbors with a y coordinate lower than height are returned.
    exclude_center : bool
        Determines whether to exclude the coordinate itself as neighbor.

    Returns
    -------
    real_neighbors : list of 3-tuple
        The neighboring 3D coordinates of "coord" in the xy-plane.
    """

    d = [-1, 0, 1]
    z = coord[0]
    coord = coord[1:]
    directions = np.stack(np.meshgrid(d, d), axis=-1).reshape(-1, 2)
    if exclude_center:
        directions = np.delete(directions, 4, axis=0)
    neighbors = directions + coord
    real_neighbors = []
    for n in neighbors:
        y = n[0]
        x = n[1]
        if y < 0 or x < 0 or y > height - 1 or x > width - 1: #check that coordinate y,x is within the image boundaries
            continue
        else:
            real_neighbors.append((z, y, x))

    return real_neighbors


# Assumes coord in form zyx
# Return 26 neighborhood coordinates
def get_neighbors_3D(coord, z_lower, z_upper, width=2048, height=2048, exclude_center=True):
    """ Get the 26-neighborhood coordinates from coord.

    Parameters
    ----------
    coord : 3-tuple of int, list of three int
        The 3D coordinate for which neighbors should be determined.
        Order of axes is [z, y, x]
    z_lower : int
        The lowest z-coordinate for which neighbors should be returned.
    z_upper : int
        The highest z-coodinate for which neighbors should be returned.
    normalize_z
    width : int
        The length of the x axis of the image stack.
        Only neighbors with a x coordinate lower than width are returned.
    height : int
        The length of the y axis of the image stack.
        Only neighbors with a y coordinate lower than height are returned.
    exclude_center : bool
        Determines whether to exclude the coordinate itself as neighbor.

    Returns
    -------
    real_neighbors : list of 3-tuple
        The coordinates of the 26-neighbors of "coord". Only coordinates within the specified ranges are returned.
    """
    d = [-1, 0, 1]
    directions = np.stack(np.meshgrid(d, d, d), axis=-1).reshape(-1, 3)
    if exclude_center:
        directions = np.delete(directions, 13, axis=0)
    neighbors = directions + coord
    real_neighbors = []
    for n in neighbors:
        z = n[0]
        y = n[1]
        x = n[2]
        if z < z_lower or y < 0 or x < 0 or z > z_upper-1 or y > height-1 or x > width-1: #check that coordinate z,y,x is within the image stack boundaries
            continue
        else:
            real_neighbors.append(n)

    return real_neighbors

########################################################################################################################


#Structuring Elements###################################################################################################
#resolution in form (z,y,x)
#radius is in pixels in xy direction
#radius in z direction will be automatically adapted to anisotropy
def anisotropic_spherical_3D_SE(radius, resolution):
    """Get a spherical structuring element of a given radius for an anisotropic resolution.

    IMPORTANT!!!: radius is interpreted as the radius of the structuring element in pixels in xy plane.
    The radius in z direction is inferred from the provided radius and the resolution.

    Parameters
    ----------
    radius : int
        The radius in pixels in the xy plane.
    resolution : (3,) array, 3-tuple
        Array-like, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]


    Returns
    -------
    struc : np.ndarray
        3 dimensional binary array that can be used as anisotropic spherical structuring element.

    References
    ----------
    Idea from: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/morphology/footprints.py#L225-L252
    """

    #assumes resolution in x-y plane is isotropic however different resolution in z-direction possible

    z_factor = resolution[0] / resolution[1]

    n = 2 * radius + 1
    Z, Y, X = np.mgrid[-radius:radius:n * 1j,
              -radius:radius:n * 1j,
              -radius:radius:n * 1j]
    Z = Z * z_factor
    s = X ** 2 + Y ** 2 + Z ** 2

    struc = np.array(s <= radius * radius, dtype=np.uint8)
    return struc


def structuring_element_1():
    """Custom ellipsoid-like structuring element with radius 2 in xy direction and radius 1 in z direction.
    Useful to process image stacks that have higher physical step size in z direction than in xy-direction
    """
    return np.array([[[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]],
                     [[0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0]],
                     [[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]])


#Assumes footprint to be a 3D array with equal size in all dimensions
def get_z_radius_from_footprint(footprint):
    """Get the maximal radius of a footprint in pixels in z direction.

    Parameters
    ----------
    footprint : np.ndarray
        A binary footprint / binary structuring element

    Returns
    -------
    The maximal radius of the footprint in pixels in z direction.
    """

    total_z_len = len(footprint)

    if not (total_z_len % 2):
        raise RuntimeError("footprint size has to be an odd number!")

    total_radius = int((total_z_len-1)/2)

    for i in range(0, total_radius):
        count = np.count_nonzero(footprint[i])
        if count > 0:
            return total_radius-i

    return 0

########################################################################################################################

def count_image_boundary_coords(img, label_to_count):
    """Compute how many pixels with a given label lie on the image boundary.

    Parameters
    ----------
    img : np.ndarray
        2D image. Pixel intensities are interpreted as labels.
    label_to_count : int
        The label (pixel value) that should be counted at the image boundary.

    Returns
    -------
    n : int
        The number of pixels with label label_to_count that are at the image boundary.
    """

    n = 0

    height, width = img.shape

    max_x = width-1
    max_y = height-1

    for x in range(0, max_x):
        if img[0][x] == label_to_count or img[max_y][x] == label_to_count:
            n += 1

    for y in range(1, max_y-1):
        if img[y][0] == label_to_count or img[y][max_x] == label_to_count:
            n += 1

    return n


#Morphology#############################################################################################################
def dilateBinaryStack2D(input_folder_path, output_folder_path, disk_radius, isTif=False, channel_one=False):
    """ Perform binary morphological dilation on every 2D image in input_folder_path separately.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological dilation should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the dilated 2D images are stored.
    disk_radius : int
        The radius of the disk which is used as structuring element in pixels.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    i = 0
    for file in input_files_list:
        print("Dilate Image: ", i)
        img = read_image(file)
        img = binary_dilation(img, disk(disk_radius))

        outpath = output_folder / file.name
        img = img*255
        save_image(img, outpath, dtype=np.uint8)
        i += 1


def erodeBinaryStack2D(input_folder_path, output_folder_path, disk_radius, isTif=False, channel_one=False):
    """ Perform binary morphological erosion on every 2D image in input_folder_path separately.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological erosion should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the eroded 2D images are stored.
    disk_radius : int
        The radius of the disk which is used as structuring element in pixels.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    i = 0
    for file in input_files_list:
        print("Erode Image: ", i)
        img = read_image(file)
        img = binary_erosion(img, disk(disk_radius))

        outpath = output_folder / file.name
        img = img * 255
        save_image(img, outpath, dtype=np.uint8)
        i += 1


def erosion3D(input_folder_path, output_folder_path, footprint, radius, max_images_in_memory=250, isTif=False, channel_one=None, canceled=Event()):
    """Perform 3D binary morphological erosion.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological erosion should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the eroded 3D images are stored.
    footprint : np.ndarray
        The 3D footprint / structuring element for performing morphological erosion.
    radius : int
        The maximal radius in z direction of the footprint in pixels.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    #input_files_list = input_files_list[494:]

    #footprint = ball(5)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    number_images = len(input_files_list)

    splits = get_z_splits(number_images, max_images_in_memory, radius)

    for start, end in splits:

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint8)
        eros = ndimage.binary_erosion(imgStack, footprint, iterations=1, border_value=1).astype(np.uint8)
        eros = eros * 255

        store_substack_excluding_radius(eros, start, end, radius, number_images, process_files, output_folder,
                                        dtype=np.uint8, file_extension=".png")


def dilation3D(input_folder_path, output_folder_path, footprint, radius, max_images_in_memory=250, isTif=False, channel_one=None, canceled=Event()):
    """Perform 3D binary morphological dilation.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological dilation should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the dilated 3D images are stored.
    footprint : np.ndarray
        The 3D footprint / structuring element for performing morphological dilation.
    radius : int
        The maximal radius in z direction of the footprint in pixels.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    #input_files_list = input_files_list[494:]

    #footprint = ball(5)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    number_images = len(input_files_list)
    splits = get_z_splits(number_images, max_images_in_memory, radius)

    for start, end in splits:

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint8)
        dil = ndimage.binary_dilation(imgStack, footprint, iterations=1, border_value=0).astype(np.uint8)
        dil = dil * 255

        store_substack_excluding_radius(dil, start, end, radius, number_images, process_files, output_folder,
                                        dtype=np.uint8, file_extension=".png")


def opening3D(input_folder_path, output_folder_path, footprint, radius, max_images_in_memory=250, isTif=False, channel_one=None, canceled=Event()):
    """Perform 3D binary morphological opening.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological opening should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the morphologically opened 3D images are stored.
    footprint : np.ndarray
        The 3D footprint / structuring element for performing morphological opening.
    radius : int
        The maximal radius in z direction of the footprint in pixels.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    #input_files_list = input_files_list[494:]

    #footprint = ball(5)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    number_images = len(input_files_list)
    splits = get_z_splits(number_images, max_images_in_memory, radius)

    for start, end in splits:

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint8)

        tmp = ndimage.binary_erosion(imgStack, footprint, iterations=1, border_value=1).astype(np.uint8)

        if canceled.is_set():
            return

        out = ndimage.binary_dilation(tmp, footprint, iterations=1, border_value=0).astype(np.uint8)
        out = out * 255

        store_substack_excluding_radius(out, start, end, radius, number_images, process_files, output_folder,
                                        dtype=np.uint8, file_extension=".png")


def closing3D(input_folder_path, output_folder_path, footprint, radius, max_images_in_memory=250, isTif=False, channel_one=None, canceled=Event()):
    """Perform 3D binary morphological closing.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices on which morphological closing should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the morphologically closed 3D images are stored.
    footprint : np.ndarray
        The 3D footprint / structuring element for performing morphological closing.
    radius : int
        The maximal radius in z direction of the footprint in pixels.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    #input_files_list = input_files_list[494:]

    #footprint = ball(5)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    number_images = len(input_files_list)
    splits = get_z_splits(number_images, max_images_in_memory, radius)

    for start, end in splits:

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint8)

        tmp = ndimage.binary_dilation(imgStack, footprint, iterations=1, border_value=0).astype(np.uint8)

        if canceled.is_set():
            return

        out = ndimage.binary_erosion(tmp, footprint, iterations=1, border_value=1).astype(np.uint8)
        out = out * 255

        store_substack_excluding_radius(out, start, end, radius, number_images, process_files, output_folder,
                                        dtype=np.uint8, file_extension=".png")


########################################################################################################################


#Filters################################################################################################################


def rolling_ball_2D(input_folder_path, output_folder_path, radius=50, isTif=True, channel_one=True):
    """Perform 2D rolling ball background substraction on every image in input_folder_path separately.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the images on which the rolling ball filter is applied.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    radius : int, optional
        The radius of the rolling ball filter in pixels.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    #input_files_list = input_files_list[240:340]

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, len(input_files_list)):
        print("Rolling Ball on Image: ", i)
        img = read_image(input_files_list[i])

        background = rolling_ball(img, radius=radius)
        filtered_img = img - background

        out_name = "roll_" + input_files_list[i].name
        out_path = output_folder / out_name
        save_image(filtered_img, out_path, dtype=np.uint16)


def gaussian_smooth_stack(input_folder_path, output_folder_path, sigma, max_images_in_memory=100, isTif=True, channel_one=False):
    """Perfrom 3D gaussian smoothing on an image stack.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    sigma : int, float, list of 3 int, list of 3 float
        The sigma (standard deviation) of the gaussian smoothing filter. A value can be specified for all axes individually.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    """
    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    truncate = 3.0
    # max radius of gaussian filter in z direction is int(truncate * sd + 0.5) (sd is sigma)
    # If I assume truncate=3 and sigma not greater than 2 then radius is max 7
    if isinstance(sigma, (list, tuple, np.ndarray)):
        radius = int(truncate * sigma[0] + 0.5)
    else:
        radius = int(truncate * sigma + 0.5)

    number_images = len(input_files_list)
    splits = get_z_splits(number_images, max_images_in_memory, radius)

    # Load and store overlap idea (max_images_in_memory=100, radius=7)
    # load  [0,100) [86,186) [172,272) [258,358) [358-14,357-14+100)        ...
    # store [0,93) [93,179) [179,265) [265,351) [358-14+7, 357-14+100-7)    ...

    for start, end in splits:

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint16)
        imgStack = img_as_float32(imgStack)  # to save memory. Else gaussian method transforms data into float64
        smoothed = gaussian(imgStack, sigma=sigma, truncate=3)
        smoothed = smoothed * 65535
        smoothed = smoothed.astype(np.uint16)

        store_substack_excluding_radius(smoothed, start, end, radius, number_images, process_files, output_folder, dtype=np.uint16, file_extension=".tif")


def median_smooth_stack(input_folder_path, output_folder_path, resolution, median_radius_xy, max_images_in_memory=100, isTif=True, channel_one=False):
    """Perfrom 3D median smoothing on an image stack.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    median_radius_xy : int
        The radius of the median filter in xy direction.
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    footprint = anisotropic_spherical_3D_SE(median_radius_xy, resolution)
    # max radius of median filter in z_direction
    z_factor = resolution[1] / resolution[0]
    z_radius = math.ceil(median_radius_xy * z_factor)

    number_images = len(input_files_list)
    splits = get_z_splits(number_images, max_images_in_memory, z_radius)

    # Load and store overlap idea (max_images_in_memory=100, radius=7)
    # load  [0,100) [86,186) [172,272) [258,358) [358-14,357-14+100)        ...
    # store [0,93) [93,179) [179,265) [265,351) [358-14+7, 357-14+100-7)    ...

    for start, end in splits:

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint16)
        smoothed = median(imgStack, footprint=footprint)

        store_substack_excluding_radius(smoothed, start, end, z_radius, number_images, process_files, output_folder, dtype=np.uint16, file_extension=".tif")


########################################################################################################################

###############Measure##################################################################################################
def count_objects_3D_whole(input_folder_path, isTif=False, channel_one=None):
    """Count the number of objects in the given image stack.

    This method loads and processes the whole image stack into memory at once. So be sure you have enough memory available!

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack
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

    Returns
    -------
    The number of objects in the image stack.
    """
    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    imgStack = read_img_list(input_files_list, np.uint8)

    label_img = label(imgStack, background=0, connectivity=2)  # could also use connectivity 1 or 3 if wanted
    labels = np.unique(label_img)

    return len(labels)


def count_objects_3D_memeff(input_folder_path, output_folder_path=None, max_images_in_memory=None, connectivity=1,
                            label_imgs_dtype=np.uint32, isTif=False, channel_one=None, canceled=Event(),
                            progress_status=None, maxProgress=90):
    """ Count the number of objects in the given image stack.
    A maximal number of images that will be loaded into memory and processed at once can be specified.


    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack
    output_folder_path : str, pathlib.Path
        The folder where the labeled image stack slices can be stored. To count the different objects the image stack is labeled at first.
        If you want to store the labeled images then specify an output folder path here.
        If None, then the labeled image stack will be deleted after the objects have been counted.
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    connectivity : int
        The connectivity parameter for the skimage method "skimage.measure.label"
    label_imgs_dtype : np.dtype.type
        The datatype for the label images.
        If np.uint16: Then no more than 2^16 == 65535 regions can be counted.
        If np.uint32: Then 4294967295 regions can be counted.
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    maxProgress : int
        The maximal progress percentage that will be emitted by progress_status

    Returns
    -------
    The number of objects in the image stack.
    """
    
    progress = 0
    if progress_status is not None:
        progress_status.emit(["Counting connected regions...", progress])


    input_folder_path = Path(input_folder_path)
    delete_output_later = False

    if output_folder_path is None:
        delete_output_later = True

        output_folder_path = input_folder_path / f"tmp_out_{time.strftime('%Y%m%d_%H%M%S')}"

    if max_images_in_memory is None:
        file_paths_help = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)
        if len(file_paths_help) < 1:
            raise RuntimeError("There needs to be at least 1 image in your specified input_folder!")
        shape_help = get_img_shape_from_filepath(file_paths_help[0])
        max_images_in_memory = get_max_images_in_memory_from_img_shape(shape_help)

    label_stats = label_3D_memeff(input_folder_path, output_folder_path, max_images_in_memory=max_images_in_memory, connectivity=connectivity, out_dtype=label_imgs_dtype, isTif=isTif, channel_one=channel_one, canceled=canceled, progress_status=progress_status, maxProgress=maxProgress)

    if canceled.is_set():
        return

    num_regions, max_label = label_stats

    if delete_output_later:
        shutil.rmtree(output_folder_path)

    return num_regions


def count_labels_3D_memeff(input_folder_path, isTif=None, channel_one=None, return_labels=False, canceled=Event(), progress_status=None, maxProgress=90):
    """Count the number of labels in a labeled image stack.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack
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
    return_labels : bool
        Determines whether to return a list of the labels
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    maxProgress : int
        The maximal progress percentage that will be emitted by progress_status

    Returns
    -------
    If return_labels == False: Return the total number of unique labels in the given labeled image stack
    If return_labels == True: Return the total number of unique labels in the given labeled image stack as well as a list of the encountered labels
    """

    progress = 0
    if progress_status is not None:
        progress_status.emit(["Counting labels...", progress])

    input_files_list = get_file_list_from_directory(input_folder_path, isTif, channel_one=channel_one)

    number_images = len(input_files_list)

    labels = set()

    for i in range(number_images):

        if canceled.is_set():
            return

        progress = int((i / number_images) * maxProgress)
        if progress_status is not None:
            progress_status.emit(["Counting labels...", progress])

        img = read_image(input_files_list[i])

        elems = np.unique(img)

        labels.update(elems)

    num_labels = len(labels)

    if return_labels:
        return num_labels, list(labels)

    return num_labels


def remove_small_objects_3D_wholeStack(input_folder_path, output_folder_path, region_min, isTif=False, channel_one=None):
    """Remove small objects from a binary image stack given in slices.
    This method loads and processes the whole image stack into memory at once. So be sure you have enough memory available!

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    region_min : int
        Objects with a smaller number of pixels than region_min are removed.
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
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    imgStack = read_img_list(input_files_list, dtype=np.bool)

    out = skimage.morphology.remove_small_objects(imgStack, region_min, connectivity=1)

    for i in range(len(out)):

        out_path = output_folder / input_files_list[i].name
        save_image(out[i], out_path, dtype=np.uint8)


def remove_small_objects_3D_memeff(input_folder_path, output_folder_path, min_num_voxels, z_remove_width, max_images_in_memory=100,
                                   store_label_imgs=False, label_imgs_dtype=np.uint32, isTif=False, channel_one=None, canceled=Event(),
                                   progress_status=None, maxProgress=90):
    """Remove small objects from a binary image stack given in slices.
    A maximal number of images that will be loaded into memory and processed at once can be specified.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    min_num_voxels : int
        Objects with less than min_num_voxels voxels are removed.
    z_remove_width
        Objects that span over less than z_remove_width pixels in z-direction are removed.
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    store_label_imgs : bool, optional
        If True, then the labeled image stack is stored. For small object removal the image stack is labeled beforehand.
    label_imgs_dtype : np.dtype.type, optional
        The datatype for the label images.
        If np.uint16: Then no more than 2^16 == 65535 regions can be counted.
        If np.uint32: Then 4294967295 regions can be counted.
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    maxProgress : int
        The maximal progress percentage that will be emitted by progress_status
    """

    progress = 0
    if progress_status is not None:
        progress_status.emit(["Starting remove small objects...", progress])


    input_folder_path = Path(input_folder_path)
    output_folder_path = Path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    delete_label_imgs_later = not store_label_imgs

    label_output_folder = output_folder_path / f"label_images_before_rem_small_objects"

    if canceled.is_set():
        return

    label_stats = label_3D_memeff(input_folder_path, label_output_folder, max_images_in_memory, out_dtype=label_imgs_dtype,
                                  return_region_sizes=True, isTif=isTif, channel_one=channel_one, canceled=canceled,
                                  progress_status=progress_status, maxProgress=int(maxProgress/2))

    if canceled.is_set():
        return

    progress = int(maxProgress/1.5)
    if progress_status is not None:
        progress_status.emit(["Remove small objects...", progress])

    num_labels, max_label, region_sizes = label_stats

    #Go through region_sizes and determine which labels to delete
    labels_to_remove = []

    for key, value in region_sizes.items():
        voxel_num = value[0]
        z_width = value[1]

        if voxel_num < min_num_voxels or z_width < z_remove_width or key == 0:
            labels_to_remove.append(key)

    #Determine mapping of labels(index of array new_labels) to 0(background) and 1(object)
    new_labels = np.ones(shape=(max_label+1,), dtype=np.uint8)
    new_labels = new_labels * 255
    new_labels = new_labels.astype(np.uint8)
    new_labels[labels_to_remove] = 0

    # Set labels in labels_to_remove to 0 (background/non_object) other to 1(object)
    label_imgs_are_tif = False
    if np.dtype(label_imgs_dtype).itemsize > 2:
        label_imgs_are_tif = True

    label_files_list = get_file_list_from_directory(label_output_folder, isTif=label_imgs_are_tif)
    img_height, img_width = get_img_shape_from_filepath(label_files_list[0])

    for i in range(len(label_files_list)):

        if canceled.is_set():
            return

        img = read_image(label_files_list[i])
        img = img.ravel() #represent as 1d to efficiently change the labels
        img = new_labels[img]
        img = np.reshape(img, (img_height, img_width))

        out_name = Path(label_files_list[i].name)
        out_name = out_name.with_suffix(".png")
        out_path = output_folder_path / out_name

        save_image(img, out_path, np.uint8)

    if delete_label_imgs_later:
        shutil.rmtree(label_output_folder)

    if progress_status is not None:
        progress_status.emit(["Remove small objects finished...", maxProgress])


def label_3D_memeff(input_folder_path, output_folder_path, max_images_in_memory=100, connectivity=1, out_dtype=np.uint32,
                    return_region_sizes=False, isTif=False, channel_one=None, canceled=Event(), progress_status=None, maxProgress=90):
    """

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    connectivity : int
        The connectivity parameter for the skimage method "skimage.measure.label"
    out_dtype : np.dtype.type, optional
        The datatype for the label images.
        If np.uint16: Then no more than 2^16 == 65535 regions can be counted.
        If np.uint32: Then 4294967295 regions can be counted.
    return_region_sizes : bool
        If True, return a dictionary that returns the number of voxels for each label
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    maxProgress : int
        The maximal progress percentage that will be emitted by progress_status

    Returns
    -------
    If return_region_sizes == True, then return the total number of labels, the maximum integer in the labeled image and
    the number of voxels belonging to each label

    If return_region_sizes == False, then return the total number of labels, the maximum integer in the labeled image
    """

    progress = 0
    if progress_status is not None:
        progress_status.emit(["Label the objects...", progress])

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)
    #input_files_list = input_files_list[200:300]

    if len(input_files_list) == 0:
        raise RuntimeError(f"No files found in {input_folder_path} !")

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    number_images = len(input_files_list)

    output_list_files = []
    for i in range(number_images):
        out_name = Path(input_files_list[i].name)
        if np.dtype(out_dtype).itemsize > 2:
            out_name = out_name.with_suffix(".tif")
        else:
            out_name = out_name.with_suffix(".png")

        out_path = output_folder / out_name
        output_list_files.append(out_path)

    img_height, img_width = get_img_shape_from_filepath(input_files_list[0])

    splits = get_z_splits(number_images, max_images_in_memory, 0)

    current_max_label = 0

    for start, end in splits:

        progress = int((start/number_images) * (maxProgress/2))

        if progress_status is not None:
            progress_status.emit(["Label the objects...", progress])

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, np.uint8)

        labels, num_labels = label(imgStack, return_num=True, connectivity=connectivity) #labels is of dtype int64
        labels[labels != 0] += current_max_label

        current_max_label += num_labels

        save_img_stack_slices(labels, output_list_files[start:end], out_dtype, compression=True)

    # Correct errors at z borders between max_images_in_memory

    if progress_status is not None:
        progress_status.emit(["Correction of boundary errors...", int(maxProgress/2)])

    labels_to_merge = set()

    for _, end in splits:
        if end != number_images:
            # load both images where the gap between number_max_images_in_memory is
            img_below = read_image(output_list_files[end-1])
            img_above = read_image(output_list_files[end])

            object_regions_below = img_below != 0
            object_regions_above = img_above != 0

            six_connected = np.logical_and(object_regions_below, object_regions_above)

            #get connected label values
            labels_conn_below = img_below[six_connected]
            labels_conn_above = img_above[six_connected]

            conn_labels = zip(labels_conn_below, labels_conn_above)

            labels_to_merge.update(conn_labels)

    #Get sets of labels that are connected and need to be merged to one label
    merge_labels = []
    keys = []
    values = []
    old_num_labels = current_max_label

    g = ig.Graph(edges=list(labels_to_merge), directed=False)

    at_least_one_edge_vertices_ids = g.vs.select(_degree_gt=0).indices
    at_least_one_edge_vertices_ids = np.asarray(at_least_one_edge_vertices_ids)

    g_smaller = g.induced_subgraph(at_least_one_edge_vertices_ids)  # only labels that are connected to toher labels need to be taken into account

    vertex_clustering = ig.Graph.connected_components(g_smaller)  # defaults to strong connected components but for undirected graph it is ignored

    for connected_labels in vertex_clustering:
        connected_labels = at_least_one_edge_vertices_ids[connected_labels]
        merge_labels.append(connected_labels)

    #create mapping of labels to new_labels (the first label in a connected label will be assigned to all connected labels)
    for m_labels in merge_labels:
        map_these = m_labels[1:]
        value = m_labels[0]

        for m_t in map_these:
            keys.append(m_t)
            values.append(value)


    new_labels = np.arange(0, old_num_labels+1, dtype=out_dtype)
    new_labels[keys] = values

    max_label = np.max(new_labels)

    new_num_labels = old_num_labels - len(keys)
    region_sizes = dict()
    # Now merge the labels: Go through the images and set the new labels
    for i in range(number_images):

        if canceled.is_set():
            return

        img = read_image(output_list_files[i])
        img = img.ravel() #represent as 1d to efficiently change the labels
        img = new_labels[img]
        img = np.reshape(img, (img_height, img_width))

        #if return_region_sizes a dict ist returned with the number of voxels for each label and the number of z-slices that contain that label
        # {label: [num_voxels, z_width], ...}
        if return_region_sizes:
            values, counts = np.unique(img, return_counts=True)
            for j in range(len(values)):
                label_l = values[j]
                if label_l in region_sizes:
                    region_sizes[label_l][0] += counts[j]
                    region_sizes[label_l][1] += 1
                else:
                    region_sizes[label_l] = [counts[j], 1]

        save_image(img, output_list_files[i], out_dtype, compression=True)

    if progress_status is not None:
        progress_status.emit(["Labeling finished...", maxProgress])

    if return_region_sizes:
        return new_num_labels, max_label, region_sizes

    return new_num_labels, max_label


def count_segmentation_pixels(input_folder_path, isTif=False, channel_one=True, canceled=Event(), progress_status=None, maxProgress=90):
    """ Count the total number of non-zero pixels in a 3D image stack given in slices.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack.
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
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    maxProgress : int
        The maximal progress percentage that will be emitted by progress_status

    Returns
    -------
    num_pixels : int
        The number of non-zero voxels in the provided image stack.
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    numFiles = len(input_files_list)
    num_pixels = 0

    for i in range(numFiles):

        if canceled.is_set():
            return

        progress = int((i / numFiles) * maxProgress)
        if progress_status is not None:
            progress_status.emit(["Computing total segmentation volume...", progress])

        img = read_image(input_files_list[i])
        num_pixels += np.count_nonzero(img)

    return num_pixels


def write_counting_stats_to_txt(output_file_path, num_labels, region_sizes=None, labels=None): #region_sizes assumed to be dict and labels assumed to be list
    """Write statistics from counting pars tuberalis cells or labels to a txt file.

    Parameters
    ----------
    output_file_path
    num_labels : int
        The number of unique labels.
    region_sizes : dict
        A dictionary that maps labels to their number of voxels that have the corresponding label
    labels : list of int, np.ndarray
        A list containing the unique labels
    """

    with open(output_file_path, 'w') as f:

        f.write("Number of Cells: " + str(num_labels) + "\n")

        if region_sizes is not None:
            f.write("Size of labeled regions [voxel_num, z_width]: \n")
            f.write(json.dumps(region_sizes))

        if labels is not None:
            labels = list(labels)

            f.write("\nLabels = [")

            labels_string = ','.join(str(l) for l in labels)
            f.write(labels_string)
            f.write("]\n")


def get_voxel_num_from_physical_volume(resolution, volume_size):
    """Get the rounded down number of voxels that have a phyical volume of volume_size.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y and x direction in order [z_resolution, y_resolution, x_resolution]
    volume_size : int, float
        The volume size in physical units (micrometer^3).

    Returns
    -------
    voxel_num : int
        The number of voxels that have an approximate physical volume of volume_size.
    """
    voxel_volume = resolution[0] * resolution[1] * resolution[2]
    voxel_num = int(volume_size // voxel_volume)

    return voxel_num

def get_voxel_num_from_physical_length(resolution, physical_length, axis):
    """Get number of voxels that have a specified phyical length in a given direction (over a given axis).

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y and x direction in order [z_resolution, y_resolution, x_resolution]
    physical_length : int, float
        The physical length (in micrometers).
    axis : int
        The axis over which the length will be measured.

    Returns
    -------
    voxel_num : int
        The number of voxels that have an approximate length pf physical_length over the specified axis.
    """
    
    voxel_length = resolution[axis]
    voxel_num = int(physical_length // voxel_length)

    return voxel_num


def get_max_images_in_memory_from_img_shape(shape, for_watershed_split=False):
    """ Get the number of images of a certain shape that can be loaded into memory so that
    not more than 16GB RAM are used.

    It is assumed that the images are 16bit (= 2 Byte).

    Parameters
    ----------
    shape : 2-tuple of int
        The width and height of one 2D image slice.
    for_watershed_split : bool
        Get the maximum number of images loaded into memory for watershed cell splitting.
        The algorithms has higher memory usage, so less images can be loaded into memory.

    Returns
    -------
    An integer which specifies the number of 2D image slice that can be loaded into memory at once (for 16GB RAM limit).
    """

    current_image_slice_num_pixels = shape[0] * shape[1]
    # For 16bit images we normally want to allow to load 100*2048*2048 pixels into memory at on once
    if not for_watershed_split:
        return int(419430400//current_image_slice_num_pixels)
    else:
        # For watershed only allow to load 50*2048*2048 pixels into memory at once
        return int(209715200//current_image_slice_num_pixels)


# Method _texture_filter from scikit-image/skimage/feature/_basic_features.py
# It allows to compute the hessian matrix without additional gaussian_smoothing
# which is currently not possible with the method feature.hessian_matrix from skimage
# texture_filter returns the ordered hessian matrix eigenvalues
######################################################################################
#    Title: _texture_filter
#    Date: 12th May 2023
#    Code version: v0.20.0
#    Availability: https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_basic_features.py
######################################################################################
def texture_filter(gaussian_filtered):
    H_elems = [
        np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)
    ]
    eigvals = feature.hessian_matrix_eigvals(H_elems)
    return eigvals
########################################################################################################################
