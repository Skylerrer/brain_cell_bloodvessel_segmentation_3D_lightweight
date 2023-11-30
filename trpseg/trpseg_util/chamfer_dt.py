"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

import os
import math
from threading import Event

from pathlib import Path
from numba import jit
import numpy as np

from trpseg.trpseg_util.utility import get_file_list_from_directory, read_image, save_image, get_stack_shape_from_pathlist, get_z_splits, read_img_list

# This file implements the 3D chamfer distance transform for anisotropic voxel grids from the following paper:
######################################################################################
#   Title: Weighted Distance Transforms for Images Using Elongated Voxel Grids
#   Authors: Ida-Maria Sintorn, Gunilla Borgefors
#   Date: 1st January 2002
#   Availability: https://link.springer.com/chapter/10.1007/3-540-45986-3_22
######################################################################################


# Basic Algorithm for isotropic grids in arbitrary dimensions described in:
######################################################################################
#   Title: Distance Transformations in Arbitrary Dimensions
#   Authors: Gunilla Borgefors
#   Date: 3rd September 1984
#   Availability: https://www.sciencedirect.com/science/article/abs/pii/0734189X84900355
######################################################################################


#Forward Pass
    #Go from top down
    #Slide over every pixel and if pixel in center of 3x3x3 cube is != 0 then compute new distance
    #Look at relative neighbors [-1,-1,-1]e, [-1,-1,0]d, [-1,-1,1]e, [-1,0,-1]d, [-1,0,0]c, [-1,0,1]d, [-1,1,-1]e, [-1,1,0]d, [-1,1,1]e
    # [0,-1,-1]b, [0,-1,0]a, [0,-1,1]b, [0,0,-1]a

@jit(nopython=True)
def setMinForwardPaddedShape(dist_out, shape, weights):
    """
    Perform a forward pass over the image and store minimal distances to pixels with value 0.
    Parameters
    ----------
    dist_out : np.ndarray
        Array in which results are stored.
    shape
        The shape of the padded image stack.
    weights
        The weights for the possibly anisotropic distance computations.

    Notes
    ----------
    jit compilation is used to achieve a huge speed up
    """
    a, b, c, d, e = weights

    for z in range(1,shape[0]):
        for y in range(1, shape[1]-1):
            for x in range(1, shape[2]-1):

                currentMin = dist_out[z,y,x]

                if currentMin != 0:
                    #Go through thirteen neighbors and check whether neighbor + the appropriate weight < currentMin
                    dist1 = dist_out[z-1, y-1,x-1] + e
                    dist2 = dist_out[z-1,y-1,x] + d
                    dist3 = dist_out[z-1,y-1,x+1] + e
                    dist4 = dist_out[z-1, y, x-1] + d
                    dist5 = dist_out[z-1, y, x] + c
                    dist6 = dist_out[z-1, y, x+1] + d
                    dist7 = dist_out[z-1, y+1,x-1] + e
                    dist8 = dist_out[z-1, y+1, x] + d
                    dist9 = dist_out[z-1, y+1, x+1] + e
                    dist10 = dist_out[z, y-1, x-1] + b
                    dist11 = dist_out[z, y-1, x] + a
                    dist12 = dist_out[z, y-1, x+1] + b
                    dist13 = dist_out[z, y, x-1] + a

                    newMin = min(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10, dist11, dist12, dist13)

                    if newMin < currentMin:
                        dist_out[z,y,x] = newMin


# Backward Pass
    # Go from bottom up
    # Look at relative neighbors [0,0,1]a, [0,1,-1]b, [0,1,0]a, [0,1,1]b
    # [1,-1,-1]e, [1,-1,0]d, [1,-1,1]e, [1,0,-1]d, [1,0,0]c, [1,0,1]d, [1,1,-1]e, [1,1,0]d, [1,1,1]e

@jit(nopython=True)
def setMinBackwardPaddedShape(dist_out, shape, weights):
    """
    Perform a backward pass over the image stack and store minimal distances to pixels with value 0.
    Parameters
    ----------
    dist_out : np.ndarray
        Array in which results are stored.
    shape
        The shape of the padded image stack.
    weights
        The weights for the possibly anisotropic distance computations.

    Notes
    ----------
    jit compilation is used to achieve a huge speed up
    """
    a, b, c, d, e = weights

    for z in range(shape[0]-2, -1, -1):
        for y in range(shape[1]-2, 0, -1):
            for x in range(shape[2]-2, 0, -1):

                currentMin = dist_out[z,y,x]

                if currentMin != 0:

                    # Go through thirteen neighbors and check whether neighbor + the appropriate weight < currentMin
                    dist1 = dist_out[z, y, x+1] + a
                    dist2 = dist_out[z, y+1, x-1] + b
                    dist3 = dist_out[z, y+1, x] + a
                    dist4 = dist_out[z, y+1, x+1] + b
                    dist5 = dist_out[z+1, y-1, x-1] + e
                    dist6 = dist_out[z+1, y-1, x] + d
                    dist7 = dist_out[z+1, y-1, x+1] + e
                    dist8 = dist_out[z+1, y, x-1] + d
                    dist9 = dist_out[z+1, y, x] + c
                    dist10 = dist_out[z+1, y, x+1] + d
                    dist11 = dist_out[z+1, y+1, x-1] + e
                    dist12 = dist_out[z+1, y+1, x] + d
                    dist13 = dist_out[z+1, y+1, x+1] + e


                    newMin = min(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10, dist11, dist12, dist13)

                    if newMin < currentMin:
                        dist_out[z, y, x] = newMin


#For more information see: "Weighted Distance Transforms for Images Using Elongated Voxel Grids", Sintorn,Borgefors
#Cube (planes from top to bottom):
#[[[e,d,e],
#  [d,c,d],
#  [e,d,e]],
#
# [[b,a,b],
#  [a,0,a],
#  [b,a,b]],
#
# [[e,d,e],
#  [d,c,d],
#  [e,d,e]]]

def compute_weights(resolution):
    """Compute the weights for the anisotropic chamfer distance transformation

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    5-tuple of floats that are the computed weights

    Notes
    ----------
    Uses formulas from:
    ######################################################################################
    #   Title: Weighted Distance Transforms for Images Using Elongated Voxel Grids
    #   Authors: Ida-Maria Sintorn, Gunilla Borgefors
    #   Date: 1st January 2002
    #   Availability: https://link.springer.com/chapter/10.1007/3-540-45986-3_22
    ######################################################################################
    """

    # z_f is anisotropy factor
    z_f = resolution[0]/resolution[1]

    sqrt2 = math.sqrt(2)
    a = (3 * z_f ** 2 + z_f * (sqrt2 - 2 - math.sqrt(z_f ** 2 + 2)) + math.sqrt(z_f ** 2 + 2) + 1 - sqrt2) / (5 * z_f ** 2 - 2 * z_f + 1) + \
        (math.sqrt(z_f ** 2 * (4 * math.sqrt(z_f ** 2 + 2) * (2 * sqrt2 - 1 + z_f) + 5 * z_f ** 2 * (2 * sqrt2 - 3) + 2 * z_f * (3 - 4 * sqrt2) + 6 * sqrt2 - 19))) / (5 * z_f ** 2 - 2 * z_f + 1)

    b = a - 1 + sqrt2
    c = z_f * a
    d = z_f * a - z_f + math.sqrt(1 + z_f*z_f)
    e = z_f * a - z_f + math.sqrt(z_f*z_f + 2)

    return np.float32(a), np.float32(b), np.float32(c), np.float32(d), np.float32(e)



#Memory-efficient block-wise chamfer distance transform
def ch_distance_transform_files_memeff(input_folder_path, output_folder_path, resolution, max_images_in_memory=100, isTif=False, channel_one=False, output_dtype=np.uint16, canceled=Event()):
    """
    Perform 3D chamfer distance transform on given image slices.
    Process Stack in blocks to reduce RAM usage.
    Allows also anisotropic resolutions (z resolution != y resolution = x resolution).

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the binary image slices to be processed.
    output_folder_path: str, pathlib.Path
        The path to the folder where the output images are stored
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    max_images_in_memory : int, optional
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

    input_file_paths = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)
    original_stack_shape = get_stack_shape_from_pathlist(input_file_paths)
    padded_stack_shape = original_stack_shape + 2
    original_stack_shape = tuple(original_stack_shape)
    padded_stack_shape = tuple(padded_stack_shape)
    padded_z_plane_shape = padded_stack_shape[1:]

    output_folder_path = Path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)


    if canceled.is_set():
        return

    #Initialize distance mmap

    inf = 2048*16 #this value is chosen arbitrary and used instead of infinity in the paper (only needs to be larger than the greatest possible distance)


    tmp_file_names = "tmp_dist"


    #initialize output images
    for i in range(0, original_stack_shape[0]):

        if canceled.is_set():
            return

        img = read_image(input_file_paths[i])
        img = img.astype(np.float32)
        img[img != 0] = inf
        img = np.pad(img, (1,1), mode='constant', constant_values=inf)
        current_out_name = tmp_file_names + str(i).zfill(4) + ".tif"
        current_out_path = output_folder_path / current_out_name
        save_image(img, current_out_path, np.float32)



    #Compute weights
    weights = compute_weights(resolution)

    print("Chamfer Distance Transform: Start Forward Pass")

    dist_file_paths = get_file_list_from_directory(output_folder_path, isTif=True, channel_one=None)
    num_dist_files = len(dist_file_paths)
    splits = get_z_splits(num_dist_files, max_images_in_memory, 0)

    # padding z-plane
    pad_plane = np.ones(shape=padded_z_plane_shape, dtype=np.float32)
    pad_plane *= inf
    next_padding_plane = pad_plane

    #Forward Pass
    for j in range(0, len(splits)):
        start, end = splits[j]

        current_substack = read_img_list(dist_file_paths[start:end], dtype=np.float32)

        #pad substack
        current_substack = np.concatenate(([next_padding_plane], current_substack), axis=0)
        current_substack_shape = current_substack.shape
        setMinForwardPaddedShape(current_substack, current_substack_shape, weights)

        for i in range(1, current_substack_shape[0]):
            save_image(current_substack[i], dist_file_paths[start+i-1], dtype=np.float32)


        next_padding_plane = current_substack[-1]

        if canceled.is_set():
            return


    print("Chamfer Distance Transform: Start Backward Pass")

    # padding z-plane
    pad_plane = np.ones(shape=padded_z_plane_shape, dtype=np.float32)
    pad_plane *= inf
    next_padding_plane = pad_plane

    # Backward Pass
    for j in range(len(splits)-1,-1,-1):
        start, end = splits[j]

        current_substack = read_img_list(dist_file_paths[start:end], dtype=np.float32)

        # pad substack
        current_substack = np.concatenate((current_substack, [next_padding_plane]), axis=0)
        current_substack_shape = current_substack.shape
        setMinBackwardPaddedShape(current_substack, current_substack_shape, weights)

        for i in range(0, current_substack_shape[0]-1):
            img = current_substack[i]
            os.remove(dist_file_paths[start + i])
            img = img[1:-1, 1:-1]  # remove padding
            img = img * resolution[1]  # correct for x_resolution and y_resolution != 1.0

            outname = Path(input_file_paths[start + i].name)
            if np.dtype(output_dtype).itemsize > 2:
                outname = outname.with_suffix(".tif")
            else:
                outname = outname.with_suffix(".png")

            outname = output_folder_path / outname
            save_image(img, outname, output_dtype)

        next_padding_plane = current_substack[-1]

        if canceled.is_set():
            return
