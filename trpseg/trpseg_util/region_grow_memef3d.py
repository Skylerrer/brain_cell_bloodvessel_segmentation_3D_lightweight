"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a 3D memory efficient region growing algorithm.
# It can e.g. be use to segment high signal artefacts that you want to remove from the original image stacks.
# Methods like substract_mask or substract_multiple_masks from trpseg_utility/utility.py can be used to remove the
# segmented artefacts.
########################################################################################################################

import os
from queue import Queue, PriorityQueue

import numpy as np
from pathlib import Path

from trpseg.trpseg_util.utility import read_image, save_image, get_neighbors_2D_with_z, get_file_list_from_directory


# A Queue that only adds items if not already in queue.
# -> only contains unique items
class UniqueQueue(Queue):
    def _init(self, maxsize):
        Queue._init(self, maxsize)
        self.values = set()

    def _put(self, item):
        if item not in self.values:
            self.values.add(item)
            Queue._put(self, item)

    def _get(self):
        item = Queue._get(self)
        self.values.remove(item)
        return item


# A PriorityQueue that only adds items if not already in PriorityQueue.
# -> contains only unique items
class UniquePriorityQueue(PriorityQueue):
    def _init(self, maxsize):
        PriorityQueue._init(self, maxsize)
        self.values = set()

    def _put(self, item):
        if item[1] not in self.values:
            self.values.add(item[1])
            PriorityQueue._put(self, item)

    def _get(self):
        item = PriorityQueue._get(self)
        self.values.remove(item[1])
        return item[1]


#Grow from possibly several seed points and return 2D coordinates
# In the already_seen set 3D coordinates are stored
def region_growing_2D(img, seed_points, min_val, max_val, already_seen, max_area=1000000):
    """ Do region growing in 2D from seedpoints. Look at all 8 neighbors and add them to region
    if they have a value in the interval [min_val, max_val].

    Although region growing is performed here in 2D, 3D coordinates are used!

    Parameters
    ----------
    img : np.ndarray
        2D image on which region growing should be performed.
    seed_points : list of 3-tuples
        The list of coordinates of seed points in order (z, y, x)
    min_val : int, float
        The minimal value for a pixel to be added to the growing region.
    max_val : int, float
        The maximal value for a pixel to be added to the growing region.
    already_seen : set
        A set that stores which pixels have already been seen. It stores 3D coordinates (3-tuples)
    max_area : int
        When the region grows over this number of pixels an exception is raised because to many pixels in region.

    Returns
    -------
    filled : list of 2-tuple
        The 2D coordinates of the grown region.

    Notes
    -------
    3D seed_points are used, so that 3D coordinates can be set in already_seen set and this method can
    be used by other methods that do region growing in 3D.
    """

    if len(seed_points) > 0:
        z = seed_points[0][0]
    else:
        return np.array([], dtype=np.uint16)

    neighborhoodQueue = UniqueQueue()
    filled = []
    numFilled = 0

    for seed_point in seed_points:
        neighborhoodQueue.put(seed_point, block=False)

    while not neighborhoodQueue.empty():
        if numFilled > max_area:
            raise Exception("Stopped region growing early because pixel limit reached!!!")

        currentCoords = neighborhoodQueue.get(block=False)
        neighbors = get_neighbors_2D_with_z(currentCoords, exclude_center=False)
        for neighbor in neighbors:
            if neighbor not in already_seen:
                neighbor_2d = neighbor[1:]
                if min_val <= img[neighbor_2d] <= max_val:
                    filled.append(neighbor_2d)
                    numFilled += 1
                    neighborhoodQueue.put(neighbor, block=False)
                already_seen.add((z, neighbor_2d[0], neighbor_2d[1]))

    return filled


# seed_points is list of 3D points in form [(z, y, x), ...]
# However only seed points from the same z slice are allowed!!!
# Region growing done in each image in 2D and then seedpoints are propagated to neighboring slices (6-connected in z direction)
def region_growing_3D_memory_efficient(input_folder_path, output_folder_path, seed_points, min_val, max_val=65535, z_min=0, z_max=10000, max_2d_area=1000000, isTif=True, channel_one=False):
    """ Perform region growing in 3D starting from seed points.
    Important: The seed points are given as 3D tuples however they have to be all from the same image (z slice)!

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the images on which region growing should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the binary output images are stored.
    seed_points : list of 3-tuples
        The list of coordinates of seed points in order (z, y, x)
    min_val : int, float
        The minimal value for a pixel to be added to the growing region.
    max_val : int, float
        The maximal value for a pixel to be added to the growing region.
    z_min : int
        The minimal z coordinate that is considered for region growing.
    z_max : int
        The maximal z coordinate that is considered for region growing.
    max_2d_area : int
        When the region in one 2D slice grows over this number of pixels an exception is raised because region grown too large
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
    Regions are first grown in 2D in one image slice. For region growing in 2D all 8 neighbors of a pixel are considered.
    The grown region is then propagated as seed points to neighboring z-slices (6-neighborhood) on which again region is grown in 2D.
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    num_images = len(input_files_list)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    #get z coordinate of seed_point (assume seed_point is in form zyx)
    first_z = seed_points[0][0]
    if not z_min <= first_z < z_max:
        raise RuntimeError("The given seed point is not applicable with the given z_min and z_max values.")

    #initialize variables that store pixels that are filled and pixels that have already been processed and have not to be looked at anymore
    filled_pixels_in_slices = [[] for _ in range(num_images)]
    already_seen = set()

    # initial region growth
    img = read_image(input_files_list[first_z])

    init_reg = region_growing_2D(img, seed_points, min_val, max_val, already_seen, max_area=max_2d_area)

    filled_pixels_in_slices[first_z] = filled_pixels_in_slices[first_z] + init_reg

    # queue that holds z slices that have to be considered for potential growth
    potential_z_growth = UniquePriorityQueue()
    z_up = first_z + 1
    z_down = first_z - 1

    if z_up < num_images and z_up <= z_max:
        prio = -1  # prio is -1* the distance to first_z
        potential_z_growth.put((prio, z_up))

    if z_down >= 0 and z_down >= z_min:
        prio = -1  # prio is -1* the distance to first_z
        potential_z_growth.put((prio, z_down))

    while not potential_z_growth.empty():

        current_z = potential_z_growth.get(block=False)

        seeds = []
        if current_z < num_images-1:
            seeds_up_2D = filled_pixels_in_slices[current_z+1]
            seeds_up_3D = [(current_z,) + t for t in seeds_up_2D]
            seeds = seeds + seeds_up_3D
        if current_z > 0:
            seeds_down_2D = filled_pixels_in_slices[current_z-1]
            seeds_down_3D = [(current_z,) + t for t in seeds_down_2D]
            seeds = seeds + seeds_down_3D

        seeds = set(seeds)
        seeds = seeds.difference(already_seen)
        seeds = list(seeds)

        if len(seeds) == 0:
            continue
        else:
            img = read_image(input_files_list[current_z])
            grown_reg = region_growing_2D(img, seeds, min_val, max_val, already_seen, max_area=max_2d_area)
            if len(grown_reg) == 0:
                continue
            filled_pixels_in_slices[current_z] = filled_pixels_in_slices[current_z] + grown_reg
            z_up = current_z + 1
            z_down = current_z - 1

            if z_up < num_images and z_up <= z_max:
                prio = -abs(first_z-z_up)  # prio is -1* the distance to first_z
                potential_z_growth.put((prio, z_up))

            if z_down >= 0 and z_down >= z_min:
                prio = -abs(first_z - z_down)  # prio is -1* the distance to first_z
                potential_z_growth.put((prio, z_down))

    write_filled_coords_to_masks(filled_pixels_in_slices, input_files_list, output_folder)


def write_filled_coords_to_masks(filled_pixels_in_slices, input_files_list, output_folder):
    """ Write the grown coordinates to images.

    Pixels that belong to the grown region get value 255 and other get value 0.

    Parameters
    ----------
    filled_pixels_in_slices : list of lists of 3-tuples
        This list consists of as many lists as the image stack that will be created has 2D z slices.
        The lists store which coordinates are set to 255 in the binary images that will be written.
    input_files_list : list of pathlib.path
        List of filepaths that is used to determine the names of the output files that will be written.
    output_folder_path : str, pathlib.Path
        The path to the folder where the binary output images are stored.
    """

    for z in range(0, len(filled_pixels_in_slices)):
        output_path = Path("artefact_" + input_files_list[z].name)
        output_path = output_path.with_suffix(".png")

        output_path = output_folder / output_path
        save_coords_as_mask(2048, 2048, filled_pixels_in_slices[z], output_path)


def save_coords_as_mask(width, height, coords, output_path, dtype=np.uint8):
    """Create 2D binary image where coordinates in coords are set to 255"""
    mask = np.zeros((height, width), dtype=np.uint8)

    #for pixel in coords:
        #mask[pixel] = 255

    mask[tuple(np.asarray(coords).T)] = 255

    save_image(mask, output_path, dtype)
