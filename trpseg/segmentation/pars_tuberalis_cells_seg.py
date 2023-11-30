"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a classical segmentation pipeline to segment and count pars tuberalis cells from 16bit images.
# The segmentation consists of pre-processing, global thresholding, post-processing
# For cell counting several methods can be used:
# 1. watershed_split_memeff(...) does perform watershed transformation to split clustered cells and count them
#    This method is only useful if most cells are relatively well isolated and there are only clusters of 2-3 cells.
#    Also, the watershed algorithm assumes the cells have a roundish shape
# 2. trpseg_utility count_objects_3D_memeff(...) does count directly how many objects there are in a binary image.
#    This method does not split clustered or connected cells
# 3. trpseg_utility count_labels_3D_memeff(...) does count how many different labels there are in an already labeled
#    image stack.
# 4. trpseg_utility estimateParsCellCounts does approximate the cell count by dividing the whole segmentation volume
#    by the average pars tuberalis cell size
########################################################################################################################

"""
In more detail: Cells that expressed a specific TRP channel were marked with GFP (genetic labeling + immunofluorescence).
The additional Antibody labeling against GFP is used to amplify the signal.
In the end Light-Sheet Flourescence Microscopy (LSFM) is used to acquire the images.

A classical segmentation pipeline is used. Where the main idea is to do:
1. Median Filtering
2. Global Thresholding
3. Remove objects too far from outer boundary
4. Remove small objects
5. Close small holes
6. (optional Watershed Transform) + Cell counting
"""

import os
import shutil
from threading import Event
import time

import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from scipy import ndimage
import igraph as ig
from skimage.morphology import disk
from skimage.filters import median
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max


from trpseg.trpseg_util.utility import read_image, save_image, get_z_splits,\
     read_img_list, closing3D, anisotropic_spherical_3D_SE, get_z_radius_from_footprint,\
     remove_small_objects_3D_memeff, get_file_list_from_directory,\
     save_img_stack_slices, get_img_shape_from_filepath, count_segmentation_pixels,\
     get_max_images_in_memory_from_img_shape

from trpseg.trpseg_util.brain_region_segmentation import segment_brain_region
from trpseg.trpseg_util.chamfer_dt import ch_distance_transform_files_memeff



#max_dist is physical distance
def new_pars_seg_opti(input_folder_path, distance_folder_path, output_folder_path, median_radius, threshold, remove_by_distance, max_dist, isTif=True, channel_one=True, canceled=Event()):
    """Perform pars tuberalis cells segmentation.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack form which pars tuberalis cells should be segmented.
    distance_folder_path : str, pathlib.Path
        The path to the folder containing the image transformation for the stack.
        It should specify for every voxel the physical distance to the brain boundary.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    median_radius : int
        The radius of the median filter that will be applied to the 2D image slices.
        The radius is measured in pixels.
    threshold : int, float
        The global threshold that will be applied to the median filtered images to segment the pars tuberalis cells.
    remove_by_distance : bool
        Determines whether segmented regions that are farther away than max_dist from the brain boundary will be removed.
    max_dist : int, float
        The maximal allowed distance for pars tuberalis cells from the brain boundary in phyisical units (micrometers).
        The distance of a pars tuberalis cell to the brain boundary is measured from the pars tuberalis cell pixel which is closest to the brain boundary
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

    Notes
    ----------
    This is the currently used method to segment pars tuberalis cells. The main idea is to do:
    1. Median Smoothing
    2. Global thresholding
    3. Remove objects too far away from outer brain boundary (Tanycytes signal + Noise) -> uses distance transformation
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    numFiles = len(input_files_list)

    if numFiles == 0:
        raise RuntimeError(f"No files named *_C01_*.tif found in {input_folder_path}")

    if remove_by_distance and max_dist == None:
        raise RuntimeError("You have to specify a max_dist if you set remove_by_distance to True")

    if remove_by_distance:
        distance_input_files_list = get_file_list_from_directory(distance_folder_path, isTif=False, channel_one=None)

        numDistanceFiles = len(distance_input_files_list)

        if numDistanceFiles != numFiles:
            raise Exception("The number of distance transfrom files does not equal the number of input files!\n Distance Transform files have to be png.")

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    # Important parameters
    global_threshold = threshold
    medianDiskRadius = median_radius

    for i in range(0, numFiles):

        if canceled.is_set():
            return

        print(" Process Image ", i)
        img = read_image(input_files_list[i])
        median_img = median(img, disk(medianDiskRadius))

        th_img = (median_img > global_threshold) * 1
        th_img = th_img.astype(np.uint8)

        if remove_by_distance:
            th_img = remove_distant_objects(th_img, distance_input_files_list[i], max_dist)

        th_img = th_img * 255

        file_out_name = Path(input_files_list[i].name)
        file_out_name = file_out_name.with_suffix(".png")

        out_path = output_folder / file_out_name

        save_image(th_img, out_path, np.uint8)



def remove_distant_objects(th_img, distance_img_path, max_dist):
    """Remove objects that have a distance larger than max_dist in the given distance transformation.

    This method is used to remove segmented objects that are too far away from the brain boundary.

    Parameters
    ----------
    th_img : np.ndarray
        The thresholded binary 2D image. Pars tuberalis cells have pixel value 1 and other 0.
    distance_img_path : str, pathlib.Path
        The path to the distance transformation for the currently considered 2D image slice.
    max_dist : int, float
        The maximal allowed distance for pars tuberalis cells from the brain boundary in physical units (micrometers).
        The distance of a pars tuberalis cell to the brain boundary is measured from the pars tuberalis cell pixel which is closest to the brain boundary.


    Returns
    -------
    th_img : np.ndarray
        The 2D binary image slice were objects that have too high distance in the given distance transform image have been removed.
    """

    pars_labels = label(th_img, background=0)
    pars_regions = regionprops(pars_labels)

    dist_img = read_image(distance_img_path)
    dist_img = dist_img.astype(np.uint16)

    for p_reg in pars_regions:
        slice = p_reg.slice
        dist = np.min(dist_img[slice])

        #Version to get the distance from the outer boundary to the centroid of regions
        #This is slower than the above where the minimal distance of a region to the outer boundary is determined!
        #centroid = p_reg.centroid
        #y = int(centroid[0])
        #x = int(centroid[1])
        #dist = dist_img[y, x]

        if dist > max_dist:
            slice = p_reg.slice

            roi = th_img[slice] #This returns a view/reference of the th_img region
            binary_roi = p_reg.image

            roi[binary_roi] = 0 #This changes the values in the th_img ndarray

    return th_img


def get_default_final_pseg_output_name():
    """Default folder name for the final output."""
    return "PSEG_final_output"

def get_default_brain_tissue_sigma():
    """The default sigma (standard deviation) of the gaussian filter that is applied
    to the image before brain tissue detection."""
    return 14

def get_median_radius_from_resolution(resolution):
    """Get the default radius for the median filter which is used as preprocessing step before pars tuberalis cells segmentation.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    The default radius for the preprocessing median filter.

    Notes
    ----------
    This method returns a discrete integer (the radius in pixels).
    Experiments showed a radius of 0.64 micrometers is appropriate.
    Median radius of 2 for x-y resolution of 0.325 and 0.5416 works well.
    """

    xy_res = resolution[1]
    return int(np.ceil(0.64/xy_res))


def get_num_voxels_from_physical_volume(resolution, physical_volume):
    """

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    physical_volume : int, float
        The volume in micrometers^3

    Returns
    -------
    voxel_num : int
        The rounded down voxel number that has approximately the given physical_volume.
    """
    voxel_volume = resolution[0] * resolution[1] * resolution[2]
    voxel_num = int(physical_volume // voxel_volume)

    return voxel_num

# We assume pars tuberalis cells to be larger than 422 micrometer^3 (from experiments)
# Choose a bit smaller min_size to be sure no pars tuberalis cells are removed (50 micrometer^3 tolerance)
def get_default_min_pars_cell_size():
    """Get the default minimal physical volume size of pars tuberalis cells.

    Notes
    ----------
    We assume pars tuberalis cells to be larger than 422 micrometer^3
    Return a bit smaller value than 422 to be sure remove_small_objects_3D does not remove pars tuberalis cells (50 micrometer^3 tolerance)
    """

    return 372


def get_default_avg_pars_cell_size():
    """Get the default average size of a pars tuberalis cell."""
    return 550


def get_remove_smaller_z_length_from_resolution(resolution):
    """Get a default value for remove_smaller_z_length variable.
    Segmented objects that span over less z slices than returned here will be removed.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    In tests pars tuberalis cells had a minimal z width of 6 micrometers.
    Divide 6 micrometers by resolution in z direction to get z length in pixels.
    If a segmented region expands over less than this minimal z length pixels then it is removed.
    """

    z_res = resolution[0]

    return int(np.floor(6.0/z_res))


def get_closing_radius_from_resolution(resolution):
    """Get the radius for the 3D morphological closing operation used in post-processing.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    IMPORTANT!!!: the closing radius will be a single integer that describes the radius in xy-plane.
    Radius in z-direction needs to be determined individually if anisotropic.
    This will be done in this code in method anisotropic_spherical_3D_SE(...)

    In tests a closing radius of 3px worked well for images with xy_res 0.5416
    For images with xy_res 0.325 a closing radius of 5px worked well
    -> closing_radius = np.floor(1.626/xy_res)
    """

    xy_res = resolution[1]

    return int(np.floor(1.626/xy_res))


def segment_pars_tuberalis_cells(input_folder_path, output_folder_path, resolution, threshold, remove_by_distance,
                                 brain_region_threshold=None, remove_smaller_than=None, remove_smaller_z_length=None,
                                 max_dist=None, brain_region_sigma=None,
                                 brain_channel_one=False, median_radius=None, closing_radius=None,
                                 delete_intermed_results=True, distance_folder_path=None, prepend_to_folder_name=None,
                                 max_images_in_memory=None, canceled=Event(), progress_status=None):

    """Execute the pipeline to segment pars tuberalis cells from the 3D image stack given in slices in input_folder_path.

    This pipeline includes pre- and post-processing operations.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the 3D image stack slices. The slices are assumed to be tif files from two channels.
        Images from channel 0 have "_C00_" in their name and images from channel 1 have "_C01_" in their name.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored. Subfolders will be created in this folder.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    threshold : int, float
        The global threshold that will be applied to the median filtered images to segment the pars tuberalis cells.
    remove_by_distance : bool
        Determines, whether to remove segmented objects that are too far away from the outer brain boundary.
    brain_region_threshold : int, float
        The global threshold for brain tissue detection. Pixels in the smoothed channel0 images with a value greater than threshold
        are set to 255, others to 0.
        If remove_by_distance is True then a value for brain_region_threshold has to be given. Else, an exception is raised.
    remove_smaller_than : int, float, optional
        The minimal size for segmented objects given in micrometers^3. Smaller objects are removed from the result.
        If None, then a default value will automatically be assigned.
    remove_smaller_z_length : int, optional
        Segmented objects that span over less z slices than returned here will be removed.
        This parameter is given in pixels in z direction (=number of z slices)!
        If None, then a default value will automatically be determined.
    max_dist : int, float, optional
        The maximal allowed distance for pars tuberalis cells from the brain boundary in phyisical units (micrometers).
        The distance of a pars tuberalis cell to the brain boundary is measured from the pars tuberalis cell pixel which is closest to the brain boundary.
        If max_dist is None and remove_by_distance is True then an Exception is raised.
    brain_region_sigma : int, float, optional
        The sigma parameter (standard deviation) of the gaussian filter that is applied to images for brain tissue detection.
        If None, then a default value will automatically be assigned.
    brain_channel_one : bool
        Determines whether to use images from channel 0 or images from channel 1 for brain tissue detection.
        If True, use channel 1 images, else use channel 0 images.
    median_radius : int, optional
        The radius of the median filter that will be applied to the 2D image slices of channel 1.
        The radius is measured in pixels.
        If None, then a default value will automatically be determined from the resolution.
    closing_radius : int, optional
        The radius of the ball-like structuring element in pixels in the xy-plane for the 3D morphological closing post-processing step.
        If None, then a default value will automatically be determined from the resolution.
    delete_intermed_results : bool
        Determines whether to delete intermediate images that are written to disk during processing.
        If True, then the brain tissue segmentation images, the non-postprocessed output images after pars_seg_opti(...)
        and the images where small objects have been removed but no closing was performed yet will be deleted.
        This means only the distance transformation as well as the final completely post-processed segmentation results will be stored.
    distance_folder_path : str, pathlib.Path, optional
        If a distance transformation specifying the distance of every voxel from the outer brain boundary has already
        been computed before, then put the path to the folder where it is stored here.
        This saves time, since then the distance transformation will not be computed here again.
    prepend_to_folder_name : str
        A string that will be prepended to the folders that will be created inside the main output folder (output_folder_path)
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    """

    #Initialize Parameters
    if max_images_in_memory is None:
        file_paths_help = get_file_list_from_directory(input_folder_path, isTif=True)
        shape_help = get_img_shape_from_filepath(file_paths_help[0])
        max_images_in_memory = get_max_images_in_memory_from_img_shape(shape_help)

    if prepend_to_folder_name is None:
        prepend_to_folder_name = time.strftime("%H%M%S_")

    if brain_region_sigma is None:
        brain_region_sigma = get_default_brain_tissue_sigma()

    if median_radius is None:
        median_radius = get_median_radius_from_resolution(resolution)

    if remove_smaller_than is None:
        remove_smaller_than = get_default_min_pars_cell_size()

    min_pars_voxel_num = get_num_voxels_from_physical_volume(resolution, remove_smaller_than)

    if remove_smaller_z_length is None:
        remove_smaller_z_length = get_remove_smaller_z_length_from_resolution(resolution)

    if closing_radius is None:
        closing_radius = get_closing_radius_from_resolution(resolution)

    if brain_region_threshold is None and remove_by_distance:
        raise RuntimeError("You have to specify a brain tissue threshold if you want to remove pars tuberalis cells by distance!")

    if max_dist is None and remove_by_distance:
        raise RuntimeError("You have to specify a distance if you want to remove pars tuberalis cells by distance!")


    progress = 0

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    #Initialize output folder names
    out_folder_name1 = prepend_to_folder_name + "PSEG_brain_region"
    output_folder_outer_tmp = output_folder / out_folder_name1
    out_folder_name2 = prepend_to_folder_name + "PSEG_raw_segmentation_result"
    output_folder_path_tmp = output_folder / out_folder_name2
    out_folder_name3 = prepend_to_folder_name + "PSEG_removedSmallRegions"
    output_folder_remsmall_tmp = output_folder / out_folder_name3
    out_folder_name4 = prepend_to_folder_name + get_default_final_pseg_output_name()
    final_output_folder_path = output_folder / out_folder_name4

    #Start segmentation pipeline
    start = timer()
    #Compute approximate distance transformation and store as 16bit pngs
    if distance_folder_path == None and remove_by_distance:
        progress = 2
        if progress_status is not None:
            progress_status.emit(["Segmenting brain tissue...", progress])

        out_folder_name5 = prepend_to_folder_name + "PSEG_distance_transform"
        distance_folder_path = output_folder / out_folder_name5
        segment_brain_region(input_folder_path, output_folder_outer_tmp, brain_region_sigma, brain_region_threshold, channel_one=brain_channel_one, canceled=canceled)

        if canceled.is_set():
            return

        progress += 5
        if progress_status is not None:
            progress_status.emit(["Computing distance to brain boundary...", progress])


        ch_distance_transform_files_memeff(output_folder_outer_tmp, distance_folder_path,
                                                resolution=resolution, isTif=None, channel_one=None,
                                                canceled=canceled)

        if delete_intermed_results:
            shutil.rmtree(output_folder_outer_tmp)

        if canceled.is_set():
            return

    end = timer()

    print(f"Distance Transform took {end-start} seconds.")

    if canceled.is_set():
        return

    progress = 10
    if progress_status is not None:
        progress_status.emit(["Segmenting pars tuberalis cells...", progress])

    start = timer()
    new_pars_seg_opti(input_folder_path, distance_folder_path, output_folder_path_tmp, median_radius, threshold, remove_by_distance, max_dist, canceled=canceled)
    end = timer()
    print(f"Pars Tuberalis Cell segmentation took {end - start} seconds.")

    if canceled.is_set():
        return

    progress = 70
    if progress_status is not None:
        progress_status.emit(["Removing too small regions...", progress])

    start = timer()
    remove_small_objects_3D_memeff(output_folder_path_tmp, output_folder_remsmall_tmp, min_pars_voxel_num,
                                   remove_smaller_z_length, max_images_in_memory=max_images_in_memory, canceled=canceled)
    if delete_intermed_results:
        shutil.rmtree(output_folder_path_tmp)
    end = timer()
    print(f"Remove small regions took {end - start} seconds.")

    if canceled.is_set():
        return

    progress = 80
    if progress_status is not None:
        progress_status.emit(["Closing small holes...", progress])

    start = timer()
    postprocess_pars_seg(output_folder_remsmall_tmp, final_output_folder_path, resolution, closing_radius=closing_radius, canceled=canceled)
    if delete_intermed_results:
        shutil.rmtree(output_folder_remsmall_tmp)
    end = timer()
    print(f"Closing3D took {end - start} seconds.")

    if canceled.is_set():
        return

    progress = 100
    if progress_status is not None:
        progress_status.emit(["Finished...", progress])


def postprocess_pars_seg(input_folder_path, output_folder_path, resolution, closing_radius, canceled=Event()):
    """Do 3D binary morphological closing to fill small holes and smooth the segmentation a bit.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the 2D binary image slices.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    closing_radius : int
        The radius of the ball-like structuring element in pixels in the xy-plane for the 3D morphological closing postprocessing step.
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    """

    footprint = anisotropic_spherical_3D_SE(closing_radius, resolution)
    overlap = get_z_radius_from_footprint(footprint) + 1
    closing3D(input_folder_path, output_folder_path, footprint, overlap, canceled=canceled)


def watershed_split_memeff(input_folder_path, output_folder_path, resolution, max_images_in_memory=None,
                           out_dtype=np.uint16, isTif=False, channel_one=True, canceled=Event(), progress_status=None):
    """Perform memory efficient watershed transformation to generate a labeled image stack in which touching pars tuberalis cells are splitted.

    This method can be used to count the approximate number of pars tuberalis cells. Assuming cells have a roundish shape.

    IMPORTANT: Good results can only be obtained if cells have a roundish shape and are not clumped together in too large clusters!!!

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary 3D image stack slices from the pars tuberalis cell segmentation pipeline.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once.
    out_dtype : np.dtype.type
        The datatype for the outputted labeled images.
        If np.uint16: Then no more than 2^16 == 65535 objects can be labeled. Else, overflow will happen and labeling restarts at 0.
        If np.uint32: Then 2^32-1=4294967295 objects can be labeled.
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


    Returns
    -------
    If process gets canceled: return None
    Else: 3-tuple (new_num_labels, max_label, list(set(new_labels))
          new_num_label: the number of different labels (==number of pars tuberalis cells) in the outputted labeled image stack
          max_label: the maximal assigned label in the labeled image stack
                     (IMPORTANT: not all labels up to max_label have to be used. There can be labels missing that are smaller than max_label)
          list(set(new_labels)): list of int which are the labels that have been assigned to different objects

    Notes
    ----------
    If you have enough RAM availabe we recommend using the method watershed_split_whole instead of watershed_split_memeff.
    watershed_split_memeff() processes the 3D image stack in blocks which allows to handle image stacks which are too large to completely fit in RAM.

    Problem: Pars Tuberalis regions could get split if they lie at the boundary between two image blocks
             We try to mitigate those boundary artefact by merging splitted regions at boundaries according to certain criteria
             However, this merging is not really exact. It works as follows: If two objects are 6-connected across the boundary then merge them if
             one of them is smaller than the minimal pars tuberalis cell size.
    """

    progress = 0

    if progress_status is not None:
        progress_status.emit(["Initializing watershed...", progress])

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)
    num_images = len(input_files_list)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    if num_images < 2:
        raise RuntimeError("There are less than 2 images in your specified input folder but at least 2 are needed!")


    output_list_files = []
    for i in range(num_images):
        out_name = Path(input_files_list[i].name)
        if np.dtype(out_dtype).itemsize > 2:
            out_name = out_name.with_suffix(".tif")
        else:
            out_name = out_name.with_suffix(".png")

        out_path = output_folder / out_name
        output_list_files.append(out_path)

    use_compression = True if np.dtype(out_dtype).itemsize > 2 else False # Use compression for outputted tif images if datatyp is larger than 2 Bytes

    img_height, img_width = get_img_shape_from_filepath(input_files_list[0])


    if max_images_in_memory is None:
        max_images_in_memory = get_max_images_in_memory_from_img_shape((img_height, img_width), for_watershed_split=True)


    splits = get_z_splits(num_images, max_images_in_memory, 0)
    num_splits = len(splits)

    footprint = get_footprint_for_peak_local_max(resolution)
    struct = ndimage.generate_binary_structure(3, 1)

    current_max_label = 0
    label_sizes = dict()

    for s in range(num_splits):

        if canceled.is_set():
            return

        progress = int((s/num_splits)*100)

        if progress_status is not None:
            progress_status.emit(["Splitting cells...", progress])

        start = splits[s][0]
        end = splits[s][1]

        curr_files = input_files_list[start:end]

        imgStack = read_img_list(curr_files, dtype=np.uint8)

        distance = ndimage.distance_transform_edt(imgStack, sampling=resolution)

        if canceled.is_set():
            return

        max_coords = peak_local_max(distance, labels=imgStack, footprint=footprint, exclude_border=False)
        local_maxima = np.zeros_like(imgStack, dtype=bool)
        local_maxima[tuple(max_coords.T)] = True

        if canceled.is_set():
            return

        markers = ndimage.label(local_maxima, structure=struct)[0]

        if canceled.is_set():
            return

        labels = watershed(-distance, markers, mask=imgStack)

        if canceled.is_set():
            return

        distance = 0  # for easier garbage collection

        labels[labels != 0] += current_max_label

        values, counts = np.unique(labels, return_counts=True)
        current_max_label += len(values)-1 #-1 because we dont want to count the background label (value 0)

        # Store region sizes
        for j in range(len(values)):
            label_l = values[j]
            if label_l in label_sizes:
                label_sizes[label_l] += counts[j]
            else:
                label_sizes[label_l] = counts[j]

        save_img_stack_slices(labels, output_list_files[start:end], out_dtype, compression=use_compression)


    # Resolve problems at the boundaries between blocks
    #Merge
    label_connections = set() #store tuples of form (label0, label1) whith 6-connected labels

    for _, end in splits:
        if end != num_images:
            # load both z-slices where the split happens number_max_images_in_memory
            img_below = read_image(output_list_files[end - 1])
            img_above = read_image(output_list_files[end])

            object_regions_below = img_below != 0
            object_regions_above = img_above != 0

            six_connected = np.logical_and(object_regions_below, object_regions_above)

            # get connected label values
            labels_conn_below = img_below[six_connected]
            labels_conn_above = img_above[six_connected]

            conn_labels = zip(labels_conn_below, labels_conn_above)

            #unique_connections, num_connections = np.unique(conn_labels, return_counts=True)
            #unique_connections = zip(unique_connections, num_connections)

            label_connections.update(conn_labels)


    # Determine which labels to merge
    # We merge two labels if one of them has a size smaller than the minimal pars tuberalis cell size

    min_voxel_num = get_num_voxels_from_physical_volume(resolution, get_default_min_pars_cell_size())

    for conn in label_connections.copy():
        l0 = conn[0]
        l1 = conn[1]

        if label_sizes[l0] < min_voxel_num or label_sizes[l1] < min_voxel_num:  #if one object is smaller than minimum pars tuberalis cell size then merge
            continue
        else:
            label_connections.remove(conn)  # else dont merge them


    # Get sets of labels that are connected and need to be merged to one label -> connected components
    #If we have edges (1, 2) and (2, 3) then merge (1,2,3) to label 1 (the first label in this set)
    merge_labels = []
    keys = []
    values = []
    old_num_labels = current_max_label

    g = ig.Graph(edges=list(label_connections), directed=False)  # adds also vertices for not existing labels. Graph always consists of vertices from 0 to max label

    at_least_one_edge_vertices_ids = g.vs.select(_degree_gt=0).indices  #select only vertices that have edges -> our labels to merge
    at_least_one_edge_vertices_ids = np.asarray(at_least_one_edge_vertices_ids)

    g_smaller = g.induced_subgraph(at_least_one_edge_vertices_ids)

    vertex_clustering = ig.Graph.connected_components(g_smaller)

    for connected_labels in vertex_clustering:
        connected_labels = at_least_one_edge_vertices_ids[connected_labels]  #get the id in the original graph g
        merge_labels.append(connected_labels)


    # create mapping of labels to new_labels (the first label in a connected label will be assigned to all connected labels)
    for m_labels in merge_labels:
        map_these = m_labels[1:]
        value = m_labels[0]

        for m_t in map_these:
            keys.append(m_t)
            values.append(value)

    new_labels = np.arange(0, old_num_labels + 1, dtype=out_dtype)
    new_labels[keys] = values

    max_label = np.max(new_labels)

    new_num_labels = old_num_labels - len(keys)

    # Now merge the labels: Go through the images and set the new labels
    for i in range(num_images):

        if canceled.is_set():
            return

        img = read_image(output_list_files[i])
        img = img.ravel()  # represent as 1d to efficiently change the labels
        img = new_labels[img]  # change to new labels
        img = np.reshape(img, (img_height, img_width))

        save_image(img, output_list_files[i], out_dtype, compression=use_compression)

    progress = 100

    if progress_status is not None:
        progress_status.emit(["Splitting cells...", progress])

    return new_num_labels, max_label, list(set(new_labels))



def watershed_split_whole(input_folder_path, output_folder_path, resolution, out_dtype=np.uint16, isTif=False, channel_one=None):
    """Perform watershed transformation to generate a labeled image stack in which touching pars tuberalis cells are splitted.

    This method can be used to count the approximate number of pars tuberalis cells. Assuming cells have a roundish shape.

    IMPORTANT: Good results can only be obtained if cells have a roundish shape and do not lie together in too large clusters!!!

    Parameters
    ----------

    input_folder_path : str, pathlib.Path
        The path to the folder containing the binary 3D image stack slices from the pars tuberalis cell segmentation pipeline.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    out_dtype : np.dtype.type
        The datatype for the outputted labeled images.
        If np.uint16: Then no more than 2^16 == 65535 objects can be counted.
        If np.uint32: Then 4294967295 objects can be counted.
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
    num_regions : int
        The determined number of unique labels in the labeled image stack (= the number of pars tuberalis cells)
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    imgStack = read_img_list(input_files_list, dtype=np.uint8)

    distance = ndimage.distance_transform_edt(imgStack, sampling=resolution)
    distance = distance.astype(np.uint16)

    footprint = get_footprint_for_peak_local_max(resolution)

    max_coords = peak_local_max(distance, labels=imgStack, footprint=footprint, exclude_border=False)
    local_maxima = np.zeros_like(imgStack, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True

    struct = ndimage.generate_binary_structure(3, 2)  # 3D 2-connected binary structure -> 18 connectivity

    # ndimage.label uses 64bit if input array has more than 2 ** 31 - 1 elements else 32bit is used
    # we need only 32bit since we assume there are not more than 2^31 - 1 pars tuberalis cells.
    markers = ndimage.label(local_maxima, structure=struct)[0]  # ndimage label returns tuple (labeled_array, num_features) -> take 0
    markers = markers.astype(np.uint32)
    labels = watershed(-distance, markers, mask=imgStack)

    for i in range(len(labels)):
        out_name = Path(input_files_list[i].name)
        out_name = out_name.with_suffix(".tif")
        outpath = output_folder / out_name
        save_image(labels[i], outpath, np.uint16)

    num_regions = len(max_coords)

    return num_regions


def get_footprint_for_peak_local_max(resolution):
    """Get an anisotropic cubic footprint for skimage.feature.peak_local_max that works well for determining the pars tuberalis cells
     from a distance transformed pars tuberalis cells segmentation.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    footprint: np.ndarray
        The anisotropic cubic footprint that is appropriate for skimage.feature.peak_local_max computations on distance
        transformed pars tuberalis cells segmentation result.

    Notes
    ----------
    Divides 9 micrometers by resolution and round up to the nearest odd integer -> value of 9 micrometers worked well in experiments
    e.g. if we have resolution (1.0, 0.54166,0.54166) we get 9 pixels (z)| 17 pixels footprint (yx)
    if we have resolution (2.0, 0.325,0.325) we get 5 pixels (z)| 27 pixels (yx)
    """



    ndim = len(resolution)
    z_footprint_width = 9.0 / resolution[0]
    xy_footprint_width = 9.0 / resolution[1]

    #Round to closest odd integer
    z_footprint_width = (z_footprint_width // 2) * 2 + 1
    xy_footprint_width = (xy_footprint_width // 2) * 2 + 1

    z_footprint_width = int(z_footprint_width)
    xy_footprint_width = int(xy_footprint_width)

    if z_footprint_width == xy_footprint_width:
        footprint = np.ones((xy_footprint_width,) * ndim, dtype=bool)

    elif z_footprint_width < xy_footprint_width:
        footprint = np.ones((xy_footprint_width,) * ndim, dtype=bool)
        diff = xy_footprint_width - z_footprint_width
        diff = diff // 2
        footprint[0:diff, :, :] = False
        footprint[-diff:, :, :] = False

    else:
        footprint = np.ones((z_footprint_width,) * ndim, dtype=bool)
        diff = z_footprint_width - xy_footprint_width
        diff = diff // 2
        footprint[:, 0:diff, :] = False
        footprint[:, -diff:, :] = False
        footprint[:, :, 0:diff] = False
        footprint[:, :, -diff:] = False

    return footprint


def estimateParsCellCounts(input_folder_path, resolution, average_cell_size, isTif=None, channel_one=True, canceled=Event(), progress_status=None, maxProgress=90): #average_cell_size e.g. 550
    """Estimate the pars tuberalis cells number by dividing the total cell volume of a given pars tuberalis cell segmention by the average cell size.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y and x direction in order [z_resolution, y_resolution, x_resolution]
    average_cell_size : int, float
        The average size of a pars tuberalis cell in micrometers^3
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
    cell_count : int
        The estimated number of pars tuberalis cells in the provided segmentation.
    """

    progress = 0
    if progress_status is not None:
        progress_status.emit(["Estimating Cell Count...", progress])

    voxel_count = count_segmentation_pixels(input_folder_path, isTif=isTif, channel_one=channel_one, canceled=canceled, progress_status=progress_status, maxProgress=maxProgress)

    if canceled.is_set():
        return

    cell_volume = voxel_count * resolution[0] * resolution[1] * resolution[2]

    cell_count = cell_volume/average_cell_size

    return cell_count
