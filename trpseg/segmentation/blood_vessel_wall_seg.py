"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a classical segmentation pipeline to segment the blood vessel wall.
# It does not segment the lumen of the blood vessels.
########################################################################################################################

import os
import time
from threading import Event

import joblib
from pathlib import Path
from timeit import default_timer as timer
import numpy as np
from skimage.filters import median, threshold_local
from skimage.morphology import disk
from skimage.restoration import rolling_ball


from trpseg import DATA_DIRECTORY
from trpseg.trpseg_util.utility import read_image, save_image, get_file_list_from_directory,\
     remove_small_objects_3D_memeff, get_max_images_in_memory_from_img_shape, get_img_shape_from_filepath
from trpseg.trpseg_util.randomForest_seg import predictImagePixels



def get_default_min_signal_intensity():
    """Get the default minimal signal intensity.
    Pixels/Voxels with a intensity lower than that are considered to not belong to the object we want to segment.

    Notes
    ----------
    Appropriate value determined in experiments.
    """
    return 700


def get_default_high_confidence_threshold():
    """Get the default high confidence threshold.
    Pixels/Voxels with a intensity value larger than this are considered to belong to the object we want to segement.

    Notes
    ----------
    Appropriate value determined in experiments.
    """
    return 2000


def get_default_local_th_offset(resolution):
    """Get the offset for the local thresholding. The local thresholding operation looks at the local pixel neighborhoods to
    determine a threshold value for every pixel. This threshold value will be substracted by the offset here.

    Example: For a pixel a local threshold was determined as 700. Now the local offset is substracted: 700 - (-90) = 790
             and the local threshold now is 790.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    Appropriate values determined in experiments.
    """

    if resolution[1] > 0.35:
        return -60
    else:
        return -90


#Experminenting showed that for a1/c2/c4/ML3 image stacks values between 550 and 850 work relatively well. 600 is okay for most of them
def get_default_rball_th(resolution):
    """Get the default global threshold value that will be used on the rolling ball filtered images.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    Appropriate values determined in experiments.
    """

    if resolution[1] > 0.35:
        return 300
    else:
        return 600


def get_default_border_rf_model_path(resolution):
    """Get the default path to the random forest (RF) model used to determine unwanted parts in the blood vessel segmentation.
    The RF model creates a very coarse segmentation for the blood vessels that ideally excludes unwanted parts that are present in other
    segmentations.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    """

    return os.path.join(DATA_DIRECTORY, r"border_forest_main.joblib")


def get_default_median_radius(resolution):
    """Get the default radius for the disk of the median filter applied to the original image prior to local thresholding.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    This method returns a discrete integer (the radius in pixels).
    Experiments showed a radius of 1.4 micrometers is appropriate.
    """

    xy_res = resolution[1]
    radius = int(1.4/xy_res)
    if radius < 1:
        radius = 1
    return radius


def get_default_local_th_blocksize(resolution):
    """Get the default blocksize of the local threshold. The local threshold method will look for every pixel/voxel at
    a rectangular neighborhood region of size blocksize x blocksize to determine the threshold.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    This method returns a discrete odd integer specifying the blocksize in pixels.
    Experiments showed a blocksize of 5.525 micrometers is appropriate.
    """

    xy_res = resolution[1]
    block_size = 2 * np.floor((5.525/xy_res)/2) + 1  # Get nearest odd integer
    return int(block_size)


def get_default_rball_radius(resolution):
    """Get the default radius for the rolling ball filter.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    This method returns a discrete integer (the radius in pixels).
    Experiments showed a radius of 4.875 micrometers is appropriate.
    """

    xy_res = resolution[1]
    rball_radius = int(4.875/xy_res)
    if rball_radius < 8:
        rball_radius = 8
    return rball_radius


def get_default_minimal_object_size(resolution):
    """Get the default minimal volume of objects that should be segmented.
    Objects with a smaller number of voxels than returned here will be removed in the final segmentation result.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    This method returns a discrete integer (the voxel number).
    Experiments showed a minimal volume of 212 micrometers^3 is appropriate.
    """

    voxel_volume = resolution[0] * resolution[1] * resolution[2]

    min_size = int(212.0/voxel_volume)
    if min_size < 1:
        min_size = 1
    return min_size


def get_default_minimal_z_length(resolution):
    """Get the default minimal length of blood vessels in z-direction in pixels
    Objects with a smaller length in z-direction than returned here will be removed in the final segmentation result.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------
    This method returns a discrete integer representing the minimal number of voxels needed in z-direction.
    Experiments showed a minimal length of 10 micrometers is appropriate.
    """

    z_res = resolution[0]

    return int(10.0/z_res)


def blood_vessel_wall_segmentation(input_folder_path, output_folder_path, resolution, min_signal_intensity=None,
                                   high_confidence_threshold=None, local_th_offset=None, rball_th=None,
                                   store_intermed_results=True, remove_border=True, border_rf_model_path=None,
                                   remove_small_objects=True, minimal_object_size=None, minimal_object_z_length=None,
                                   max_images_in_memory=None, prepend_to_folder_name=None, isTif=True,
                                   channel_one=False, canceled=Event(), progress_status=None):
    """
    Perform our blood vessel wall segmentation pipeline.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack from which blood vessels should be segmented
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    min_signal_intensity: int, float
        Pixels with a value lower than min_signal_intensity will not be segmented.
    high_confidence_threshold : int, float
        Pixels with a value larger than high_confidence_threshold will be interpreted as blood vessels and added to the segmentation
    local_th_offset : int, float
        The offset that will be substracted from the local thresholds determined by skimage.filters.threshold_local.
        See also method get_default_local_th_offset().
    rball_th :int, float
        The global threshold value that will be used on the rolling ball filtered images.
    store_intermed_results : bool
        Whether to store intermediate result images in addition to final result.
        Useful if you want to see which part of the pipeline contributes in which way.
    remove_border : bool
        Whether to use a random forest model to predict pixels that should not be segmented. They will be removed.
    border_rf_model_path : str, pathlib.Path
        The path to the random forest (RF) model file which was created from a trained RF model via joblib.dump
    remove_small_objects : bool
        Determines whether small objects should be removed from the segmentation.
    minimal_object_size : int
        The minimal size for segmented objects given as number of voxels.
        If remove_small_objects is True, then objects consisting of less than minimal_object_size voxels are removed from the segmentation.
    minimal_object_z_length : int
        The minimal length in z-direction for segmented objects given as number of voxels (== number of 2D image slices).
        If remove_small_objects is True, then objects spanning over less than minimal_object_z_length image slices are removed from the segmentation.
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    prepend_to_folder_name : str
        A string that will be prepended to the folders that will be created inside of the main output folder (output_folder_path)
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

    Notes
    ----------
    1. minimal signal threshold (pixels with value lower than this will not be segmented)
    2. high confidence threshold (pixels with a value larger than this will directly be added to the segmentation)
    3. local thresholding of median filtered image
    4. global thresholding of rball filtered gaussian smoothed image
    5. use random forest model to remove pixels with high signals that are probably not blood vessels
    6. remove too small objects from the segmentation
    """

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)


    numFiles = len(input_files_list)

    if numFiles < 2 and remove_small_objects:
        raise RuntimeError("There are less than 2 images in your specified input folder but at least 2 are needed!")

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)


    ############################################Initialize folders######################################################
    if prepend_to_folder_name is None:
        prepend_to_folder_name = time.strftime("%H%M%S_")

    # Determine which folder will contain the final output (append string "(final result)" to folder name)
    not_postprocessed_isFinalOutput = False
    border_removed_isFinalOutput = False
    remove_small_objects_isFinalOutput = False

    if remove_small_objects:
        remove_small_objects_isFinalOutput = True
    elif remove_border:
        border_removed_isFinalOutput = True
    else:
        not_postprocessed_isFinalOutput = True


    out_name1 = prepend_to_folder_name + "not_postprocessed"
    if not_postprocessed_isFinalOutput:
        out_name1 = out_name1 + "(finalOutput)"
    output_folder_not_postprocessed = output_folder / out_name1
    os.makedirs(output_folder_not_postprocessed, exist_ok=True)


    if store_intermed_results:
        out_name2 = prepend_to_folder_name + "localTh"
        output_folder_localTh = output_folder / out_name2
        os.makedirs(output_folder_localTh, exist_ok=True)

        out_name3 = prepend_to_folder_name + "rballTh"
        output_folder_rballTh = output_folder / out_name3
        os.makedirs(output_folder_rballTh, exist_ok=True)

    if remove_border:
        out_name4 = prepend_to_folder_name + "borderRemoved"
        if border_removed_isFinalOutput:
            out_name4 = out_name4 + "(finalOutput)"
        output_folder_border_rem = output_folder / out_name4
        os.makedirs(output_folder_border_rem, exist_ok=True)

        if border_rf_model_path is None:
            border_rf_model_path = get_default_border_rf_model_path(resolution)
            border_rf_model = joblib.load(border_rf_model_path)
        else:
            border_rf_model = joblib.load(border_rf_model_path)

    if remove_small_objects:
        out_name5 = prepend_to_folder_name + "smallObjectsRemoved"
        if remove_small_objects_isFinalOutput:
            out_name5 = out_name5 + "(finalOutput)"
        output_folder_small_rem = output_folder / out_name5
        os.makedirs(output_folder_small_rem, exist_ok=True)

        if minimal_object_size is None:
            minimal_object_size = get_default_minimal_object_size(resolution)
        if minimal_object_z_length is None:
            minimal_z_expansion = get_default_minimal_z_length(resolution)



    ################################################Initialize Parameters###############################################
    if max_images_in_memory is None:
        img_shape = get_img_shape_from_filepath(input_files_list[0])
        max_images_in_memory = get_max_images_in_memory_from_img_shape(img_shape)

    if min_signal_intensity is None:
        min_signal_intensity = get_default_min_signal_intensity()

    if high_confidence_threshold is None:
        high_confidence_threshold = get_default_high_confidence_threshold()

    if local_th_offset is None:
        local_th_offset = get_default_local_th_offset(resolution)

    if rball_th is None:
        rball_th = get_default_rball_th(resolution)

    med_disk_radius = get_default_median_radius(resolution)

    th_local_blocksize = get_default_local_th_blocksize(resolution)

    rball_radius = get_default_rball_radius(resolution)


    ##################################Segmentation Pipeline Starts Here#################################################
    progress = 0
    if progress_status is not None:
        progress_status.emit(["Executing segmentation pipeline...", progress])

    times_local_th = []
    times_rball = []
    times_border_removal = []

    for i in range(0, numFiles):
        print("Segmentation Pipeline. Process Image: ", i)
        progress = int((i / numFiles) * 80)

        if progress_status is not None:
            progress_status.emit(["", progress])

        if canceled.is_set():
            return

        img = read_image(input_files_list[i])

        out_name_not_postprocessed = Path(input_files_list[i].name)
        out_name_not_postprocessed = out_name_not_postprocessed.with_suffix(".png")
        out_path_not_postprocessed = output_folder_not_postprocessed / out_name_not_postprocessed

        if store_intermed_results:
            out_name_th_local = Path("localTH_" + input_files_list[i].name)
            out_name_th_local = out_name_th_local.with_suffix(".png")
            out_path_th_local = output_folder_localTh / out_name_th_local

            out_name_rball = Path("rball_" + input_files_list[i].name)
            out_name_rball = out_name_rball.with_suffix(".png")
            out_path_rball = output_folder_rballTh / out_name_rball


        #0. Only pixels whose intensity value is larger than or equal to min_signal_intensity are potentially segmented
        th0 = img >= min_signal_intensity

        #1. First binarization -> segment highest intensity pixels -> very likely that they represent vessels
        th1 = img > high_confidence_threshold


        start = timer()
        #2. Local Thresholding
        med_img = median(img, disk(med_disk_radius))

        th_adapt = threshold_local(med_img, block_size=th_local_blocksize, offset=local_th_offset, method="mean")
        th_adapt = med_img > th_adapt

        if store_intermed_results:
            save_image(th_adapt*255, out_path_th_local, np.uint8)

        if canceled.is_set():
            return

        end = timer()
        times_local_th.append(end - start)

        #3. Rolling Ball filter background subtraction + global thresholding
        start = timer()
        rball = rolling_ball(med_img, radius=rball_radius)
        rball = rball.astype(np.int32)
        rball = img.astype(np.int32)-rball
        rball = np.clip(rball, 0, 65535)
        rball = rball.astype(np.uint16)
        rball = rball > rball_th
        if store_intermed_results:
            save_image(255*rball, out_path_rball, np.uint8)

        end = timer()
        times_rball.append(end - start)

        th_res = np.logical_or(th1, rball)

        th_res = np.logical_or(th_res, th_adapt)
        th_res = np.logical_and(th0, th_res)

        save_image(255*th_res, out_path_not_postprocessed, np.uint8)

        if canceled.is_set():
            return

        #4. Random Forest removal of bright non-vascular objects misidentified as vessel wall
        start = timer()
        if remove_border:
            out_name_border_rem = Path(input_files_list[i].name)
            out_name_border_rem = out_name_border_rem.with_suffix(".png")
            out_path_border_rem = output_folder_border_rem / out_name_border_rem

            mask = predictImagePixels(img, border_rf_model, min_signal_intensity)
            non_blood_vessels = mask != 1
            th_res[non_blood_vessels] = 0

            save_image(255*th_res, out_path_border_rem, np.uint8)

            if canceled.is_set():
                return

        end = timer()
        times_border_removal.append(end - start)

    print(f"Local thresholding took {np.sum(times_local_th)} seconds.")
    print(f"Rolling ball filtering took {np.sum(times_rball)} seconds.")
    print(f"Border removal took {np.sum(times_border_removal)} seconds.")

    if progress_status is not None:
        progress_status.emit(["Executing segmentation pipeline...", 90])


    #5. Remove Small Objects
    start = timer()
    if remove_small_objects:
        if remove_border:
            in_f = output_folder_border_rem
        else:
            in_f = output_folder_not_postprocessed

        remove_small_objects_3D_memeff(in_f, output_folder_small_rem, minimal_object_size, minimal_z_expansion,
                                       max_images_in_memory=max_images_in_memory, canceled=canceled)

    end = timer()
    print(f"Remove small objects 3D took {end - start} seconds.")
