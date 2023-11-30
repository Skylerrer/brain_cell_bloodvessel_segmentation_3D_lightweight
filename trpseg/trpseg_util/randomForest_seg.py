"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a random forest approach for 2D image segmentation.
# Code for training a random forest classifier with given groundtruth as well as code to perform prediction
# using a trained random forest model can be found here.
# The prediction is performed on pixels and not on superpixels like in scripts/superpixel_rf_segmentation_testing.py
########################################################################################################################

import os
import time

import numpy as np
import joblib
from pathlib import Path
from skimage import filters
from sklearn.ensemble import RandomForestClassifier

from trpseg.trpseg_util.utility import read_image, save_image, get_file_list_from_directory, texture_filter


def getPredFeatureStack(img, min_signal):
    """ Get feature stack from an image for prediction.

    A minimal signal can be specified, so that prediction is only done on interesting pixels with a value larger than min_signal.
    This speeds up the prediction process.

    Parameters
    ----------
    img : np.ndarray of shape (N, M)
        A 2D image.
    min_signal : int, float
        Features are only extracted from pixels with a value larger than min_signal.

    Returns
    -------
    all_X : np.ndarray
        2D array that has length equal to the number of pixels for which the features are extracted.
        For every pixel with a value larger than min_signal it stores its feature vector.
    mask : np.ndarray
        Binary 2D array specifying which pixels have a value larger than min_signal == for which pixels feature vectors are extracted
    """

    scales = [0.3, 1.0, 3.5, 5.0, 10.0]

    mask = img > min_signal #e.g. 600

    numPotentialPixels = np.count_nonzero(mask)

    all_X = np.empty((numPotentialPixels, 0), dtype=np.float64)

    for scale in scales:
        X = []
        smoothed = filters.gaussian(img, sigma=scale, truncate=3, preserve_range=False)
        #smoothed_pars = filters.gaussian(pars_img, sigma=scale, truncate=3)
        laplace = filters.laplace(smoothed, ksize=3)
        scharr_mag = filters.scharr(smoothed)
        #struc = feature.structure_tensor(smoothed, sigma=0.1, order='rc')
        #struc_eig = feature.structure_tensor_eigenvalues(struc)
        #s_eig1 = struc_eig[0]
        #s_eig2 = struc_eig[1]
        hess_eig = texture_filter(smoothed)
        h_eig1 = hess_eig[0]
        h_eig2 = hess_eig[1]
        X.append(smoothed[mask])
        #X.append(smoothed_pars.ravel())
        X.append(laplace[mask])
        X.append(scharr_mag[mask])
        #X.append(s_eig1[mask])
        #X.append(s_eig2[mask])
        X.append(h_eig1[mask])
        X.append(h_eig2[mask])

        X = np.asarray(X)
        X = X.T

        all_X = np.append(all_X, X, axis=1)

    return all_X, mask


def getFeatureStack(img, seg_img):
    """ Get the feature vectors for training for pixels that have been annotated in seg_img.

    Parameters
    ----------
    img : np.ndarray
        2D image from which features should be extracted
    seg_img : np.ndarray
        2D label image. The image whith the ground truth annotation for img.
        0 means not annotated
        1 means non-blood vessel
        2 means blood vessel

    Returns
    -------
    all_X : np.ndarray
        2D array that has length equal to the number of pixels for which the features are extracted.
        For every pixel that has been annotated with label 1 or 2 it stores its feature vector.

    Notes
    -------
    Features are:
    1. Gaussian Smoothed image at different scales (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
    2. Laplacian of gaussian: smooth_img_normalized = filters.gaussian(img, sigma=1, truncate=3)     edge = filters.laplace(smooth_img_normalized, 3)
    3. gaussian gradient magnitude
    4. CURRENTLY NOT USED: Structure Tensor Eigenvalues (skimage structure_tensor_eigenvalues)
    5. Hessian Matrix Eigenvalues (skimage hessian_matrix_eigvals)
    """

    scales = [0.3, 1.0, 3.5, 5.0, 10.0]

    y = seg_img.ravel()
    mask = y > 0
    num_rows = np.count_nonzero(mask)

    all_X = np.empty((num_rows, 0), dtype=np.float64)

    for scale in scales:
        X = []
        smoothed = filters.gaussian(img, sigma=scale, truncate=3, preserve_range=False)
        #smoothed_pars = filters.gaussian(pars_img, sigma=scale, truncate=3)
        laplace = filters.laplace(smoothed, ksize=3)
        scharr_mag = filters.scharr(smoothed)
        #struc = feature.structure_tensor(smoothed, sigma=0.1, order='rc')
        #struc_eig = feature.structure_tensor_eigenvalues(struc)
        #s_eig1 = struc_eig[0]
        #s_eig2 = struc_eig[1]
        hess_eig = texture_filter(smoothed)
        h_eig1 = hess_eig[0]
        h_eig2 = hess_eig[1]
        X.append(smoothed.ravel())
        #X.append(smoothed_pars.ravel())
        X.append(laplace.ravel())
        X.append(scharr_mag.ravel())
        #X.append(s_eig1.ravel())
        #X.append(s_eig2.ravel())
        X.append(h_eig1.ravel())
        X.append(h_eig2.ravel())

        X = np.asarray(X)
        X = X.T

        X = X[mask]

        all_X = np.append(all_X, X, axis=1)

    return all_X


def getTrainingData(folder_path, useTransformations):
    """ Get the feature vectors and class labels for training from images in folder_path

    Parameters
    ----------
    folder_path : str, pathlib.Path
        The path to the folder containing the training images as well as their annotations.
        Images from which features are extracted are assumed to contain "_C00_" in their name.
        Annotation images are assumed to contain "Segmentation" in their name.
        The names have to be chosen in a way so that when the C00 and the Segmentation image paths are sorted separately,
        the Segmentation image path at index i belongs to the C00 image path at index i
    useTransformations : bool
        Whether to use data augmentation.

    Returns
    -------
    all_X : np.ndarray
        2D array that has length equal to the number of pixels for which the features are extracted.
        For every pixel that has been annotated with label 1 or 2.

    Notes
    -------
    Works also for only partially annotated images.
    0 means not annotated
    1 means non-blood vessel
    2 means blood vessel
    """

    folder = Path(folder_path)

    segmented_list = folder.glob("*Segmentation*")
    orig_list = folder.glob("*_C00_*")

    segmented_files_list = [file_path for file_path in segmented_list]
    segmented_files_list = sorted(segmented_files_list)
    #segmented_files_list = segmented_files_list[0:1]

    orig_files_list = [file_path for file_path in orig_list]
    orig_files_list = sorted(orig_files_list)
    #orig_files_list = orig_files_list[0:1]


    num_seg = len(segmented_files_list)
    num_orig = len(orig_files_list)


    if num_seg != num_orig:
        raise RuntimeError("The number of segmented and original files must be equal!")

    numFeatures = 25
    all_X = np.empty((0, numFeatures), dtype=np.float64)
    all_y = np.empty((0,), dtype=np.uint8)


    # Create train dataset
    # For each file compute features and then append them to X and and their groundTruth to y
    # do only for pixel that have been annotated

    for i in range(0, num_orig):
        print("Get features from image file ", i)
        orig_img = read_image(orig_files_list[i])
        seg_img = read_image(segmented_files_list[i])

        X = getFeatureStack(orig_img, seg_img)

        y = seg_img.ravel()
        mask = y > 0    #currently 0 means not annotated -> we dont need the data at these pixels
        y = y[mask]

        all_X = np.append(all_X, X, axis=0)
        all_y = np.append(all_y, y, axis=0)

        if useTransformations:
            orig_img = orig_img.astype(np.float64)

            #multiplicative brightness change---------------------
            m_transformed07 = orig_img*0.7
            m_transformed07 = m_transformed07.astype(np.uint16)
            m_transformed13 = orig_img*1.3
            m_transformed13 = np.clip(m_transformed13,0,65535)
            m_transformed13 = m_transformed13.astype(np.uint16)

            X = getFeatureStack(m_transformed07, seg_img)
            all_X = np.append(all_X, X, axis=0)
            all_y = np.append(all_y, y, axis=0)

            X = getFeatureStack(m_transformed13, seg_img)
            all_X = np.append(all_X, X, axis=0)
            all_y = np.append(all_y, y, axis=0)

            # additive brightness change--------------------------
            a_transformed200 = orig_img + 100
            a_transformed200 = np.clip(a_transformed200, 0, 65535)
            a_transformed200 = a_transformed200.astype(np.uint16)

            a_transformed_n_200 = orig_img - 100
            a_transformed_n_200 = np.clip(a_transformed_n_200, 0, 65535)
            a_transformed_n_200 = a_transformed_n_200.astype(np.uint16)

            X = getFeatureStack(a_transformed200, seg_img)
            all_X = np.append(all_X, X, axis=0)
            all_y = np.append(all_y, y, axis=0)

            X = getFeatureStack(a_transformed_n_200, seg_img)
            all_X = np.append(all_X, X, axis=0)
            all_y = np.append(all_y, y, axis=0)

    return all_X, all_y


# Train a random forest classifier
def trainRandomForest(folder_path, useTransformations, output_file_path):
    """

    Parameters
    ----------
    folder_path : str, pathlib.Path
        The path to the folder containing the training images as well as their annotations.
        Images from which features are extracted are assumed to contain "_C00_" in their name.
        Annotation images are assumed to contain "Segmentation" in their name.
        The names have to be chosen in a way so that when the C00 and the Segmentation image paths are sorted separately,
        the Segmentation image path at index i belongs to the C00 image path at index i
    useTransformations : bool
        Whether to use data augmentation during training (does additive + multiplicative brightness changes).
    """

    #rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=35)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_samples=0.5)

    X, y = getTrainingData(folder_path, useTransformations=useTransformations)
    print("Start training!")
    start = time.time()

    rf.fit(X, y)

    end = time.time()
    print("Training time: ", end - start)

    # save
    joblib.dump(rf, output_file_path, compress=3)


def predictFolder(input_folder_path, output_folder_path, model_path, min_signal):
    """Predict the class for images in input_folder_path and store predictions in output_folder_path

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the images on which prediction should be performed.
    output_folder_path: str, pathlib.Path
        The path to the folder where the output images are stored.
    model_path : str, pathlib.Path
        The path to the trained model file.
    min_signal : int, float
        Prediction is only performed for pixels with values larger than min_signal. The pixels for which
        no prediction is performed are assumed to be non-blood vessels.
    """

    rf = joblib.load(model_path)

    input_files_list = get_file_list_from_directory(input_folder_path, isTif=True, channel_one=False)

    numImages = len(input_files_list)

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, numImages):
        print("Predict Image: ", i)
        img = read_image(input_files_list[i])

        seg = predictImagePixels(img, rf, min_signal)

        out_name = Path(input_files_list[i].name)
        out_name = out_name.with_suffix(".png")
        out_path = output_folder / out_name
        save_image(seg, out_path, dtype=np.uint8)


def predictImagePixels(img, rf_model, min_signal):
    """ Predict the pixel classes for image img.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    rf_model : RandomForestClassifier
        The loaded random forest classifier model.
    min_signal : int, float
        Prediction is only performed for pixels with values larger than min_signal. The pixels for which
        no prediction is performed are assumed to be non-blood vessels.

    Returns
    -------
    seg : np.ndarray
        The segmentation result as 2D numpy array with type np.uint8
    """

    X, mask = getPredFeatureStack(img, min_signal)

    seg = np.ones(shape=img.shape, dtype=np.uint16)
    seg = seg * 1

    if len(X) > 0:
        pred = rf_model.predict(X)
        seg[mask] = pred

    #predict returns 1 for non blood vessels and 2 for blood vessels
    seg = seg - 1

    return seg.astype(np.uint8)


def main():
    pass


if __name__ == "__main__":
    main()
