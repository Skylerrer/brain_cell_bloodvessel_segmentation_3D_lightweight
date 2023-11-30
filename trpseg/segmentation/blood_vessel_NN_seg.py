"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

######################################################File Content######################################################
# This file implements a deep learning based algorithm to segment complete/filled blood vessels from microscopy images
# where only the blood vessel wall was fluorescently labeled.
# A 2D U-Net architecture is used.
########################################################################################################################

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import shutil
from threading import Event
import math
from dataclasses import dataclass
from datetime import datetime


import tensorflow as tf
import numpy as np
from scipy import ndimage
from timeit import default_timer as timer
from pathlib import Path
from tqdm import trange
from tensorflow.python.data import AUTOTUNE


from trpseg import DATA_DIRECTORY
from trpseg import OUTPUT_DIRECTORY
from trpseg.trpseg_util.utility import read_image, read_img_list, save_image, \
    remove_small_objects_3D_memeff, opening3D, get_voxel_num_from_physical_volume, \
    get_voxel_num_from_physical_length, get_file_list_from_directory, get_img_shape_from_filepath, \
    get_z_radius_from_footprint, get_max_images_in_memory_from_img_shape,\
    anisotropic_spherical_3D_SE



def get_next_valid_input_size_for_unet(current_size, num_downsamplings):
    """Get integer larger than or equal to current_size that is a valid input size for a UNet with the given number of down_samplings

    Parameters
    ----------
    current_size : int
        The current size of an axis of the input.

    num_downsamplings : int
        The number of downsamplings that is performed by the UNet (== the number of MaxPoolingLayers).

    Returns
    -------
    current_size : int
        A valid size for an input dimension to a UNet with the specified number of down samplings.

    Notes
    ----------
    Every axis of the input to a UNet has to have a size that is num_downsampling times divisible by 2 <=> one time divisible by  2^(num_downsamplings)
    If, this is not the case for current_size then the next larger valid integer will be returned.
    Else, current_size will be returned
    """
    # The size has to be num_downsampling times divisible by 2 <=> one time divisible by  2^(num_downsamplings)
    # Return next larger integer that fullfills this

    if current_size < 1:
        raise RuntimeError(f"current_size needs to be >= 1 but got {current_size}")

    if num_downsamplings < 1:
        raise RuntimeError(f"num_downsamplings needs to be >= 1 but got {num_downsamplings}")

    necessary_divisor = pow(2, num_downsamplings)

    remainder = current_size % necessary_divisor

    if remainder > 0:
        return current_size + (necessary_divisor-remainder)

    return current_size


@dataclass
class Settings:
    """Class that stores the important hyperparameters."""
    name: str
    learning_rate: float
    num_epochs: int
    batch_size: int


def get_img_tiles(img, tile_size=(512, 512), offset=(512, 512)):
    """Split an image into tiles of a given size and with a given offset.

    Parameters
    ----------
    img : np.ndarray
        The image that should be split into tiles.
    tile_size : 2-tuple of int
        The size of the tiles in pixels in both dimensions. Order is (height, width) == (length over axis 0, length over axis 1)
    offset : 2-tuple of int
        The offset used to create the tiles. Offset specified for both axes. Order is (height_offset, width_offset) == (length over axis 0, length over axis 1)

    Returns
    -------
    List of 2D numpy ndarrays : The image tiles that all have a size of tile_size.
    """

    img_shape = img.shape
    img_tiles = []
    for j in range(int(math.ceil(img_shape[1] / (offset[1] * 1.0)))):
        for i in range(int(math.ceil(img_shape[0] / (offset[0] * 1.0)))):
            cropped_img = img[offset[0] * i:min(offset[0] * i + tile_size[0], img_shape[0]),
                          offset[1] * j:min(offset[1] * j + tile_size[1], img_shape[1])]

            # if image size is not divisible by tile_size then padding is needed to make all tiles equally sized
            if cropped_img.shape != tile_size:
                pad_to_bottom = tile_size[0] - cropped_img.shape[0]
                pad_to_right = tile_size[1] - cropped_img.shape[1]
                cropped_img = np.pad(cropped_img, ((0,pad_to_bottom), (0, pad_to_right)), 'constant', constant_values=0)

            img_tiles.append(cropped_img)

    return img_tiles


def load_images(paths, useTiling=False, dtype=tf.float32, tile_size=(512, 512), offset=(512, 512)):
    """Load images from disk and possibly split them into tiles.

    Parameters
    ----------
    paths : list of str, list of pathlib.Path
        List of paths to images that should be loaded into memory.
    useTiling : bool
        Determines whether to split the images into tiles.
    dtype : np.dtype.type
        The datatype under which the images should be loaded into memory.
    tile_size : 2-tuple of int
        Only used if useTiling == True.
        The size of the tiles specified in 2D. Order is (height, width) == (length over axis 0, length over axis 1)
    offset : 2-tuple of int
        Only used if useTiling == True.
        The offset used to create the tiles. Offset specified for both axes. Order is (height_offset, width_offset) == (length over axis 0, length over axis 1)


    Returns
    -------
    imgs : 4D tensorflow tensor
    The images/image tiles stored in a tensor. Order: [batch, height, width, channel]
    """

    if useTiling:
        imgs = read_img_list_with_tiling(paths, tile_size, offset)
    else:
        imgs = read_img_list(paths, np.uint16)

    imgs = tf.cast(imgs, dtype)
    num_slices = len(imgs)

    if useTiling:
        imgs = tf.reshape(imgs, (num_slices, tile_size[0], tile_size[1], 1))
        imgs.set_shape((num_slices, tile_size[0], tile_size[1], 1))
    else:
        img_height, img_width = get_img_shape_from_filepath(paths[0])

        imgs = tf.reshape(imgs, (num_slices, img_height, img_width, 1))
        imgs.set_shape((num_slices, img_height, img_width, 1))


    return imgs


def read_img_list_with_tiling(input_file_paths, tile_size, offset):
    """Read the images from the given file paths into memory and split them into tiles of tile_size.

    Parameters
    ----------
    input_file_paths : list of str, list of pathlib.Path
        The file paths to the images that should be read into memory and tiled.
    tile_size : 2-tuple of int
        The size of the tiles specified in 2D. Order is (height, width) == (length over axis 0, length over axis 1)
    offset : 2-tuple of int
        The offset used to create the tiles. Offset specified for both axes. Order is (height_offset, width_offset) == (length over axis 0, length over axis 1)

    Returns
    -------
    imgs : np.ndarray
    Array that contains the image tiles.
    """

    imgs = []
    for file_path in input_file_paths:
        tile_imgs = read_image(file_path)
        tile_imgs = get_img_tiles(tile_imgs, tile_size, offset)
        imgs += tile_imgs
    imgs = np.stack(imgs, axis=0)
    return imgs


# - a function which loads the dataset from the disk and outputs batches
def load_dataset(path, batch_size=64, with_masks=False, shuffle=False, useTiling=False, tile_size=(512, 512), offset=(512, 512)):
    """Load a dataset from disk and output batched dataset.

    Parameters
    ----------
    path : str, pathlib.Path
        The path to the folder that stores a dataset. This folder should contain a folder called original in which the original tif images are stored.
        If with_masks==True, then there should be a second folder called masks containing the binary png masks representing the ground truth segmentation.
    batch_size : int
        The batch size of the dataset.
    with_masks : bool
        Determines whether to also load ground truth masks.
    shuffle : bool
        Determines whether to randomly shuffle the datasets elements.
    useTiling : bool
        Determines whether to split the images into tiles.
    tile_size : 2-tuple of int
        The size of the tiles specified in 2D. Order is (height, width) == (length over axis 0, length over axis 1)
    offset : 2-tuple of int
        The offset used to create the tiles. Offset specified for both axes. Order is (height_offset, width_offset) == (length over axis 0, length over axis 1)


    Returns
    -------
    The dataset loaded into memory.
    """

    folder = Path(path)

    img_paths = folder / "original"

    image_paths = list(map(str, img_paths.rglob("*.tif")))
    image_paths = sorted(image_paths)
    imgs = load_images(image_paths, useTiling, tile_size=tile_size, offset=offset)

    if with_masks:
        mask_paths = folder / "masks"
        mask_paths = list(map(str, mask_paths.rglob("*.png")))
        mask_paths = sorted(mask_paths)

        if len(image_paths) != len(mask_paths):
            raise RuntimeError("Len of images and masks must be equal!")

        masks = load_images(mask_paths, useTiling, dtype=tf.uint8, tile_size=tile_size, offset=offset)
        mask_ds = tf.data.Dataset.from_tensor_slices(masks)


    if with_masks:
        #for i in range(0, len(masks)):# Uncomment the following if only wants tiles with both classes for training
            #u = np.unique(masks[i])#
            #if len(u) < 2:#
                #rem_tiles.append(i)#

        #tiles_with_both_classes = np.setdiff1d(all_masks_ids, rem_tiles)#
        #masks = tf.gather(masks, tiles_with_both_classes)#
        #imgs = tf.gather(imgs, tiles_with_both_classes)#

        mask_ds = tf.data.Dataset.from_tensor_slices(masks)

    image_ds = tf.data.Dataset.from_tensor_slices(imgs)


    seed = 12441
    if shuffle:
        image_ds = image_ds.shuffle(batch_size * 4, seed=seed)
        if with_masks:
            mask_ds = mask_ds.shuffle(batch_size * 4, seed=seed)

    if batch_size > 0:
        image_ds = image_ds.batch(batch_size)  # [Batch, Height, Width, Channel]
        if with_masks:
            mask_ds = mask_ds.batch(batch_size)  # [Batch, Height, Width, Channel]

    image_ds = image_ds.cache().prefetch(AUTOTUNE)

    if with_masks:
        mask_ds = mask_ds.cache().prefetch(AUTOTUNE)
        return tf.data.Dataset.zip((image_ds, mask_ds))

    else:
        return image_ds


def create_unet_small(useSubsampling: bool, input_shape=(2048, 2048)):
    """Create a very small version of the 2D UNet presented in "U-Net:Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et. al."""

    inputs = tf.keras.Input([input_shape[0], input_shape[1], 1])

    if useSubsampling:
        inputs = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(inputs)

    rescaled_inputs = tf.keras.layers.Rescaling(scale=1. / 65535)(inputs)


    def double_conv_block(x, filterNum):
        x = tf.keras.layers.Conv2D(filterNum, (3, 3), padding='SAME', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filterNum, (3, 3), padding='SAME', activation='relu')(x)
        return x

    def down_block(x, filterNum):
        skip = double_conv_block(x, filterNum)
        down = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(skip)
        return skip, down

    def up_block(x, skipped, filterNum):
        x = tf.keras.layers.Conv2DTranspose(filterNum, (2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([x, skipped])
        x = double_conv_block(x, filterNum)
        return x

    s1, d1 = down_block(rescaled_inputs, 8)
    s2, d2 = down_block(d1, 16)
    center = double_conv_block(d2, 32)
    o4 = up_block(center, s2, 16)
    o5 = up_block(o4, s1, 8)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(o5)  #if more than 2 classes then use softmax

    if useSubsampling:
        outputs = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    model = tf.keras.Model(inputs, outputs)

    #model.summary()

    return model


def create_unet_opti_small(useSubsampling, input_shape=(2048, 2048)):
    """Create a very small version of the 2D UNet presented in "U-Net:Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et. al.

    Has an additional data augmentation layer (RandomBrightness).

    And, use optimizations described in "3D U-Net:Learning Dense Volumetric Segmentation from Sparse Annotation", Cicek et. al.
    1. Batch Normalization for faster convergence
    2. Avoid bottlenecks by doubling the number of feature maps before max pooling.
       Detailed reasoning for 2.: "Rethinking the Inception Architecture for Computer Vision", Szegedy et. al.
    """

    inputs = tf.keras.Input([input_shape[0], input_shape[1], 1])

    if useSubsampling:
        inputs = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(inputs)

    inputs = tf.keras.layers.Rescaling(scale=1. / 65535)(inputs)

    inputs = tf.keras.layers.RandomBrightness(factor=0.01, value_range=(0.0, 1.0))(inputs)  #data augmentation (output has same dtype as input and is in value_range)


    def double_conv_block(x, filterNum1, filterNum2):
        x = tf.keras.layers.Conv2D(filterNum1, (3, 3), padding='SAME', kernel_initializer="random_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filterNum2, (3, 3), padding='SAME', kernel_initializer="random_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def down_block(x, filterNum1, filterNum2):
        skip = double_conv_block(x, filterNum1, filterNum2)
        down = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(skip)
        return skip, down

    def up_block(x, skipped, filterNumUp, filterNum1, filterNum2):
        x = tf.keras.layers.Conv2DTranspose(filterNumUp, (2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([x, skipped])
        x = double_conv_block(x, filterNum1, filterNum2)
        return x

    s1, d1 = down_block(inputs, 8, 16)
    s2, d2 = down_block(d1, 16, 32)
    center = double_conv_block(d2, 32, 64)
    o4 = up_block(center, s2, 64, 32, 32)
    o5 = up_block(o4, s1, 32, 16, 16)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(o5)  #if more than 2 classes then use softmax

    if useSubsampling:
        outputs = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    model = tf.keras.Model(inputs, outputs)

    #model.summary()

    return model


# Paper: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation":
# Like suggested in [13] we avoid bottlenecks by doubling the number of channels already before max pooling. We also adopt
# this scheme in the synthesis path.  And use batch normalization [4] for faster convergence.

#RandomBrightness are not used when training=False. This is the case when we for example call model.predict()
#BatchNormalization is used when training=False and also when training=True -> However, different behaviour during training
# -> see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

def create_unet_opti_large(useSubsampling, input_shape=(2048, 2048)):
    """Create a slightly smaller version of the 2D UNet presented in "U-Net:Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et. al.

    Has an additional data augmentation layer (RandomBrightness).

    And, use optimizations described in "3D U-Net:Learning Dense Volumetric Segmentation from Sparse Annotation", Cicek et. al.
    1. Batch Normalization for faster convergence
    2. Avoid bottlenecks by doubling the number of feature maps before max pooling.
       Detailed reasoning for 2.: "Rethinking the Inception Architecture for Computer Vision", Szegedy et. al.
    """
    inputs = tf.keras.Input([input_shape[0], input_shape[1], 1])

    if useSubsampling:
        inputs = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(inputs)

    inputs = tf.keras.layers.Rescaling(scale=1. / 65535)(inputs)

    inputs = tf.keras.layers.RandomBrightness(factor=0.01, value_range=(0.0, 1.0))(inputs)  # data augmentation (output has same dtype as input and is in value_range)

    def double_conv_block(x, filterNum1, filterNum2):
        x = tf.keras.layers.Conv2D(filterNum1, (3, 3), padding='SAME', kernel_initializer="random_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filterNum2, (3, 3), padding='SAME', kernel_initializer="random_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def down_block(x, filterNum1, filterNum2):
        skip = double_conv_block(x, filterNum1, filterNum2)
        down = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))(skip)
        return skip, down

    def up_block(x, skipped, filterNumUp, filterNum1, filterNum2):
        x = tf.keras.layers.Conv2DTranspose(filterNumUp, (2, 2), strides=(2, 2),
                                            kernel_initializer="random_normal")(x)
        x = tf.keras.layers.concatenate([x, skipped])
        x = double_conv_block(x, filterNum1, filterNum2)
        return x

    s1, d1 = down_block(inputs, 16, 32)
    s2, d2 = down_block(d1, 32, 64)
    s3, d3 = down_block(d2, 64, 128)
    center = double_conv_block(d3, 128, 256)
    o4 = up_block(center, s3, 256, 128, 128)
    o5 = up_block(o4, s2, 128, 64, 64)
    o6 = up_block(o5, s1, 64, 32, 32)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(o6)  # if more than 2 classes then use softmax

    if useSubsampling:
        outputs = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    model = tf.keras.Model(inputs, outputs)

    #model.summary()

    return model


#existing_model parameter makes it possible to specify a tf.keras.Model that should be trained (can be a pretrained model)
#this model just has to be able to handle the input from train_ds + val_ds and it has to output images of the same shape as the input
def train_network(train_ds, val_ds, settings: Settings, existing_model=None):
    """Train the create_unet_opti_large model or if existing_model is not None then train that model.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        The training dataset.
    val_ds : tf.data.Dataset
        The validation dataset.
    settings : Settings
        A settings object, containing the hyperparameters.
    existing_model : tf.keras.Model, optional
        An existing possibly pre-trained model that you want to train.
        If None, then the create_unet_opti_large model architecture is used.

    Returns
    -------
    model : tf.keras.Model
        The trained neural network model.
    """

    settings.name = settings.name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = f"..\\output\\dataNN\\logs\\{settings.name}\\"
    checkpoint_path = "../output/dataNN/checkpoints/chkpt-{epoch:03d}-{val_loss:.4f}/"
    checkpoint_dir = "../output/dataNN/checkpoints/"
    Path(log_dir).mkdir(parents=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    #Get the shape of the images used for training
    tmp_shape_helper = train_ds.as_numpy_iterator().next()[0]
    input_shape = (tmp_shape_helper.shape[1], tmp_shape_helper.shape[2])
    tmp_shape_helper = -1

    if isinstance(existing_model, tf.keras.Model):
        model = existing_model
    else:
        #model = create_unet_opti_small(useSubsampling=False)
        model = create_unet_opti_large(useSubsampling=False, input_shape=input_shape)

    model.compile()

    optimiser = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)


    @tf.function
    def train_step(batch, training=True):
        with tf.GradientTape() as tape:
            groundtruth = batch[1]
            unsegmented = batch[0]
            network_output = model(unsegmented)
            loss = get_loss(network_output, groundtruth)
        if training:
            grads = tape.gradient(loss, model.trainable_weights)
            optimiser.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    # setup tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    callback_list = tf.keras.callbacks.CallbackList([tensorboard_callback, model_checkpoint_callback],
                                                        add_history=True, model=model)


    logs = {}
    callback_list.on_train_begin(logs=logs)

    for epoch in trange(settings.num_epochs):
        # training for a single epoch
        losses = []
        callback_list.on_epoch_begin(epoch, logs)
        i = 1
        for batch in train_ds:
            print(f"Epoch {epoch} | Batch {i}")
            loss = train_step(batch)
            losses.append(loss)
            i += 1

        epoch_loss = np.mean(losses)
        print("Training Loss:", epoch_loss)
        logs = {'train_loss': epoch_loss}

        losses = []

        for batch in val_ds:
            loss = train_step(batch, training=False)
            losses.append(loss)

        val_logs = {'val_loss': np.mean(losses)}
        callback_list.on_epoch_end(epoch, logs=logs | val_logs)

    return model


# Code modified from:
######################################################################################
#   Title: testClassBalanceLoss
#   Paper: "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature"
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/ImageProcessing/MachineLearning/VesselFilling/Training/class_balancing_loss.py
######################################################################################.

def get_loss(network_output, groundtruth):
    """Compute loss (balanced cross entropy + rate correction) given the network output for a batch as well as the groundtruth for the batch.

    Parameters
    ----------
    network_output : tf.Tensor
        The network output for a batch (in our case probabilities between 0 and 1)
    groundtruth : tf.Tensor
        The ground truth for the batch. 0 will be interpreted as 0 and everything greater than 0 will be interpreted as 1.

    Returns
    -------
    result :
        The loss of the network output on a batch given the groundtruth for the batch.
        This will be a single floating point number.
    """

    network_output = tf.clip_by_value(network_output, 1e-7, 1.0 - 1e-7) #to prevent log(0) errors -> does not change prediction too much
    y_plus = network_output[groundtruth != 0]  # This creates a 1D Tensor with all probabilities from network_output where groundtruth is != 0
    y_minus = 1 - network_output[groundtruth == 0]  # This creates a 1D Tensor with all probabilities from network_output where groundtruth is == 0

    num_pos = len(y_plus)
    num_neg = len(y_minus)
    # Prevent 0 division errors
    # For balanced_cross_entropy it does not matter to what we set num_pos and num_neg because if num_pos = 0 then res1 = 0 because y_plus empty and if num_neg = 0 then res2 = 0
    # However, in rate_correction() it does matter -> set to 1 -> so error does not explode but will be high if there e.g shouldnt be positive prediction but there are (false positives)

    if num_pos == 0:
        num_pos = 1
    if num_neg == 0:
        num_neg = 1

    num_pos = tf.cast(num_pos, tf.float32)
    num_neg = tf.cast(num_neg, tf.float32)


    pos = tf.math.logical_and(groundtruth == 0, network_output > 0.5)  # pos has the same shape as network_output and marks the false positives
    neg = tf.math.logical_and(groundtruth > 0, network_output <= 0.5)  # neg has the same shape as network_output and marks the false negatives

    false_pos = 1 - network_output[pos]  # This creates a 1D Tensor with all probabilities from network_output where pos is True
    false_neg = network_output[neg]  # This creates a 1D Tensor with all probabilities from network_output where neg is True

    #Balanced binary cross entropy loss +
    #False prediction rate correction
    L1 = balanced_cross_entropy(y_plus, y_minus, num_pos, num_neg)
    L2 = rate_correction(false_pos, false_neg, num_pos, num_neg)
    result = L1 + L2

    return result


# Code modified from:
######################################################################################
#   Title: cross_entropy
#   Paper: "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature"
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/ImageProcessing/MachineLearning/VesselFilling/Training/class_balancing_loss.py
######################################################################################.

def balanced_cross_entropy(y_plus, y_minus, num_pos, num_neg):
    """Compute the balanced cross-entropy loss.

    Parameters
    ----------
    y_plus : tf.Tensor
        The 1D array containing the predicted probabilites where groundtruth is != 0
    y_minus : tf.Tensor
        The 1D array containing 1-(the predicted probabilites where groundtruth is == 0)
    num_pos : tf.float32
        The total number of positive class labels (class label 1) in the ground truth of the batch.
    num_neg : tf.float32
        The total number of negative class labels (class label 0) in the ground truth of the batch.

    Returns
    -------
    res :
        The balanced cross entropy loss.
    """

    res1 = tf.reduce_sum(tf.math.log(tf.convert_to_tensor(y_plus, tf.float32)))
    res2 = tf.reduce_sum(tf.math.log(tf.convert_to_tensor(y_minus, tf.float32)))

    res = -((1/(num_pos))*res1)-((1/(num_neg))*res2)

    return res


# Code modified from:
######################################################################################
#   Title: rate_correction
#   Paper: "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature"
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/ImageProcessing/MachineLearning/VesselFilling/Training/class_balancing_loss.py
######################################################################################.

def rate_correction(false_pos, false_neg, num_pos, num_neg):
    """Compute a rate correction loss that penalizes false positives and false negatives directly.

    Parameters
    ----------
    false_pos : tf.Tensor
        The 1D array containing 1-(the predicted probabilites of the false positive predictions)
    false_neg : tf.Tensor
        The 1D array containing the predicted probabilites of the false negative predictions
    num_pos : tf.float32
        The total number of positive class labels (class label 1) in the ground truth of the batch.
    num_neg : tf.float32
        The total number of negative class labels (class label 0) in the ground truth of the batch.

    Returns
    -------
    res :
        The rate correction loss.
    """

    res1 = tf.reduce_sum(tf.math.log(tf.convert_to_tensor(false_pos, tf.float32)))
    res2 = tf.reduce_sum(tf.math.log(tf.convert_to_tensor(false_neg, tf.float32)))

    l1 = len(false_pos)
    l2 = len(false_neg)

    #Prevent 0 division errors
    # if l1 is 0 we can set is to any number we want because tf.reduce_sum(tf.math.abs(false_pos - 0.5) will be 0 anyways if false_pos is empty
    # same for l2
    if l1 == 0:
        l1 = 1
    if l2 == 0:
        l2 = 1

    l1 = tf.cast(l1, tf.float32)
    l2 = tf.cast(l2, tf.float32)

    g1 = 0.5 + ((1 / (l1)) * tf.reduce_sum(tf.math.abs(false_pos - 0.5)))
    g2 = 0.5 + ((1 / (l2)) * tf.reduce_sum(tf.math.abs(false_neg - 0.5)))

    res = -((g1/(num_pos))*res1)-((g2/(num_neg))*res2)

    return res


def networkSegmentTrain():
    """Train the Neural Network"""
    # Each optimizer call to optimizer.apply_gradients() on a batch is one optimization step
    # When using batchsize 16 and tile_size 512x512 for 2048x2048 images then we have 46images = 46 batches
    # When training for 200 epochs then we have 200* 46 = 9200 training steps
    # Example decay: lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([230, 4600, 6900], [5e-3, 1e-3, 5e-4, 1e-4])
    # Example decay: lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([4600, 6900], [1e-3, 5e-4, 1e-4])
    #For 70 images , 150 epochs ->  150*70=10500
    #For 70 images, 200 epochs -> 200*70 = 14000

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([7000, 10500, 12600], [5e-4, 1e-4, 5e-5, 1e-5])
    # lr = 1e-3
    settings = Settings("vesselNN", lr, 5, 16) #The larger the batchsize the larger the memory requirements
    train_ds = load_dataset(r"..\..\training_resources\training_dataNN\train", shuffle=False, batch_size=settings.batch_size, useTiling=True, with_masks=True)
    val_ds = load_dataset(r"..\..\training_resources\training_dataNN\val", shuffle=False, batch_size=settings.batch_size, useTiling=True, with_masks=True)

    model = train_network(train_ds, val_ds, settings)
    model.save(f"..\\resources\\NN_models\\{settings.name}")


def get_default_NN_model_path():
    """Get the path to the default UNet model used to perform segmentation of entire blood vessels.
    """

    return os.path.join(DATA_DIRECTORY, "NN_models", "vesselNN_20230831_172111_epoch063_cplex")


def networkPredictFolder(input_folder_path, output_folder_path, trained_model_path=None, model_architecture="unet2D_large", isTif=True, channel_one=False, canceled=Event(), progress_status=None, max_progress=50):
    """
    Perform neural network prediction on all the images in input_folder_path to create segmentation of blood vessels.
    If no path to a trained model is given then a default already trained 2D U-Net is used.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack on which prediction should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    trained_model_path : str, pathlib.Path
        The path to a trained tf.keras.model that was previously saved via tf.keras.Model.save
    model_architecture : str
        You can choose between unet2D_small and unet2D_large.
        unet2D_small uses the model architecture from create_unet_opti_small
        unet2D_large uses the model architecture from create_unet_opti_large
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

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    img_height, img_width = get_img_shape_from_filepath(image_paths[0])


    if model_architecture not in ["unet2D_small", "unet2D_large"]:
        raise RuntimeError(f"{model_architecture} is an unsupported model_architecture. Options are 'unet2D_small' and 'unet2D_large'!")


    #Determin if images need to be padded to have valid input_shape for unet
    #Get valid input_shape for unet
    if model_architecture == "unet2D_small":
        target_height = get_next_valid_input_size_for_unet(img_height, 2)
        target_width = get_next_valid_input_size_for_unet(img_width, 2)
    else:  # model_architecture == "unet2D_large":
        target_height = get_next_valid_input_size_for_unet(img_height, 3)
        target_width = get_next_valid_input_size_for_unet(img_width, 3)


    use_padding = False
    padding = ((0, 0), (0, 0))
    padding_height = 0
    padding_width = 0

    if target_width != img_width or target_height != img_height:
        use_padding = True
        padding_height = target_height-img_height
        padding_width = target_width-img_width
        padding = ((0, padding_height), (0, padding_width))


    # load model and predict
    if trained_model_path is None:
        if model_architecture == "unet2D_small":
            trained_model_path = os.path.join(DATA_DIRECTORY, "NN_models", "vesselNN_20230211_121842_simple_kaggle300")
        else:  #model_architecture == "unet2D_large"
            trained_model_path = get_default_NN_model_path()

    model = tf.keras.models.load_model(trained_model_path, compile=False)

    tmp_checkpoint_dir = os.path.join(OUTPUT_DIRECTORY, r"dataNN\tmp_checkpoint")
    os.makedirs(tmp_checkpoint_dir, exist_ok=True)
    tmp_checkpoint = os.path.join(tmp_checkpoint_dir, "my_checkpoint")

    model.save_weights(tmp_checkpoint)    #This is done so we can load new model with other input_shape


    if model_architecture == "unet2D_small":
        model = create_unet_opti_small(useSubsampling=False, input_shape=(target_height, target_width))
    else:
        model = create_unet_opti_large(useSubsampling=False, input_shape=(target_height, target_width))

    model.load_weights(tmp_checkpoint)
    model.compile()


    numImages = len(image_paths)

    progress = 0

    if progress_status is not None:
        progress_status.emit(["Predicting...", progress])

    for i in range(0, numImages):

        if canceled.is_set():
            return

        progress = int((i / numImages) * max_progress)

        if progress_status is not None:
            progress_status.emit(["", progress])

        img = read_image(image_paths[i])

        if use_padding:
            img = np.pad(img, padding, 'constant', constant_values=0)

        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, (1, target_height, target_width, 1))
        img.set_shape((1, target_height, target_width, 1))

        model_out = model.predict(img)
        segmentation = (model_out > 0.5) * 255
        segmentation = np.squeeze(segmentation)
        segmentation.astype(np.uint8)

        if use_padding:
            if padding_height > 0:
                segmentation = segmentation[:-padding_height, :]
            if padding_height > 0:
                segmentation = segmentation[:, :-padding_width]

        out_name = Path(image_paths[i].name)
        out_name = out_name.with_suffix(".png")
        out_path = output_folder / out_name
        save_image(segmentation, out_path, np.uint8)



def get_default_postprocessing_footprint(resolution):
    """
    Get footprint(/structuring element) for postprocessing of the binary images produced by UNet prediction.
    The footprint is spherical and possible anisotropic depending on the resolution.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Notes
    ----------

    In tests a spherical footprint with physical radius of 1.1 worked well


    """
    xy_res = resolution[1]

    xy_radius = int(np.floor(1.1 / xy_res))

    footprint = anisotropic_spherical_3D_SE(xy_radius, resolution)

    return footprint


def postProcessDLRes(input_folder_path, output_folder_path, resolution, removeSmallObjects, fill_holes, do_opening,
                     max_images_in_memory=100, footprint=None, canceled=Event(), progress_status=None, progress_min=50):
    """Perform post-processing tailored to the output of the U-Net prediction of filled blood vessels.

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D binary image stack on which postprocessing should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    removeSmallObjects : bool
        Determines whether to remove small objects in postprocessing
    fill_holes : bool
        Determines whether to fill small holes in postprocessing
    do_opening : bool
        Determines whether to perform 3D opening in postprocessing
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
    footprint : np.ndarray, optional
        The footprint used for morphological postprocessing operations
        If None, then a default footprint is used.
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status
    progress_min : int
        The progress status percentage to start with
    """

    if not removeSmallObjects and not fill_holes and not do_opening:
        return

    output_folder_path = Path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    input_folder_path = Path(input_folder_path)
    current_input_folder_path = Path(input_folder_path)
    current_output_folder_path = output_folder_path

    output_folder_tmp = time.strftime("%d%m%Y%H%M%S_") + "out_tmp"
    output_folder_tmp = output_folder_path / output_folder_tmp
    os.makedirs(output_folder_tmp, exist_ok=True)

    if footprint is None:
        footprint = get_default_postprocessing_footprint(resolution)

    z_radius = get_z_radius_from_footprint(footprint)

    min_voxel_num = get_voxel_num_from_physical_volume(resolution, 845) #in case of resolution=[2.0,0.325,0.325] -> 4000 voxel
    min_z_length = get_voxel_num_from_physical_length(resolution, 30, axis=0)

    progress = progress_min
    if progress_status is not None:
        progress_status.emit(["Postprocessing...", progress])

    if fill_holes:

        if canceled.is_set():
            return

        print("Fill holes")
        #fill_holes
        input_files_list = get_file_list_from_directory(current_input_folder_path, isTif=False, channel_one=False)

        num_images = len(input_files_list)

        for i in range(0, num_images):
            if canceled.is_set():
                return
            img = read_image(input_files_list[i])
            img = ndimage.binary_fill_holes(img)

            out_name = Path(input_files_list[i].name)
            out_path = current_output_folder_path / out_name

            save_image(img, out_path, dtype=np.uint8)

        in_to_out_path = current_input_folder_path
        current_input_folder_path = current_output_folder_path
        if in_to_out_path == input_folder_path:
            in_to_out_path = output_folder_tmp
        current_output_folder_path = in_to_out_path

    if canceled.is_set():
        return

    progress = 70
    if progress_status is not None:
        progress_status.emit(["", progress])

    if removeSmallObjects:
        print("Remove Small Objects")
        remove_small_objects_3D_memeff(current_input_folder_path, current_output_folder_path, min_voxel_num, z_remove_width=min_z_length, max_images_in_memory=max_images_in_memory, canceled=canceled)

        in_to_out_path = current_input_folder_path
        current_input_folder_path = current_output_folder_path
        if in_to_out_path == input_folder_path:
            in_to_out_path = output_folder_tmp
        current_output_folder_path = in_to_out_path


    if canceled.is_set():
        return

    progress = 80
    if progress_status is not None:
        progress_status.emit(["", progress])

    if do_opening:
        print("Open")
        #opening
        opening3D(current_input_folder_path, current_output_folder_path, footprint, radius=z_radius, canceled=canceled)

        in_to_out_path = current_input_folder_path
        current_input_folder_path = current_output_folder_path
        if in_to_out_path == input_folder_path:
            in_to_out_path = output_folder_tmp
        current_output_folder_path = in_to_out_path

    if canceled.is_set():
        return


    progress = 90
    if progress_status is not None:
        progress_status.emit(["", progress])


    #If final images are in tmp folder then move them to output_folder_path and remove tmp folder

    if current_input_folder_path != output_folder_path:

        input_files_list = get_file_list_from_directory(current_input_folder_path, isTif=False)

        for i in range(len(input_files_list)):

            destination = output_folder_path / input_files_list[i].name
            shutil.move(input_files_list[i], destination)

    shutil.rmtree(output_folder_tmp)


def network_predict_and_postprocess(input_folder_path, output_folder_path, model_path, resolution, remove_small_objects,
                                    fill_holes, do_opening, max_images_in_memory=None, canceled=Event(),
                                    progress_status=None):
    """Wrapper method to perform neural network prediction and followed up postprocessing on images in input_folder_path

    Parameters
    ----------
    input_folder_path : str, pathlib.Path
        The path to the folder containing the image slices of the 3D image stack on which prediction and post-processing should be performed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the results are stored.
    model_path : str, pathlib.Path
        The path to a trained tf.keras.model that was previously saved via tf.keras.Model.save and will be used for prediction
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    remove_small_objects : bool
        Determines whether to remove small objects in postprocessing
    fill_holes : bool
        Determines whether to fill small holes in postprocessing
    do_opening : bool
        Determines whether to perform 3D opening in postprocessing
    do_3D_smoothing : bool
        Determines whether to perform 3D binary smoothing in postprocessing
    max_images_in_memory : int, optional
        The maximum number of input images loaded into the memory and processed at once
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status


    Returns
    -------
    The path to the folder where the final output is stored.
    """

    if canceled.is_set():
        return

    output_folder_path = Path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    time_str = time.strftime("%H%M%S_")
    network_out_folder_name = time_str + "network_output"
    network_output_folder = output_folder_path / network_out_folder_name
    os.makedirs(network_output_folder, exist_ok=True)

    if max_images_in_memory is None:
        file_paths_help = get_file_list_from_directory(input_folder_path, isTif=True)
        shape_help = get_img_shape_from_filepath(file_paths_help[0])
        max_images_in_memory = get_max_images_in_memory_from_img_shape(shape_help)

    start = timer()

    networkPredictFolder(input_folder_path, network_output_folder, model_path, canceled=canceled, progress_status=progress_status)

    end = timer()
    print(f"UNet prediction took {end-start} seconds.")

    if canceled.is_set():
        return ""

    postprocess_out_folder_name = time_str + "postprocessed_network_output"
    postprocess_output_folder = output_folder_path / postprocess_out_folder_name
    os.makedirs(postprocess_output_folder, exist_ok=True)

    start = timer()

    postProcessDLRes(network_output_folder, postprocess_output_folder, resolution, remove_small_objects, fill_holes,
                     do_opening, max_images_in_memory=max_images_in_memory, canceled=canceled, progress_status=progress_status)

    end = timer()
    print(f"Postprocessing of UNet results took {end - start} seconds.")

    if canceled.is_set():
        return ""


    #get folder where final output is stored
    if not remove_small_objects and not fill_holes and not do_opening:
        final_out_folder = str(network_output_folder)

    else:
        final_out_folder = str(postprocess_output_folder)


    return final_out_folder


def write_total_volume_to_txt(output_file_path, total_volume):
    """Write the value total_volume to a txt file

    Parameters
    ----------
    total_volume: float
        The estimated total volume of the fenestrated blood vessels
    """

    out_str = "Total fenestrated blood vessel volume (computed from summing up pixel volumes) [" + u"\u03bc" + "m^3]: " + str(total_volume)

    with open(output_file_path, 'w', encoding="utf-8") as f:
        f.write(out_str)



def main():

    networkSegmentTrain()


if __name__ == "__main__":
    main()
