import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from scipy import ndimage
from tensorflow.keras import models

from extensions.extensions import clean_mask, crop_pad_rotate_array


def denoise_images(img: NDArray, settings: dict, slices: NDArray) -> NDArray:
    """
    Denoise images with non-local means algorithm

    Parameters
    ----------
    img: image to be denoised
    settings: dict with useful info
    slices: Numpy array with slice position strings

    Returns
    -------
    denoised image
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma

    # input image dimensions (may be more than one image (slice))
    img_dims = img.shape
    n_slices = img_dims[0]

    # nlm config
    patch_kw = dict(
        patch_size=5,  # 5x5 patches
        patch_distance=6,  # 13x13 search area
        channel_axis=None,
    )

    for slice_idx in range(n_slices):
        c_img = img[slice_idx]

        # estimate the noise standard deviation from the noisy image
        sigma_est = np.mean(estimate_sigma(c_img, channel_axis=None))
        # fast algorithm, sigma provided
        denoised_img = denoise_nl_means(c_img, h=5 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)

        img[slice_idx] = denoised_img

    if settings["debug"]:
        for slice_idx in slices:
            plt.figure(figsize=(5, 5))
            plt.imshow(img[slice_idx], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    settings["debug_folder"],
                    "denoised_u_net_refs_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            plt.close()

    return img


def u_net_segmentation_3ch(img: NDArray, n_slices: int, settings: dict, logger: logging.Logger) -> NDArray:
    """
    U-Net segmentation:
    - 3 channel (background = 0, LV myocardium, rest of heart)

    Parameters
    ----------
    img: image to be segmented
    n_slices: number of slices
    settings
    logger

    Returns
    -------
    segmentation mask
    """

    # we need to make sure the input array has the exact size
    # if not then pad or crop to the correct size
    correct_size = [img.shape[0], 256, 96]
    img_dims_original = list(img.shape)
    if correct_size[1] != img_dims_original[1] or correct_size[2] != img_dims_original[2]:
        img = crop_pad_rotate_array(img, correct_size, False)
        logger.debug("Resising image to the correct size for U-Net segmentation: " + str(correct_size))

    # Loss functions used by the U-Net model
    # @tf.function
    def soft_dice_loss(y_true: object, y_pred: object, epsilon: object = 1e-6) -> object:
        """
        Soft dice loss calculation for arbitrary batch size, number of classes,
        and number of spatial dimensions.
        Assumes the `channels_last` format.

        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output,
            must sum to 1 over c channel (such as after softmax)
            epsilon: Used for numerical stability to avoid divide by zero errors

        # References
            V-Net: Fully Convolutional Neural Networks for
            Volumetric Medical Image Segmentation
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

            Adapted from
            https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022

        Returns
        -------
        Average Dice loss over classes and batch
        """

        # skip the batch and class axis for calculating Dice score
        # axes = tuple(range(1, len(y_pred.shape) - 1))
        numerator = 2.0 * tf.reduce_sum(input_tensor=tf.math.multiply(y_pred, y_true), axis=[1, 2])
        denominator = tf.reduce_sum(input_tensor=tf.math.square(y_pred), axis=[1, 2]) + tf.reduce_sum(
            input_tensor=tf.math.square(y_true), axis=[1, 2]
        )

        return 1 - tf.math.reduce_mean(
            input_tensor=tf.math.divide(numerator, (denominator + epsilon))
        )  # average over classes and batch

    # myocardium dice in tensorflow
    # @tf.function  # in order to be able to add breakpoints inside this function
    def dice_coeff(y_true: object, y_pred: object) -> object:
        """
        Parameters
        ----------
        y_true: ground truth segmentation mask
        y_pred: predicted segmentation mask

        Returns
        -------
        Dice coefficient for the myocardium only
        """
        y_true = y_true[:, :, :, 1]
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=3)
        y_pred = tf.reshape(y_pred, [-1])

        binary_true = tf.cast(y_true, tf.int32)
        binary_pred = tf.where(
            tf.equal(y_pred, tf.constant(1, dtype="int64")),
            tf.constant(1),
            tf.constant(0),
        )

        intersection = tf.reduce_sum(binary_true * binary_pred)
        score = (tf.constant(2) * intersection) / (tf.reduce_sum(binary_true) + tf.reduce_sum(binary_pred))

        return score

    # loading the ensemble of models and their predictions
    common_path = "/usr/local/dtcmr/unet_ensemble/"
    n_ensemble = settings["n_ensemble"]
    n_classes = 3
    predicted_label_3c = np.zeros(shape=(n_ensemble, img.shape[0], img.shape[1], img.shape[2], n_classes))
    img = np.expand_dims(img, axis=-1)

    # loop over all models
    for idx in range(n_ensemble):
        cnn_name = "unet_find_heart_full_fov_3c_" + str(idx) + ".hdf5"
        cnn_name = os.path.join(common_path, cnn_name)
        model = models.load_model(
            cnn_name,
            custom_objects={"soft_dice_loss": soft_dice_loss, "dice_coeff": dice_coeff},
        )

        # predict masks
        predicted_label_3c[idx] = model.predict(img)

    # get average prediction of all models
    mean_predicted_labels = np.mean(predicted_label_3c, axis=0)

    # We need to discretise the labels to the highest probability class
    mask_3c = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(n_slices):
        current_predicted_labels_3c = mean_predicted_labels[i, :, :, :]
        mask_3c[i, :, :] = np.argmax(current_predicted_labels_3c, axis=2)

    mask_dims = mask_3c.shape
    # revert size of U-Net mask to the original image size
    if mask_dims[1] != img_dims_original[1] or mask_dims[2] != img_dims_original[2]:
        mask_3c = crop_pad_rotate_array(mask_3c, img_dims_original, False)
        logger.debug("Reverting back image to the original size: " + str(img_dims_original))

    return mask_3c


def get_average_images(
    data: pd.DataFrame,
    slices: NDArray,
    info,
    logger: logging.Logger,
) -> NDArray:
    """
    Get average denoised and normalised image for each slice

    Parameters
    ----------
    data: database with DWIs
    slices: array with slice locations string
    info: dict
    logger

    Returns
    -------
    NDarray with average images

    """

    average_images = {}
    for i, slice_idx in enumerate(slices):
        # dataframe with current slice
        c_df = data.loc[data["slice_integer"] == slice_idx].copy()

        # drop rejected images (if they exist)
        if "to_be_removed" in c_df:
            c_df = c_df[c_df["to_be_removed"] == False]

        # stack of images
        img_stack = np.stack(c_df["image"], axis=0)

        # average image
        mean_img = np.mean(img_stack, axis=0)

        # normalise image
        mean_img = mean_img * (1 / mean_img.max())

        average_images[slice_idx] = mean_img

    logger.info("Average images calculated")

    return average_images


def plot_segmentation_unet(n_slices: int, slices: NDArray, mask_3c: NDArray, average_images: NDArray, settings: dict):
    """
    Plot the masking of the heart and LV given by the U-Net.

    Parameters
    ----------
    n_slices : int
        number of slices
    mask_3c : NDArray
        U-Net segmentation mask
    average_images : NDArray
        average denoised and normalised images for each slice
    settings : dict
    """
    for slice_idx in slices:
        # get the borders of the mask
        mask = np.full(np.shape(mask_3c[slice_idx]), False)
        struct = ndimage.generate_binary_structure(2, 2)

        # myocardial border
        myo_border = mask
        myo_border[mask_3c[slice_idx] == 1] = True
        erode = ndimage.binary_erosion(myo_border, struct)
        myo_border = myo_border ^ erode
        myo_border_pts = (myo_border > 0).nonzero()

        # heart border
        heart_border = mask
        heart_border[mask_3c[slice_idx] == 2] = True
        erode = ndimage.binary_erosion(heart_border, struct)
        heart_border = heart_border ^ erode
        heart_border = (heart_border > 0).nonzero()

        # plot average images and respective masks
        plt.figure(figsize=(5, 5))
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.scatter(
            heart_border[1],
            heart_border[0],
            marker=".",
            s=2,
            color="tab:blue",
            alpha=0.5,
        )
        plt.scatter(
            myo_border_pts[1],
            myo_border_pts[0],
            marker=".",
            s=2,
            color="tab:red",
            alpha=0.5,
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(
                settings["debug_folder"],
                "unet_masks_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def u_net_segment_heart(
    average_images: NDArray,
    slices: NDArray,
    n_slices: int,
    settings: dict,
    logger: logging.Logger,
) -> NDArray:
    """
    Parameters
    ----------
    average_images: average images array
    slices: array with slice locations string
    n_slices: int number of slices
    settings: dictionary with useful info
    logger

    Returns
    -------
    Segmented mask and average image for each slice
    """
    mask_3c = u_net_segmentation_3ch(average_images, n_slices, settings, logger)

    mask_3c = clean_mask(mask_3c)

    if settings["debug"]:
        plot_segmentation_unet(n_slices, slices, mask_3c, average_images, settings)

    # save mask to an npz file
    np.savez_compressed(
        os.path.join(settings["session"], "u_net_segmentation.npz"),
        mask_3c=mask_3c,
    )

    return mask_3c
