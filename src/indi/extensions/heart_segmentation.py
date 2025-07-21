import logging
import os

import cv2 as cv
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from indi.extensions.extensions import close_small_holes, get_cylindrical_coordinates_short_axis
from indi.extensions.get_tensor_orientation_maps import get_ha_e2a_maps
from indi.extensions.manual_lv_segmentation import (
    get_epi_contour,
    get_mask_from_poly,
    get_sa_contours,
    manual_lv_segmentation,
    plot_manual_lv_segmentation,
)
from indi.extensions.polygon_selector import spline_interpolate_contour
from indi.extensions.tensor_fittings import dipy_tensor_fit


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

    average_images = np.zeros([info["n_slices"], info["img_size"][0], info["img_size"][1]])
    for slice_idx in slices:
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


def heart_segmentation(
    data: pd.DataFrame,
    average_images: NDArray,
    slices: NDArray,
    n_slices: int,
    colormaps: dict,
    settings: dict,
    info: dict,
    logger: logging.Logger,
) -> tuple[dict, NDArray]:
    """
    Perform heart segmentation using U-Net and/or manual segmentation.

    This function segments the heart in each slice using a U-Net model (if enabled) and/or manual segmentation.
    It generates segmentation masks and contour information for the left ventricle (LV), and saves the results for each slice.
    The function also computes preliminary helix angle (HA) and mean diffusivity (MD) maps, and calculates residuals for quality control.

    Args:
        data (pd.DataFrame): DataFrame containing diffusion-weighted image data.
        average_images (NDArray): Array of average images for each slice.
        slices (NDArray): Array of slice indices to process.
        n_slices (int): Total number of slices.
        colormaps (dict): Dictionary of colormaps for visualization.
        settings (dict): Configuration and processing settings.
        info (dict): Dictionary with image and scan metadata.
        logger (logging.Logger): Logger for status and debug messages.

    Returns:
        tuple:
            segmentation (dict): Dictionary with segmentation contours and information for each slice.
            mask_3c (NDArray): 3-class segmentation mask array (background, LV myocardium, and other tissues).
            prelim_residuals (dict): Dictionary with preliminary residuals for each slice, indicating the quality of the tensor fit.

    """

    # =========================================================
    # Preliminary HA map
    # =========================================================

    # check if LV manual segmentation has been previously saved
    # if not calculate a prelim HA map
    prelim_ha = np.zeros((n_slices, info["img_size"][0], info["img_size"][1]))
    prelim_md = np.zeros((n_slices, info["img_size"][0], info["img_size"][1]))
    prelim_residuals = {}
    # mask is all ones here for now.
    thr_mask = np.ones((n_slices, info["img_size"][0], info["img_size"][1]))
    # loop over the slices
    for slice_idx in slices:
        if not os.path.exists(
            os.path.join(settings["session"], "manual_lv_segmentation_slice_" + str(slice_idx).zfill(3) + ".npz")
        ):
            # get cylindrical coordinates
            local_cylindrical_coordinates = get_cylindrical_coordinates_short_axis(
                thr_mask[[slice_idx], ...],
            )

            # get basic tensor fit and residuals
            tensor, _, _, _, temp_residuals, info = dipy_tensor_fit(
                [slice_idx],
                data,
                info,
                settings,
                thr_mask,
                average_images,
                logger,
                "NLLS",
                quick_mode=True,
            )
            # get basic HA and MD maps
            prelim_eigenvalues, prelim_eigenvectors = np.linalg.eigh(tensor[[slice_idx], ...])
            prelim_ha[slice_idx], _, _ = get_ha_e2a_maps(
                thr_mask[[slice_idx], ...],
                local_cylindrical_coordinates,
                prelim_eigenvectors,
            )
            prelim_md[slice_idx] = np.mean(prelim_eigenvalues, axis=-1)

            # threshold preliminary MD and HA maps
            prelim_ha[slice_idx] = prelim_ha[slice_idx] * thr_mask[slice_idx]
            prelim_md[slice_idx] = 1e3 * prelim_md[slice_idx] * thr_mask[slice_idx]

            # preliminary residuals
            prelim_residuals[slice_idx] = temp_residuals[slice_idx]

        else:
            # get residuals of a preliminary basic tensor fit
            _, _, _, _, temp_residuals, info = dipy_tensor_fit(
                [slice_idx],
                data,
                info,
                settings,
                thr_mask,
                average_images,
                logger,
                "NLLS",
                quick_mode=True,
            )

            # preliminary residuals
            prelim_residuals[slice_idx] = temp_residuals[slice_idx]

    # =========================================================
    # LV segmentation
    # =========================================================
    # =========================================================
    # U-Net segmentation
    # =========================================================
    if settings["u_net_segmentation"]:
        logger.info("U-Net segmentation is True")
        try:
            from indi.extensions.u_net_segmentation import plot_segmentation_unet, u_net_segment_heart
        except ModuleNotFoundError:
            logger.error("U-Net segmentation is not available. Please install the required packages.")
            raise ModuleNotFoundError("U-Net segmentation is not available. Please install tensorflow.")

        # check if U-Net segmentation has been previously saved
        if os.path.exists(os.path.join(settings["session"], "u_net_segmentation.npz")):
            # load segmentations
            logger.info("U-Net segmentation previously saved. Loading mask...")
            npzfile = np.load(os.path.join(settings["session"], "u_net_segmentation.npz"))
            mask_3c = npzfile["mask_3c"]
            if settings["debug"]:
                plot_segmentation_unet(info["n_slices"], slices, mask_3c, average_images, settings)
        else:
            # segment heart with U-Net
            logger.info("U-Net ensemble size: " + str(settings["n_ensemble"]))
            mask_3c = u_net_segment_heart(average_images, slices, info["n_slices"], settings, logger)
            logger.info("U-Net ensemble segmentation done")
    else:
        logger.info("U-Net segmentation is False")
        mask_3c = np.zeros((info["n_slices"], info["img_size"][0], info["img_size"][1]), dtype="uint8")

    # =========================================================
    # Manual LV segmentation
    # =========================================================
    # dictionary to store the segmentation splines and insertion points for each slice
    segmentation = {}
    # loop over each slice
    for slice_idx in slices:
        # check if LV manual segmentation has been previously saved
        if os.path.exists(
            os.path.join(settings["session"], "manual_lv_segmentation_slice_" + str(slice_idx).zfill(3) + ".npz")
        ):
            # load segmentations
            logger.info("Manual LV segmentation previously saved for slice: " + str(slice_idx) + ", loading mask...")
            npzfile = np.load(
                os.path.join(settings["session"], "manual_lv_segmentation_slice_" + str(slice_idx).zfill(3) + ".npz"),
                allow_pickle=True,
            )
            mask_3c[slice_idx] = npzfile["mask_3c"]
            segmentation[slice_idx] = npzfile["segmentation"]
            segmentation[slice_idx] = segmentation[slice_idx].item()

            # if there is no epicardial border defined, mark this slice to be removed in the dataframe
            if segmentation[slice_idx]["epicardium"].size == 0:
                data.loc[data["slice_integer"] == slice_idx, "to_be_removed"] = True

        else:
            # manual LV segmentation
            logger.info("Manual LV segmentation for slice: " + str(slice_idx))
            segmentation[slice_idx], thr_mask[slice_idx] = manual_lv_segmentation(
                mask_3c[slice_idx],
                average_images[slice_idx],
                prelim_ha[slice_idx],
                prelim_md[slice_idx],
                100,
                settings,
                colormaps,
                slice_idx,
                slices,
            )

            # define the final mask_3c
            if segmentation[slice_idx]["epicardium"].size != 0:
                mask_epi = get_mask_from_poly(
                    segmentation[slice_idx]["epicardium"].astype(np.int32),
                    mask_3c[slice_idx].shape,
                )
            else:
                mask_epi = np.zeros(mask_3c[slice_idx].shape, dtype="uint8")
                # mark this slice to be removed in the dataframe
                data.loc[data["slice_integer"] == slice_idx, "to_be_removed"] = True

            if segmentation[slice_idx]["epicardium"].size != 0:
                # only do the following if there is an epicardial border defined, otherwise this slice will be removed
                if segmentation[slice_idx]["endocardium"].size != 0:
                    mask_endo = get_mask_from_poly(
                        segmentation[slice_idx]["endocardium"].astype(np.int32),
                        mask_3c[slice_idx].shape,
                    )
                else:
                    mask_endo = np.zeros(mask_3c[slice_idx].shape, dtype="uint8")

                # we need to remove the mask pixels that have been thresholded out
                mask_epi *= thr_mask[slice_idx]

                if segmentation[slice_idx]["endocardium"].size != 0:
                    # erode endo mask in order to keep the endo line inside the myocardial ROI
                    kernel = np.ones((2, 2), np.uint8)
                    mask_endo = cv.erode(mask_endo, kernel, iterations=1)
                    mask_endo *= thr_mask[slice_idx]

                mask_lv = mask_epi - mask_endo
                if segmentation[slice_idx]["endocardium"].size != 0:
                    epi_contour, endo_contour = get_sa_contours(mask_lv)
                else:
                    epi_contour = get_epi_contour(mask_lv)
                    endo_contour = np.array([])

                epi_len = len(epi_contour)
                endo_len = len(endo_contour)
                epi_contour = spline_interpolate_contour(epi_contour, 20, join_ends=False)
                epi_contour = spline_interpolate_contour(epi_contour, epi_len, join_ends=False)

                if segmentation[slice_idx]["endocardium"].size != 0:
                    endo_contour = spline_interpolate_contour(endo_contour, 20, join_ends=False)
                    endo_contour = spline_interpolate_contour(endo_contour, endo_len, join_ends=False)

                segmentation[slice_idx]["epicardium"] = epi_contour
                if segmentation[slice_idx]["endocardium"].size != 0:
                    segmentation[slice_idx]["endocardium"] = endo_contour

                all_channel_mask = mask_3c[slice_idx].copy()
                all_channel_mask[all_channel_mask == 1] = 0
                all_channel_mask = all_channel_mask + mask_lv
                all_channel_mask[all_channel_mask == 3] = 1
                mask_3c[slice_idx] = all_channel_mask

                # sometimes there are holes between the myocardium and rest of the heart mask, fill them here
                mask_3c[slice_idx] = close_small_holes(mask_3c[slice_idx])

            # save mask and segmentation
            np.savez_compressed(
                os.path.join(settings["session"], "manual_lv_segmentation_slice_" + str(slice_idx).zfill(3) + ".npz"),
                mask_3c=mask_3c[slice_idx],
                segmentation=segmentation[slice_idx],
            )

        # turn to nans the pixels that are not part of the mask for the residuals and average all rows and cols
        # prelim_residuals is the average residual (one value) of the all myocardial pixels per image
        prelim_residuals[slice_idx] = prelim_residuals[slice_idx].astype(np.float32)
        prelim_residuals[slice_idx][mask_3c[slice_idx] == 0] = np.nan
        prelim_residuals[slice_idx] = np.nanmean(prelim_residuals[slice_idx], axis=(0, 1))

    if settings["debug"]:
        plot_manual_lv_segmentation(
            n_slices,
            slices,
            segmentation,
            average_images,
            mask_3c,
            settings,
            "lv_manual_mask",
            settings["debug_folder"],
        )

    logger.info("All manual LV segmentation done")

    return segmentation, mask_3c, prelim_residuals
