"""
Main python script

This script will calculate diffusion tensor maps
from a set of dicom files.

"""

import os
import sys

import matplotlib
import numpy as np

from extensions.crop_fov import crop_fov
from extensions.extensions import (  # get_xarray,
    denoise_tensor,
    export_results,
    get_cardiac_coordinates_short_axis,
    get_colourmaps,
    get_ha_line_profiles,
    get_lv_segments,
    get_snr_maps,
    query_yes_no,
)
from extensions.folder_loop_initial_setup import folder_loop_initial_setup
from extensions.get_eigensystem import get_eigensystem
from extensions.get_fa_md import get_fa_md
from extensions.get_tensor_orientation_maps import get_tensor_orientation_maps
from extensions.heart_segmentation import heart_segmentation
from extensions.image_registration import image_registration
from extensions.initial_setup import initial_setup
from extensions.read_and_pre_process_data import read_data
from extensions.remove_outliers import remove_outliers
from extensions.tensor_fittings import dipy_tensor_fit
from extensions.u_net_segmentation import get_average_images

# # for debugging numpy warnings
# np.seterr(all="raise")

# matplotlib
# matplotlib.use("macosx")
matplotlib.rcParams["toolbar"] = "None"
matplotlib.rcParams["font.size"] = 5

# script path
abspath = os.path.abspath(sys.argv[0])
script_path = os.path.dirname(abspath)

# DTCMR tailored colormaps
colormaps = get_colourmaps(script_path)

# initial setup before going into the folder loop
dti, settings, logger, log_format, all_to_be_analysed_folders = initial_setup(script_path)

# Warning about deleting DICOM data
if settings["workflow_mode"] == "anon":
    answer = query_yes_no("Are you sure you want to archive all DICOM files?")
    if answer:
        logger.info("Archiving DICOMs in an encrypted 7z file!")
    else:
        logger.error("Exiting, no permission to archive DICOM data.")
        sys.exit()


for current_folder in all_to_be_analysed_folders:
    # initial setup
    info, settings, logger = folder_loop_initial_setup(current_folder, settings, logger, log_format)

    # =========================================================
    # START processing
    # read and pre-process dicom files
    # =========================================================
    [data, info, slices] = read_data(settings, info, logger)

    # =========================================================
    # Option to perform only reading of data and anonymisation
    # =========================================================
    if settings["workflow_mode"] == "anon":
        logger.info("Anonymisation of data only mode is True. Stopping here.")
        continue

    # =========================================================
    # DWIs registration
    # =========================================================
    data, registration_image_data, ref_images = image_registration(data, slices, info, settings, logger)

    # =========================================================
    # Option to perform only registration
    # =========================================================
    if settings["workflow_mode"] == "reg":
        logger.info("Registration only mode is True. Stopping here.")
        continue

    # =========================================================
    # Remove outliers (pre-segmentation)
    # =========================================================
    if settings["remove_outliers_manually_pre"]:
        logger.info("Manual removal of outliers pre segmentation")
        [data, info, slices] = remove_outliers(
            data,
            slices,
            registration_image_data,
            settings,
            info,
            logger,
            stage="pre",
            segmentation={},
            mask=np.array([]),
        )

    # =========================================================
    # Average images
    # =========================================================
    # get average denoised normalised image for each slice
    average_images = get_average_images(
        data,
        slices,
        info,
        logger,
    )

    # =========================================================
    # Heart segmentation
    # =========================================================
    segmentation, mask_3c = heart_segmentation(
        data, average_images, slices, info["n_slices"], colormaps, settings, info, logger
    )

    # =========================================================
    # Crop FOV
    # =========================================================
    # crop the images to the heart region only
    # use the same crop for all slices and then pad with 3 pixels on all sides
    dti, data, mask_3c, segmentation, average_images, info, crop_mask = crop_fov(
        dti,
        data,
        mask_3c,
        segmentation,
        slices,
        average_images,
        registration_image_data,
        ref_images,
        info,
        logger,
        settings,
    )

    # =========================================================
    # Remove outliers (post-segmentation)
    # =========================================================
    logger.info("Manual removal of outliers post segmentation")
    [data, info, slices] = remove_outliers(
        data,
        slices,
        registration_image_data,
        settings,
        info,
        logger,
        stage="post",
        segmentation=segmentation,
        mask=mask_3c,
    )

    # =========================================================
    # Get SNR maps
    # =========================================================
    [snr, noise, snr_b0_lv, info] = get_snr_maps(data, mask_3c, average_images, slices, settings, logger, info)

    # =========================================================
    # Calculate tensor
    # =========================================================
    dti["tensor"], dti["s0"], _, _, info = dipy_tensor_fit(
        slices,
        data,
        info,
        settings,
        mask_3c,
        average_images,
        logger,
        method=settings["tensor_fit_method"],
        quick_mode=False,
    )

    # =========================================================
    # Denoise tensor with uformer models
    # =========================================================
    if settings["uformer_denoise"]:
        logger.info("Denoising tensor with uformer model: breatholds " + str(settings["uformer_breatholds"]))
        dti["tensor"] = denoise_tensor(dti["tensor"], settings)
    else:
        logger.info("Denoising tensor with uformer model is False")

    # =========================================================
    # Get Eigensystems
    # =========================================================
    dti, info = get_eigensystem(
        dti,
        slices,
        info,
        average_images,
        settings,
        mask_3c,
        logger,
    )

    # =========================================================
    # Get dti["fa"] and dti["md"] maps
    # =========================================================
    dti["md"], dti["fa"], info = get_fa_md(dti["eigenvalues"], info, mask_3c, slices, logger)

    # =========================================================
    # Get cardiac coordinates
    # =========================================================
    local_cardiac_coordinates, lv_centres, phi_matrix = get_cardiac_coordinates_short_axis(
        mask_3c, segmentation, slices, info["n_slices"], settings, dti, average_images, info
    )

    # =========================================================
    # Segment heart
    # =========================================================
    dti["lv_sectors"] = get_lv_segments(segmentation, phi_matrix, mask_3c, lv_centres, slices, logger)

    # =========================================================
    # Get dti["ha"] and dti["e2a"] maps
    # =========================================================
    dti["ha"], dti["ta"], dti["e2a"], info = get_tensor_orientation_maps(
        slices, mask_3c, local_cardiac_coordinates, dti, settings, info, logger
    )

    # =========================================================
    # Get HA line profiles
    # =========================================================
    dti["ha_line_profiles"], dti["wall_thickness"] = get_ha_line_profiles(
        dti["ha"], lv_centres, slices, mask_3c, settings, info
    )

    # =========================================================
    # Copy diffusion maps to an xarray dataset
    # =========================================================
    # ds = get_xarray(info, dti, crop_mask, slices)

    # =========================================================
    # Plot main results and save data
    # =========================================================
    export_results(data, dti, info, settings, mask_3c, slices, average_images, segmentation, colormaps, logger)

    logger.info("============================================================")
    logger.info("====================== FINISHED ============================")
    logger.info("============================================================")
