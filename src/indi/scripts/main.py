"""
Main python script

This script will calculate diffusion tensor maps
from a set of dicom files.

"""

import os
import sys

import matplotlib
import pyautogui

from indi.extensions.complex_averaging import complex_averaging
from indi.extensions.crop_fov import crop_fov, record_image_registration
from indi.extensions.extensions import (
    export_results,
    get_cardiac_coordinates_short_axis,
    get_colourmaps,
    get_ha_line_profiles,
    get_lv_segments,
    get_snr_maps,
    query_yes_no,
    remove_outliers,
    remove_slices,
)
from indi.extensions.folder_loop_initial_setup import folder_loop_initial_setup
from indi.extensions.get_eigensystem import get_eigensystem
from indi.extensions.get_fa_md import get_fa_md
from indi.extensions.get_tensor_orientation_maps import get_tensor_orientation_maps
from indi.extensions.heart_segmentation import get_average_images, heart_segmentation
from indi.extensions.image_denoising import image_denoising
from indi.extensions.image_registration import image_registration
from indi.extensions.initial_setup import initial_setup
from indi.extensions.phase_correction_for_complex_averaging import phase_correction_for_complex_averaging
from indi.extensions.read_data.read_and_pre_process_data import read_data
from indi.extensions.select_outliers import select_outliers
from indi.extensions.tensor_denoise import tensor_denoising
from indi.extensions.tensor_fittings import dipy_tensor_fit


def main():
    # # for debugging numpy warnings
    # np.seterr(all="raise")

    # matplotlib
    # better looking
    matplotlib.rcParams["font.size"] = 10
    # more suitable for manuscripts
    # matplotlib.rcParams["font.size"] = 15
    # to run efficiently
    matplotlib.rcParams["toolbar"] = "None"
    # Faster interactive matplotlib
    # if sys.platform != "darwin":
    matplotlib.use("qtagg")

    # script path
    abspath = os.path.abspath(sys.argv[0])
    script_path = os.path.dirname(abspath)

    # In TensorFlow 2.16+, to keep using Keras 2, you can first install tf_keras, and then export the environment
    # variable TF_USE_LEGACY_KERAS=1. This will direct TensorFlow 2.16+ to resolve tf.keras to the locally-installed
    # tf_keras package.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    # ITK
    # import itk

    # limit the amount of parallel threads during registration
    # itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(1)

    # DTCMR tailored colormaps
    colormaps = get_colourmaps()

    # initial setup before going into the folder loop
    dti, settings, logger, log_format, all_to_be_analysed_folders = initial_setup(script_path)

    # screen size
    settings["screen_size"] = pyautogui.size()

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
        # read and pre-process data
        # =========================================================
        [data, info, slices] = read_data(settings, info, logger)

        # =========================================================
        # Option to perform only reading of data and anonymisation
        # =========================================================
        if settings["workflow_mode"] == "anon":
            logger.info("Anonymisation of data only mode is True. Stopping here.")
            continue

        # =========================================================
        # phase correction for complex averaging
        # =========================================================
        if settings["complex_data"]:
            data = phase_correction_for_complex_averaging(data, logger, settings)

        # =========================================================
        # DWIs registration
        # =========================================================
        data, registration_image_data, ref_images, reg_mask = image_registration(data, slices, info, settings, logger)

        # =========================================================
        # NLM image denoising of all DWIs
        # =========================================================
        if settings["image_denoising"]:
            data = image_denoising(data, logger, settings, info)

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
            [data, info, slices] = select_outliers(
                data,
                slices,
                registration_image_data,
                settings,
                info,
                logger,
                stage="pre",
                segmentation={},
                mask=reg_mask,
            )
        else:
            # initialise some variables if we are not removing outliers manually
            logger.info("Manual removal of outliers pre segmentation is False")
            info["rejected_indices"] = []
            info["n_images_rejected"] = 0

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
        # Remove non segmented slices
        # =========================================================
        data, slices, segmentation = remove_slices(data, slices, segmentation, logger)

        # =========================================================
        # Crop image data
        # =========================================================
        # crop the images to the region around the segmented area only
        # use the same crop for all slices and then pad with 3 pixels on all sides
        dti, data, mask_3c, reg_mask, segmentation, average_images, info, crop_mask = crop_fov(
            dti,
            data,
            mask_3c,
            reg_mask,
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
        [data, info, slices] = select_outliers(
            data,
            slices,
            registration_image_data,
            settings,
            info,
            logger,
            stage="post",
            segmentation=segmentation,
            mask=reg_mask,
        )

        # =========================================================
        # Remove outliers and other data from table
        # =========================================================
        data, info = remove_outliers(data, info)

        # =========================================================
        # Get line profile off all remaining images to
        # assess registration
        # =========================================================
        record_image_registration(registration_image_data, ref_images, mask_3c, slices, settings, logger)

        # =========================================================
        # Get SNR maps
        # =========================================================
        [dti["snr"], noise, snr_b0_lv, info] = get_snr_maps(
            data, mask_3c, average_images, slices, settings, logger, info
        )

        # =========================================================
        # complex averaging
        # =========================================================
        if settings["complex_data"]:
            data = complex_averaging(data, logger)

        # =========================================================
        # Calculate tensor
        # =========================================================
        dti["tensor"], dti["s0"], dti["residuals_plot"], dti["residuals_map"], info = dipy_tensor_fit(
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

        if settings["tensor_denoising"]:
            # =========================================================
            # Denoise tensor with NLM
            # =========================================================
            dti = tensor_denoising(dti, slices, average_images, mask_3c, logger, settings)

        # =========================================================
        # Denoise tensor with uformer models
        # =========================================================
        if settings["uformer_denoise"]:
            try:
                from indi.extensions.uformer_denoising import denoise_tensor
            except ImportError:
                logger.error("Could not import uformer_denoising module")
                raise ImportError("Could not import uformer_denoising module. Please install torch")
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
        dti["md"], dti["fa"], dti["mode"], dti["frob_norm"], dti["mag_anisotropy"], info = get_fa_md(
            dti["eigenvalues"], info, mask_3c, slices, logger
        )

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
            dti["ha"], lv_centres, slices, mask_3c, segmentation, settings, info
        )

        # =========================================================
        # Copy diffusion maps to an xarray dataset
        # =========================================================
        # ds = get_xarray(info, dti, crop_mask, slices)

        # =========================================================
        # Plot main results and save data
        # =========================================================
        export_results(data, dti, info, settings, mask_3c, slices, average_images, segmentation, colormaps, logger)

        # =========================================================
        # Cleanup before the next folder
        # =========================================================
        logger.info("Cleaning up before the next folder")
        del (
            average_images,
            crop_mask,
            data,
            info,
            local_cardiac_coordinates,
            lv_centres,
            mask_3c,
            noise,
            phi_matrix,
            ref_images,
            registration_image_data,
            segmentation,
            slices,
            snr_b0_lv,
        )
        dti = {}

        logger.info("============================================================")
        logger.info("====================== FINISHED ============================")
        logger.info("============================================================")
