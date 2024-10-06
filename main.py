"""
Main python script

This script will calculate diffusion tensor maps
from a set of dicom files.

"""

import os
import sys

import matplotlib
import pyautogui

from extensions.complex_averaging import complex_averaging
from extensions.crop.crop import Crop

# from extensions.crop.crop import Crop
from extensions.crop_fov import crop_fov, record_image_registration
from extensions.extensions import (
    denoise_tensor,
    export_results,
    get_colourmaps,
    get_snr_maps,
    query_yes_no,
    remove_outliers,
    remove_slices,
)
from extensions.folder_loop_initial_setup import folder_loop_initial_setup
from extensions.image_registration import image_registration
from extensions.initial_setup import initial_setup
from extensions.metrics.metrics import Metrics
from extensions.read_data.read_and_pre_process_data import read_data
from extensions.registration_ex_vivo.registration import RegistrationExVivo
from extensions.rotation.rotation import Rotation
from extensions.segmentation.heart_segmentation import HeartSegmentation, ExternalSegmentation
from extensions.select_outliers.select_outliers import SelectOutliers  # , manual_image_removal
from extensions.tensor_fittings.tensor_fittings import TensorFit
from extensions.u_net_segmentation import get_average_images

# # for debugging numpy warnings
# np.seterr(all="raise")

# matplotlib
# better looking
matplotlib.rcParams["font.size"] = 5
# more suitable for manuscripts
# matplotlib.rcParams["font.size"] = 15
# to run efficiently
matplotlib.rcParams["toolbar"] = "None"
# Faster interactive matplotlib
matplotlib.use("qtagg")

# script path
abspath = os.path.abspath(sys.argv[0])
script_path = os.path.dirname(abspath)

# ITK
# import itk

# limit the amount of parallel threads during registration
# itk.MultiThreaderBase.SetGlobalMaximumNumberOfThreads(1)

# DTCMR tailored colormaps
colormaps = get_colourmaps(script_path)

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

    # for i, image in enumerate(data["image"]):
    #     plt.imsave(os.path.join(settings["debug_folder"], f"{i:02d}.png"), image, cmap="gray")
    # =========================================================
    # Crop data
    # =========================================================
    if settings["ex_vivo"]:
        logger.info("Ex-vivo: Cropping data to the region of interest")
        context = {"data": data, "slices": slices, "info": info}
        Crop(context, settings, logger).run()
        data = context["data"]
        slices = context["slices"]
        info = context["info"]

    # =========================================================
    # DWIs registration
    # =========================================================
    if settings["ex_vivo"]:
        logger.info("Ex-vivo: Using ex-vivo registration: " + settings["ex_vivo_registration"])
        context = {"data": data, "info": info}
        RegistrationExVivo(context, settings, logger).run()
        data = context["data"]
        reg_mask = context["reg_mask"]
        registration_image_data = None
        ref_images = context["ref_images"]
        dti["snr"] = context["snr"]
        snr_b0_lv = context["snr_b0_lv"]
        noise = context["noise"]
        info = context["info"]

    else:
        data, registration_image_data, ref_images, reg_mask = image_registration(data, slices, info, settings, logger)

    # =========================================================
    # Option to perform only registration
    # =========================================================
    if settings["workflow_mode"] == "reg":
        logger.info("Registration only mode is True. Stopping here.")
        continue

    # =========================================================
    # Rotation if ex-vivo
    # =========================================================
    if settings["ex_vivo"] and settings["rotate"]:
        logger.info("Ex-vivo rotation is True")
        context = {"data": data, "info": info, "slices": slices, "ref_images": ref_images, "dti": dti}
        Rotation(context, settings, logger).run()
        data = context["data"]
        slices = context["slices"]
        reg_mask = context["reg_mask"]
        ref_images = context["ref_images"]
        info = context["info"]
        dti = context["dti"]
        # data, slices, info = rotate_data(data, slices, info, settings, logger)

    # =========================================================
    # Remove outliers (pre-segmentation)
    # =========================================================
    if settings["remove_outliers_manually_pre"]:
        logger.info("Manual removal of outliers pre segmentation")
        context = {
            "data": data,
            "info": info,
            "slices": slices,
            "registration_image_data": registration_image_data,
            "stage": "pre",
            "mask": reg_mask,
            "segmentation": {},
        }
        SelectOutliers(context, settings, logger).run()
        data = context["data"]
        info = context["info"]
        slices = context["slices"]
    else:
        # initialise some variables if we are not removing outliers manually
        logger.info("Manual removal of outliers pre segmentation is False")
        data["to_be_removed"] = False
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
    if not settings["ex_vivo"]:
        context = {
            "data": data,
            "info": info,
            "average_images": average_images,
            "slices": slices,
            "colormaps": colormaps,
        }
        HeartSegmentation(context, settings, logger).run()
        data = context["data"]
        info = context["info"]
        slices = context["slices"]
        segmentation = context["segmentation"]
        mask_3c = context["mask_3c"]

    else:
        context = {
            "data": data,
            "info": info,
            "average_images": average_images,
            "slices": slices,
            "colormaps": colormaps,
        }
        ExternalSegmentation(context, settings, logger).run()
        data = context["data"]
        info = context["info"]
        slices = context["slices"]
        segmentation = context["segmentation"]
        mask_3c = context["mask_3c"]

    # =========================================================
    # Remove non segmented slices
    # =========================================================
    data, info, slices, segmentation, mask_3c = remove_slices(data, info, slices, segmentation, mask_3c, logger)

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
    if not settings["ex_vivo"]:
        logger.info("Manual removal of outliers post segmentation")
        context = {
            "data": data,
            "info": info,
            "slices": slices,
            "registration_image_data": registration_image_data,
            "stage": "post",
            "mask": reg_mask,
            "segmentation": segmentation,
        }
        SelectOutliers(context, settings, logger).run()
        data = context["data"]
        info = context["info"]
        slices = context["slices"]

        # =========================================================
        # Remove outliers from table
        # =========================================================
        data, info = remove_outliers(data, info)

    # =========================================================
    # Get line profile off all remaining images to
    # assess registration
    # =========================================================
    if not settings["ex_vivo"]:
        record_image_registration(registration_image_data, ref_images, mask_3c, slices, settings, logger)

    # =========================================================
    # Get SNR maps
    # =========================================================
    if not settings["ex_vivo"]:  # SNR maps for ex-vivo are calculated in the registration step
        [dti["snr"], noise, snr_b0_lv, info] = get_snr_maps(
            data, mask_3c, average_images, slices, settings, logger, info
        )

    # =========================================================
    # complex averaging
    # =========================================================
    if not settings["ex_vivo"]:
        if settings["complex_data"]:
            data = complex_averaging(data, logger)

    # =========================================================
    # Calculate tensor
    # =========================================================
    context = {"data": data, "info": info, "slices": slices, "mask_3c": mask_3c, "average_images": average_images}
    TensorFit(context, settings, logger, method=settings["tensor_fit_method"], quick_mode=False).run()
    dti["tensor"] = context["dti"]["tensor"]
    dti["s0"] = context["dti"]["s0"]
    dti["residuals_plot"] = context["dti"]["residuals_plot"]
    dti["residuals_map"] = context["dti"]["residuals_map"]
    info = context["info"]

    # dti["tensor"], dti["s0"], dti["residuals_plot"], dti["residuals_map"], info = dipy_tensor_fit(
    #     slices,
    #     data,
    #     info,
    #     settings,
    #     mask_3c,
    #     average_images,
    #     logger,
    #     method=settings["tensor_fit_method"],
    #     quick_mode=False,
    # )

    # =========================================================
    # Denoise tensor with uformer models
    # =========================================================
    if settings["uformer_denoise"]:
        logger.info("Denoising tensor with uformer model: breatholds " + str(settings["uformer_breatholds"]))
        dti["tensor"] = denoise_tensor(dti["tensor"], settings)
    else:
        logger.info("Denoising tensor with uformer model is False")

    # =========================================================
    # Tensor metrics
    # =========================================================
    # LV Metrics/Maps
    context = {
        "data": data,
        "info": info,
        "slices": slices,
        "dti": dti,
        "segmentation": segmentation,
        "mask_3c": mask_3c,
        "average_images": average_images,
    }
    Metrics(context, settings, logger).run()
    dti = context["dti"]
    info = context["info"]

    # =========================================================
    # Plot main results and save data
    # =========================================================
    export_results(data, dti, info, settings, mask_3c, slices, average_images, segmentation, colormaps, logger)

    # =========================================================
    # Cleanup before the next folder
    # =========================================================
    logger.info("Cleaning up...")
    del (
        average_images,
        crop_mask,
        data,
        info,
        mask_3c,
        noise,
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
