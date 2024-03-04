import logging
import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from extensions.extensions import close_small_holes, get_cylindrical_coordinates_short_axis
from extensions.get_tensor_orientation_maps import get_ha_e2a_maps
from extensions.manual_lv_segmentation import manual_lv_segmentation, plot_manual_lv_segmentation
from extensions.tensor_fittings import dipy_tensor_fit
from extensions.u_net_segmentation import plot_segmentation_unet, u_net_segment_heart


def heart_segmentation(
    data: pd.DataFrame,
    average_images: NDArray,
    slices: NDArray,
    n_slices: int,
    colormaps: dict,
    settings: dict,
    info: dict,
    logger: logging.Logger,
) -> [dict, NDArray]:
    """
    Heart segmentation

    Parameters
    ----------
    data
    average_images
    slices
    colormaps
    settings
    info
    logger

    Returns
    -------
    segmentation
    mask_3c

    """

    # =========================================================
    # Preliminary HA map
    # =========================================================

    # check if LV manual segmentation has been previously saved
    # if not calculate a prelim HA map
    if not os.path.exists(os.path.join(settings["session"], "manual_lv_segmentation.npz")):
        # create a threshold mask
        # _, thr_mask, _ = clean_image(average_images, factor=settings["threshold_strength"], blur=False)
        # mask is all ones here for now.
        thr_mask = np.ones(average_images.shape, dtype="uint8")

        # check if LV manual segmentation has been previously saved
        # if not calculate a prelim HA map
        if not os.path.exists(os.path.join(settings["session"], "manual_lv_segmentation.npz")):
            local_cylindrical_coordinates = get_cylindrical_coordinates_short_axis(
                thr_mask,
                average_images,
                slices,
                n_slices,
                settings,
                info,
            )

            tensor, _, _, _, info = dipy_tensor_fit(
                slices,
                data,
                info,
                settings,
                thr_mask,
                logger,
                "LS",
                quick_mode=True,
            )
            _, prelim_eigenvectors = np.linalg.eigh(tensor)
            prelim_ha, _, _ = get_ha_e2a_maps(
                thr_mask,
                local_cylindrical_coordinates,
                prelim_eigenvectors,
            )

            # threshold prelim HA map
            prelim_ha *= thr_mask

    # =========================================================
    # LV segmentation
    # =========================================================
    # U-Net segmentation
    if settings["u_net_segmentation"]:
        logger.info("U-Net segmentation is True")
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

    # Manual LV segmentation
    # check if LV manual segmentation has been previously saved
    if os.path.exists(os.path.join(settings["session"], "manual_lv_segmentation.npz")):
        # load segmentations
        logger.info("Manual LV segmentation previously saved. Loading mask...")
        npzfile = np.load(os.path.join(settings["session"], "manual_lv_segmentation.npz"), allow_pickle=True)
        mask_3c = npzfile["mask_3c"]
        segmentation = npzfile["segmentation"]
        segmentation = segmentation.item()
        if settings["debug"]:
            plot_manual_lv_segmentation(
                info["n_slices"],
                slices,
                segmentation,
                average_images,
                mask_3c,
                settings,
                "lv_manual_mask",
                settings["debug_folder"],
            )
    else:
        # manual LV segmentation
        logger.info("Manual LV segmentation...")
        segmentation, mask_3c = manual_lv_segmentation(
            mask_3c,
            slices,
            average_images,
            prelim_ha,
            10,
            settings,
            colormaps,
        )
        logger.info("Manual LV segmentation done")

    # sometimes there are holes between the myocardium and rest of the heart mask, fill them here
    for slice_idx in slices:
        mask_3c[slice_idx] = close_small_holes(mask_3c[slice_idx])

    return segmentation, mask_3c
