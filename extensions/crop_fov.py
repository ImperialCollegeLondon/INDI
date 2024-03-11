import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.util import compare_images


def crop_images(
    dti: dict,
    data: pd.DataFrame,
    mask_3c: NDArray,
    segmentation: dict,
    slices: NDArray,
    average_images: NDArray,
    ref_images: NDArray,
    info: dict,
    logger: logging.Logger,
    settings: dict,
) -> [dict, pd.DataFrame, NDArray, dict, NDArray, dict, dict, NDArray]:
    """
    Crop images to the heart region as defined by the segmentation

    Parameters
    ----------
    dti: dict with DTI maps
    data: dataframe with the diffusion images
    mask_3c: segmentation mask of the heart
    segmentation: segmentation information
    slices: array with slice positions
    average_images: average image of each slice
    ref_images: reference images
    info: useful info dictionary
    logger: logger
    settings: dict

    Returns
    -------
    dti: dict with dti maps
    data: dataframe with the diffusion images now cropped.
    mask_3c: U-Net mask of the heart now cropped.
    segmentation: dict with segmentation info
    average_images: average image of each slice now cropped.
    ref_images: reference_images used in the registration
    info: dictionary with useful stuff, here updates the image size to the cropped one.
    crop_mask: logical mask with the crop.
    """

    n_entries, _ = data.shape

    global_crop_mask = sum(mask_3c[i] for i in range(mask_3c.shape[0]))
    crop_mask = global_crop_mask != 0

    # pad mask but beware of not going out of the FOV limits
    pad_len = 5
    x_pad = np.ix_(crop_mask.any(1))
    y_pad = np.ix_(crop_mask.any(0))
    pad_len_x = min([pad_len, x_pad[0][0], crop_mask.shape[0] - x_pad[0][-1]])
    pad_len_y = min([pad_len, y_pad[0][0], crop_mask.shape[1] - y_pad[0][-1]])
    crop_mask[
        x_pad[0][0] - pad_len_x : x_pad[0][-1] + pad_len_x,
        y_pad[0][0] - pad_len_y : y_pad[0][-1] + pad_len_y,
    ] = True

    # crop the U-Net masks and average images
    mask_3c = mask_3c[np.ix_(np.repeat(True, info["n_slices"]), crop_mask.any(1), crop_mask.any(0))]
    average_images = average_images[np.ix_(np.repeat(True, info["n_slices"]), crop_mask.any(1), crop_mask.any(0))]
    for slice_str in slices:
        if ref_images[slice_str]["image"] is not None:
            ref_images[slice_str]["image"] = ref_images[slice_str]["image"][np.ix_(crop_mask.any(1), crop_mask.any(0))]

    # modify the segmentation data points
    pos = np.where(crop_mask)
    # coordinates of corner in [line column]
    first_corner = np.array([pos[0][0], pos[1][0]])
    for slice_idx, slice_name in enumerate(segmentation):
        segmentation[slice_name]["anterior_ip"] = np.array(segmentation[slice_name]["anterior_ip"]) - np.flip(
            first_corner
        )
        segmentation[slice_name]["inferior_ip"] = np.array(segmentation[slice_name]["inferior_ip"]) - np.flip(
            first_corner
        )
        segmentation[slice_name]["epicardium"] = np.array(segmentation[slice_name]["epicardium"]) - np.flip(
            first_corner
        )
        segmentation[slice_name]["endocardium"] = np.array(segmentation[slice_name]["endocardium"]) - np.flip(
            first_corner
        )

    # add crop info to info dictionary
    temp_val = list(first_corner)
    info["crop_corner"] = [int(i) for i in temp_val]

    # if settings["debug"]:
    #     plot_manual_lv_segmentation(
    #         info["n_slices"],
    #         slices,
    #         segmentation,
    #         average_images,
    #         mask_3c,
    #         settings,
    #         "cropped_lv_mask",
    #         os.path.join(settings["results"], "results_b"),
    #     )

    # crop the diffusion images
    for i in range(n_entries):
        c_slice_position = data.loc[i, "slice_integer"]
        background_mask = np.copy(mask_3c[c_slice_position])
        background_mask[background_mask > 0] = 1
        data.at[i, "image"] = data.loc[i, "image"][np.ix_(crop_mask.any(1), crop_mask.any(0))]
        # data.at[i, "image"] = data.loc[i, "image"] * background_mask

    # update image size
    info["original_img_size"] = info["img_size"]
    info["img_size"] = data.loc[0, "image"].shape
    logger.debug("Image size updated to: " + str(info["img_size"]))

    # record the crop positions in the info dictionary
    dti["crop_mask"] = crop_mask

    return dti, data, mask_3c, segmentation, average_images, ref_images, info, crop_mask


def record_image_registration(
    registration_image_data: dict,
    ref_images: NDArray,
    mask: NDArray,
    slices: NDArray,
    settings: dict,
    logger: logging.Logger,
):
    """
    Save registration results as line profiles, animated GIF,
    and optionally as montages for each frame

    Parameters
    ----------
    registration_image_data: dict with registration info: images before and after registration,
        and also displacement field, and grid image with displacement field applied to it
    img_post_reg: images after registration
    mask: U-Net mask of the heart
    slices: array with slice position arrays
    settings: dictionary with useful info
    logger: logger
    """

    # import imageio
    # from skimage import color

    lv_mask = np.zeros(mask.shape)
    lv_mask[mask == 1] = 1

    for slice_idx in slices:
        # find the LV centre
        count = (lv_mask[slice_idx] == 1).sum()
        x_center, y_center = np.round(np.argwhere(lv_mask[slice_idx] == 1).sum(0) / count)

        # store the line profiles for the images before and after registration
        store_h_lp_pre = registration_image_data["img_pre_reg"][slice_idx][
            :, int(x_center - 1) : int(x_center + 2) :, :
        ]
        store_h_lp_pre = np.mean(store_h_lp_pre, axis=1)
        store_h_lp_post = registration_image_data["img_post_reg"][slice_idx][
            :, int(x_center - 1) : int(x_center + 2) :, :
        ]
        store_h_lp_post = np.mean(store_h_lp_post, axis=1)

        store_v_lp_pre = registration_image_data["img_pre_reg"][slice_idx][:, :, int(y_center - 1) : int(y_center + 2)]
        store_v_lp_pre = np.mean(store_v_lp_pre, axis=2)
        store_v_lp_post = registration_image_data["img_post_reg"][slice_idx][
            :, :, int(y_center - 1) : int(y_center + 2)
        ]
        store_v_lp_post = np.mean(store_v_lp_post, axis=2)

        plt.figure(figsize=(5, 5))
        plt.subplot(2, 2, 1)
        plt.imshow(store_h_lp_pre)
        plt.axis("off")
        plt.title("horizontal pre", fontsize=7)
        plt.subplot(2, 2, 3)
        plt.imshow(store_v_lp_pre)
        plt.axis("off")
        plt.title("vertical pre", fontsize=7)
        plt.subplot(2, 2, 2)
        plt.imshow(store_h_lp_post)
        plt.axis("off")
        plt.title("horizontal post", fontsize=7)
        plt.subplot(2, 2, 4)
        plt.imshow(store_v_lp_post)
        plt.axis("off")
        plt.title("vertical post", fontsize=7)
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "registration_line_profiles_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=200,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

    # if settings["debug"]:
    #     # save the registration results for each slice as an animated gif
    #     # this time with mask
    #     for slice_idx in slices:
    #         post_reg_array = registration_image_data["img_post_reg"][slice_idx]
    #         gif_images = []
    #         for idx in range(post_reg_array.shape[0]):
    #             c_img = post_reg_array[idx] * (1 / post_reg_array[idx].max())
    #             c_img_mask = color.label2rgb(mask[slice_idx], c_img, bg_label=0, alpha=0.05)
    #             gif_images.append(c_img_mask)
    #         gif_images = gif_images / np.amax(gif_images) * 255
    #         gif_images = np.array(gif_images, dtype=np.uint8)
    #         imageio.mimsave(
    #             os.path.join(
    #                 settings["debug_folder"],
    #                 "registration_mask_slice_" + str(slice_idx).zfill(2) + ".gif",
    #             ),
    #             gif_images,
    #             duration=0.3,
    #             loop=0,
    #         )

    if settings["debug"] and settings["registration_extra_debug"] and settings["registration"] != "elastix_groupwise":
        if not os.path.exists(os.path.join(settings["debug_folder"], "extra_motion_registration")):
            os.makedirs(os.path.join(settings["debug_folder"], "extra_motion_registration"))

        logger.debug("Saving extra debug information for the registration...")

        # step of the vector field for the displacement transform
        step = 3

        for slice_idx in slices:
            X, Y = np.meshgrid(
                np.arange(0, registration_image_data["deformation_field"][slice_idx]["grid"][1].shape[1], step),
                np.arange(0, registration_image_data["deformation_field"][slice_idx]["grid"][1].shape[0], step),
            )

            # number of frames
            n_imgs = len(registration_image_data["img_pre_reg"][slice_idx])

            # make montage with:
            # - reference image
            # - current image with and without registration
            # - 3 image comparisons for reference and registered image
            # - displacement field and grid with displacement field applied to it
            for img_idx in range(n_imgs):
                c_ref = np.divide(ref_images[slice_idx]["image"], np.max(ref_images[slice_idx]["image"]))
                c_img_pre = np.divide(
                    registration_image_data["img_pre_reg"][slice_idx][img_idx],
                    np.max(registration_image_data["img_pre_reg"][slice_idx][img_idx]),
                )
                c_img_post = np.divide(
                    registration_image_data["img_post_reg"][slice_idx][img_idx],
                    np.max(registration_image_data["img_post_reg"][slice_idx][img_idx]),
                )
                comp_1 = compare_images(c_ref, c_img_post, method="checkerboard")
                comp_2 = compare_images(c_ref, c_img_post, method="diff")
                comp_3 = compare_images(c_ref, c_img_post, method="blend")

                plt.figure(figsize=(5, 5))

                plt.subplot(3, 3, 1)
                plt.imshow(c_ref)
                plt.axis("off")
                plt.title("reference", fontsize=7)

                plt.subplot(3, 3, 2)
                plt.imshow(c_img_post)
                plt.axis("off")
                plt.title("registered", fontsize=7)

                plt.subplot(3, 3, 3)
                plt.imshow(c_img_pre)
                plt.axis("off")
                plt.title("original", fontsize=7)

                plt.subplot(3, 3, 4)
                plt.imshow(comp_1)
                plt.axis("off")
                plt.title("checkerboard", fontsize=7)

                plt.subplot(3, 3, 5)
                plt.imshow(comp_2)
                plt.axis("off")
                plt.title("diff", fontsize=7)

                plt.subplot(3, 3, 6)
                plt.imshow(comp_3)
                plt.axis("off")
                plt.title("blend", fontsize=7)

                plt.subplot(3, 3, 7)
                plt.imshow(c_img_post)
                plt.imshow(
                    registration_image_data["deformation_field"][slice_idx]["grid"][img_idx, :, :],
                    alpha=registration_image_data["deformation_field"][slice_idx]["grid"][img_idx, :, :],
                    vmin=0,
                    vmax=2,
                    cmap="Oranges",
                )
                plt.axis("off")
                plt.title("deformation grid", fontsize=7)

                plt.subplot(3, 3, 8)
                plt.imshow(c_img_post, vmin=np.min(c_img_post), vmax=np.max(c_img_post) * 2)
                plt.quiver(
                    X,
                    Y,
                    -registration_image_data["deformation_field"][slice_idx]["field"][img_idx, ::step, ::step, 0],
                    registration_image_data["deformation_field"][slice_idx]["field"][img_idx, ::step, ::step, 1],
                    edgecolor="tab:blue",
                    facecolor="tab:orange",
                )
                plt.axis("off")
                plt.title("deformation field", fontsize=7)

                plt.tight_layout(pad=1.0)
                plt.savefig(
                    os.path.join(
                        settings["debug_folder"],
                        "extra_motion_registration",
                        "registration_slice_" + str(slice_idx).zfill(2) + "_img_" + str(img_idx).zfill(3) + ".png",
                    ),
                    dpi=200,
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()


def crop_fov(
    dti: dict,
    data: pd.DataFrame,
    mask_3c: NDArray,
    segmentation: dict,
    slices: NDArray,
    average_images: NDArray,
    registration_image_data: dict,
    ref_images: dict,
    info: dict,
    logger: logging.Logger,
    settings: dict,
) -> [dict, pd.DataFrame, NDArray, dict, NDArray, dict, NDArray]:
    """
    Crop images to the heart region as defined by the segmentation

    Parameters
    ----------
    dti: dictionary with DTI data
    data: dataframe with the diffusion images
    mask_3c: segmentation mask of the heart
    segmentation: segmentation information
    slices: array with slice positions
    average_images: average image of each slice
    registration_image_data: dict with registration info: images before and after registration,
        and also displacement field, and grid image with displacement field applied to it
    ref_images: reference images
    info: useful info dictionary
    logger: logger
    settings: dictionary with useful info

    Returns
    -------
    dti: dictionary with DTI data
    data: dataframe with the diffusion images now cropped.
    mask_3c: U-Net mask of the heart now cropped.
    segmentation: segmentation information now cropped.
    average_images: average image of each slice now cropped.
    info: dictionary with useful stuff, here updates the image size to the cropped one.
    crop_mask: logical mask with the crop.

    """
    dti, data, mask_3c, segmentation, average_images, ref_images, info, crop_mask = crop_images(
        dti,
        data,
        mask_3c,
        segmentation,
        slices,
        average_images,
        ref_images,
        info,
        logger,
        settings,
    )
    logger.info("Images cropped based on segmentation.")

    for slice_idx in slices:
        # crop the images before registration
        registration_image_data["img_pre_reg"][slice_idx] = registration_image_data["img_pre_reg"][slice_idx][
            np.ix_(
                np.repeat(True, registration_image_data["img_pre_reg"][slice_idx].shape[0]),
                crop_mask.any(1),
                crop_mask.any(0),
            )
        ]

        # crop the images after registration
        registration_image_data["img_post_reg"][slice_idx] = registration_image_data["img_post_reg"][slice_idx][
            np.ix_(
                np.repeat(True, registration_image_data["img_post_reg"][slice_idx].shape[0]),
                crop_mask.any(1),
                crop_mask.any(0),
            )
        ]

        # crop the deformation field
        registration_image_data["deformation_field"][slice_idx]["field"] = registration_image_data[
            "deformation_field"
        ][slice_idx]["field"][
            np.ix_(
                np.repeat(True, registration_image_data["deformation_field"][slice_idx]["field"].shape[0]),
                crop_mask.any(1),
                crop_mask.any(0),
            )
        ]

        # crop the deformation grid
        registration_image_data["deformation_field"][slice_idx]["grid"] = registration_image_data["deformation_field"][
            slice_idx
        ]["grid"][
            np.ix_(
                np.repeat(True, registration_image_data["deformation_field"][slice_idx]["grid"].shape[0]),
                crop_mask.any(1),
                crop_mask.any(0),
            )
        ]

    record_image_registration(registration_image_data, ref_images, mask_3c, slices, settings, logger)

    # if settings["debug"]:
    #     create_2d_montage_from_database(
    #         data,
    #         "b_value_original",
    #         "direction_original",
    #         info,
    #         settings,
    #         slices,
    #         "dwis_post_crop",
    #         settings["debug_folder"],
    #         [],
    #         segmentation,
    #     )

    return dti, data, mask_3c, segmentation, average_images, info, crop_mask
