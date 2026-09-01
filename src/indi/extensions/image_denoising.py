import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.noise_estimate import estimate_sigma as estimate_sigma_dipy
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

from indi.extensions.cnlm_denoise_faster import collaborative_nlm


def image_denoising(data: pd.DataFrame, logger: logging.Logger, settings: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using either:
        - a Non-Local Means (NLM) filter.
        - a custom collaborative NLM (CNLN) filter.
        - a custom adaptive soft coefficient matching (ASCM) filter.

    The denoised images replace the original images in the input DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
        settings (dict): Dictionary of configuration settings. Must include 'debug'
            (bool) and optionally 'debug_folder' (str) if debug is True.

    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    # check if denoised images exist, if so load them
    # otherwise, perform denoising
    denoised_images_path = os.path.join(settings["session"], "denoised_images.npz")

    if os.path.exists(denoised_images_path):

        logger.debug("Denoised images found, loading....")

        denoised_images = np.load(denoised_images_path)["image_stack_after"]
        original_images = np.load(denoised_images_path)["image_stack_before"]
        df_temp = pd.DataFrame.from_records(denoised_images[:, None])
        data["image"] = df_temp
        df_temp = pd.DataFrame.from_records(original_images[:, None])
        data_temp_old = pd.DataFrame.from_records(original_images[:, None])
        data_temp_old["image"] = df_temp

        return data

    else:

        if settings["image_denoising"] == "nlm":

            data_temp_old = data.copy()

            logger.debug("Image denoising with NLM...")

            # get a stack of all the DWIs
            image_before = np.stack(data["image"].values)

            # denoise with a non-local means filter
            images_after = np.zeros(image_before.shape)

            # NLM parameters
            patch_kw = dict(
                patch_size=10,
                patch_distance=20,
                channel_axis=None,
                preserve_range=True,
            )

            # denoise the images
            for i in tqdm(range(len(images_after)), desc="Denoising images with NLM"):
                sigma_est = estimate_sigma(image_before[i], channel_axis=None)
                images_after[i] = denoise_nl_means(
                    image_before[i], h=4 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
                )

            # save the denoised images in the original dataframe
            df_temp = pd.DataFrame.from_records(images_after[:, None])
            data["image"] = df_temp

            # save denoised images in a compressed numpy array
            image_stack_after = np.stack(data["image"].values)
            image_stack_before = np.stack(data_temp_old["image"].values)
            np.savez_compressed(
                denoised_images_path, image_stack_after=image_stack_after, image_stack_before=image_stack_before
            )

        elif settings["image_denoising"] == "cnlm" or settings["image_denoising"] == "ascm":

            if settings["image_denoising"] == "ascm":
                logger.debug("Image denoising with adaptive soft coefficient matching (ASCM)...")
            if settings["image_denoising"] == "cnlm":
                logger.debug("Image denoising with collaborative NLM...")

            data_temp_old = data.copy()

            # Split the direction column into three columns X, Y, Z
            data[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(data["diffusion_direction"].tolist(), index=data.index)

            # how many different configurations of diffusion directions and bvalues do we have
            unique_configs = data[["b_value", "dir_x", "dir_y", "dir_z"]].drop_duplicates()

            # for each configuration, gather all images
            for idx, config in tqdm(
                unique_configs.iterrows(), total=unique_configs.shape[0], desc="Denoising diffusion configurations"
            ):
                b_value = config["b_value"]
                dir_x = config["dir_x"]
                dir_y = config["dir_y"]
                dir_z = config["dir_z"]

                # gather all images for this configuration in a 3d numpy array
                config_images = data[
                    (data["b_value"] == b_value)
                    & (data["dir_x"] == dir_x)
                    & (data["dir_y"] == dir_y)
                    & (data["dir_z"] == dir_z)
                ]["image"].values
                config_images = np.stack(config_images, axis=0)

                sigma = estimate_sigma_dipy(config_images, N=32)
                if settings["image_denoising"] == "cnlm":
                    denoised_images = collaborative_nlm(
                        config_images, h=sigma, patch_radius=2, search_radius=10, rician=True
                    )
                elif settings["image_denoising"] == "ascm":
                    denoised_images_low = collaborative_nlm(
                        config_images, h=sigma, patch_radius=1, search_radius=10, rician=True
                    )

                    denoised_images_high = collaborative_nlm(
                        config_images, h=sigma, patch_radius=2, search_radius=10, rician=True
                    )

                    denoised_images = adaptive_soft_matching(
                        config_images, denoised_images_low, denoised_images_high, sigma
                    )

                # save the denoised images in the original dataframe
                df_temp = pd.DataFrame.from_records(denoised_images[:, None])
                df_temp.index = data[
                    (data["b_value"] == b_value)
                    & (data["dir_x"] == dir_x)
                    & (data["dir_y"] == dir_y)
                    & (data["dir_z"] == dir_z)
                ].index
                data.loc[df_temp.index, "image"] = df_temp.values

            # save denoised images in a compressed numpy array
            image_stack_after = np.stack(data["image"].values)
            image_stack_before = np.stack(data_temp_old["image"].values)
            np.savez_compressed(
                denoised_images_path, image_stack_after=image_stack_after, image_stack_before=image_stack_before
            )

    if settings["debug"]:

        # make a plot for each slice
        # unique slices
        slices = data.slice_integer.unique()

        for slice_idx in slices:
            images_before_slice = data_temp_old[data_temp_old["slice_integer"] == slice_idx]["image"].values
            images_after_slice = data[data["slice_integer"] == slice_idx]["image"].values
            images_before_slice = np.stack(images_before_slice, axis=0)
            images_after_slice = np.stack(images_after_slice, axis=0)
            residuals_slice = images_after_slice - images_before_slice

            # display 10 random images before and after denoising and also the residuals
            random_indices = np.random.choice(
                images_before_slice.shape[0], size=min(10, images_before_slice.shape[0]), replace=False
            )
            plt.figure(figsize=(20, 5))
            for i, idx in enumerate(random_indices):
                plt.subplot(3, 10, i + 1)
                plt.imshow(images_before_slice[idx, :, :], cmap="gray")
                plt.axis("off")
                plt.subplot(3, 10, i + 1 + 10)
                plt.imshow(images_after_slice[idx, :, :], cmap="gray")
                plt.axis("off")
                plt.subplot(3, 10, i + 1 + 20)
                plt.imshow(residuals_slice[idx, :, :], cmap="gray")
                plt.axis("off")
            plt.suptitle("Random images before and after denoising")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    settings["debug_folder"],
                    "image_denoising_examples_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        return data
