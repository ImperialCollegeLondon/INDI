import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.localpca import mppca
from dipy.denoise.noise_estimate import estimate_sigma as estimate_sigma_dipy
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

from indi.extensions.cnlm_denoise_faster import collaborative_nlm


def image_denoising(data: pd.DataFrame, logger: logging.Logger, settings: dict, mask: np.ndarray) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using either:
        - a Non-Local Means (NLM) filter.
        - a custom collaborative NLM (CNLM) filter.
        - a custom adaptive soft coefficient matching (ASCM) filter.

    The denoised images replace the original images in the input DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
        settings (dict): Dictionary of configuration settings. Must include 'debug'
            (bool) and optionally 'debug_folder' (str) if debug is True.
        mask (np.ndarray): Binary mask indicating the region of interest for denoising.

    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    # check if denoised images exist, if so load them
    # otherwise, perform denoising
    denoised_images_path = os.path.join(settings["session"], "denoised_images.npz")

    if os.path.exists(denoised_images_path):

        logger.debug("Denoised images found, loading....")

        image_set = np.load(denoised_images_path)
        denoised_images = image_set["image_stack_after"]
        original_images = image_set["image_stack_before"]
        df_temp = pd.DataFrame.from_records(denoised_images[:, None])
        data["image"] = df_temp
        df_temp = pd.DataFrame.from_records(original_images[:, None])
        data_temp_old = pd.DataFrame.from_records(original_images[:, None])
        data_temp_old["image"] = df_temp

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

            # in this approach we will denoise slice by slice, and have a third dimension for the different repeats of the same configuration
            if settings["image_denoising"] == "ascm":
                logger.debug("Image denoising with adaptive soft coefficient matching (ASCM)...")
            if settings["image_denoising"] == "cnlm":
                logger.debug("Image denoising with collaborative NLM...")

            data_temp_old = data.copy()

            # Split the direction column into three columns X, Y, Z
            data[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(data["diffusion_direction"].tolist(), index=data.index)

            slices = data["slice_integer"].unique()

            # loop over each slice and denoise separately
            for slice_idx in slices:
                slice_data = data[data["slice_integer"] == slice_idx]

                # Split the direction column into three columns X, Y, Z
                slice_data[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(
                    slice_data["diffusion_direction"].tolist(), index=slice_data.index
                )

                # how many different configurations of diffusion directions and bvalues do we have
                unique_configs = slice_data[["b_value_original", "dir_x", "dir_y", "dir_z"]].drop_duplicates()

                # for each configuration, gather all images
                for idx, config in tqdm(
                    unique_configs.iterrows(), total=unique_configs.shape[0], desc="Denoising diffusion configurations"
                ):
                    b_value = config["b_value_original"]
                    dir_x = config["dir_x"]
                    dir_y = config["dir_y"]
                    dir_z = config["dir_z"]

                    # gather all images for this configuration in a 3d numpy array
                    config_images = slice_data[
                        (slice_data["b_value_original"] == b_value)
                        & (slice_data["dir_x"] == dir_x)
                        & (slice_data["dir_y"] == dir_y)
                        & (slice_data["dir_z"] == dir_z)
                    ]["image"].values
                    config_images = np.stack(config_images, axis=0)

                    sigma = estimate_sigma_dipy(config_images, N=0)
                    sigma = np.median(sigma)
                    logger.debug("Estimated sigma: %f", sigma)

                    # sigma_piesno = piesno(np.transpose(config_images, (1, 2, 0)), N=1)
                    # logger.debug("Estimated sigma (piesno): %f", sigma_piesno)

                    if settings["image_denoising"] == "cnlm":
                        denoised_images = collaborative_nlm(
                            config_images,
                            h=10,
                            patch_radius=1,
                            search_radius=[2, 2],
                            rician=True,
                            sigma=sigma,
                        )
                    elif settings["image_denoising"] == "ascm":
                        denoised_images_low = collaborative_nlm(
                            config_images, h=10, patch_radius=1, search_radius=10, rician=True, sigma=sigma
                        )

                        denoised_images_high = collaborative_nlm(
                            config_images, h=10, patch_radius=2, search_radius=10, rician=True, sigma=sigma
                        )

                        denoised_images = adaptive_soft_matching(
                            config_images, denoised_images_low, denoised_images_high, sigma
                        )

                    # save the denoised images in the original dataframe for this slice
                    df_temp = pd.DataFrame.from_records(denoised_images[:, None])
                    matching_indices = data[
                        (data["slice_integer"] == slice_idx)
                        & (data["b_value_original"] == b_value)
                        & (data["dir_x"] == dir_x)
                        & (data["dir_y"] == dir_y)
                        & (data["dir_z"] == dir_z)
                    ].index
                    df_temp.index = matching_indices
                    data.loc[df_temp.index, "image"] = df_temp.values

            # save denoised images in a compressed numpy array
            image_stack_after = np.stack(data["image"].values)
            image_stack_before = np.stack(data_temp_old["image"].values)
            np.savez_compressed(
                denoised_images_path, image_stack_after=image_stack_after, image_stack_before=image_stack_before
            )

        elif settings["image_denoising"] == "mppca":

            logger.debug("Image denoising with Marcenko-Pastur PCA...")

            # new dataframe for the average data
            data_averaged = pd.DataFrame()

            # Split the direction column into three columns X, Y, Z
            data[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(data["diffusion_direction"].tolist(), index=data.index)

            # how many different configurations of slices, b-vals and directions do we have?
            unique_configs = data[["b_value_original", "dir_x", "dir_y", "dir_z", "slice_integer"]].drop_duplicates()

            # for each unique configuration of b-value, direction, and slice, we will first average the images
            for idx, config in unique_configs.iterrows():
                b_value = config["b_value_original"]
                dir_x = config["dir_x"]
                dir_y = config["dir_y"]
                dir_z = config["dir_z"]
                slice_integer = config["slice_integer"]

                # gather all images for this configuration in a 3d numpy array
                config_images = data[
                    (data["b_value_original"] == b_value)
                    & (data["dir_x"] == dir_x)
                    & (data["dir_y"] == dir_y)
                    & (data["dir_z"] == dir_z)
                    & (data["slice_integer"] == slice_integer)
                ]["image"].values
                config_images = np.stack(config_images, axis=0)

                # average the images for this configuration
                config_images_mean = np.mean(config_images, axis=0)

                # store the averaged image with all the other info from the first entry in
                # the data_averaged dataframe
                df_temp = (
                    data.loc[
                        (data["b_value_original"] == b_value)
                        & (data["dir_x"] == dir_x)
                        & (data["dir_y"] == dir_y)
                        & (data["dir_z"] == dir_z)
                        & (data["slice_integer"] == slice_integer)
                    ]
                    .iloc[0]
                    .copy()
                )
                df_temp["image"] = config_images_mean
                data_averaged = pd.concat([data_averaged, pd.DataFrame([df_temp])], ignore_index=True)

            # how many different configurations of b-vals and directions do we have?
            unique_configs_2 = (
                data_averaged[["b_value_original", "dir_x", "dir_y", "dir_z"]].drop_duplicates().reset_index(drop=True)
            )

            # how many slices do we have?
            unique_slices = data_averaged["slice_integer"].unique()

            # check if we have the same configs for each slice
            for slice in unique_slices:
                data_slice = data_averaged[data_averaged["slice_integer"] == slice]
                unique_configs_slice = (
                    data_slice[["b_value_original", "dir_x", "dir_y", "dir_z"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                if not unique_configs_slice.equals(unique_configs_2):
                    raise ValueError(f"Mismatch in configurations for slice {slice}")

            # now get the images in a X,Y,Z,N array for denoising
            array_4d = np.zeros(
                (config_images.shape[1], config_images.shape[2], len(unique_slices), unique_configs_2.shape[0])
            )

            for idx, config in unique_configs_2.iterrows():
                b_value = config["b_value_original"]
                dir_x = config["dir_x"]
                dir_y = config["dir_y"]
                dir_z = config["dir_z"]

                for slice_idx, slice in enumerate(unique_slices):
                    data_slice = data_averaged[data_averaged["slice_integer"] == slice]
                    config_images = data_slice[
                        (data_slice["b_value_original"] == b_value)
                        & (data_slice["dir_x"] == dir_x)
                        & (data_slice["dir_y"] == dir_y)
                        & (data_slice["dir_z"] == dir_z)
                    ]["image"].values
                    config_images = np.stack(config_images, axis=0)
                    array_4d[:, :, slice_idx, idx] = config_images.mean(axis=0)

            if settings["image_denoising"] == "cnlm2":
                sigma = estimate_sigma_dipy(array_4d, N=0)
                sigma = np.median(sigma)
                logger.debug("Estimated sigma: %f", sigma)
                array_4d_denoised = collaborative_nlm(
                    array_4d,
                    h=10,
                    patch_radius=3,
                    search_radius=2,
                    rician=True,
                    sigma=sigma,
                )
            elif settings["image_denoising"] == "mppca":
                array_4d_denoised, sigma = mppca(array_4d, patch_radius=[4, 4, 1], return_sigma=True, pca_method="svd")

            plt.figure()
            img_1 = array_4d[:, :, 0, 2]
            img_1_denoised = array_4d_denoised[:, :, 0, 2]
            plt.subplot(1, 3, 1)
            plt.imshow(img_1, cmap="gray")
            plt.title("Original")
            plt.subplot(1, 3, 2)
            plt.imshow(img_1_denoised, cmap="gray")
            plt.title("Denoised")
            plt.subplot(1, 3, 3)
            plt.imshow(abs(img_1 - img_1_denoised), cmap="gray")
            plt.title("Difference")
            plt.show()

            # store data before denoising
            data_temp_old = data_averaged.copy()

            # save the images back to the data_averaged dataframe
            for idx, config in unique_configs_2.iterrows():
                b_value = config["b_value_original"]
                dir_x = config["dir_x"]
                dir_y = config["dir_y"]
                dir_z = config["dir_z"]

                for slice_idx, slice in enumerate(unique_slices):
                    # Find the index that matches all conditions
                    mask = (
                        (data_averaged["b_value_original"] == b_value)
                        & (data_averaged["dir_x"] == dir_x)
                        & (data_averaged["dir_y"] == dir_y)
                        & (data_averaged["dir_z"] == dir_z)
                        & (data_averaged["slice_integer"] == slice)
                    )
                    matching_indices = data_averaged[mask].index

                    # Assign to the single matching row
                    if len(matching_indices) > 0:
                        data_averaged.at[matching_indices[0], "image"] = array_4d_denoised[:, :, slice_idx, idx]

            # update the original data with the denoised images from data_averaged
            data = data_averaged.copy()

        elif settings["image_denoising"] == "mppca2":
            logger.debug("Image denoising with Marcenko-Pastur PCA (method 2 no averaging)...")

            data_temp_old = data.copy()

            # how many slices do we have?
            unique_slices = data["slice_integer"].unique()

            # loop and denoise each slice separately
            for slice in unique_slices:
                data_slice = data[data["slice_integer"] == slice]

                slice_images = np.stack(data_slice["image"].values, axis=-1)

                # add an empty dimension in the third axis to match the expected 4D array shape
                slice_images = slice_images[:, :, np.newaxis, :]

                # denoise images
                slice_images_denoised, sigma = mppca(
                    slice_images, patch_radius=[4, 4, 1], return_sigma=True, pca_method="svd"
                )

                # Find the index that matches all conditions
                mask = data["slice_integer"] == slice
                matching_indices = data[mask].index

                # Assign the denoised images back to the original data
                for i, idx in enumerate(matching_indices):
                    data.at[idx, "image"] = slice_images_denoised[:, :, 0, i]

            plt.figure()
            img_1 = data_temp_old["image"].values[2]
            img_1_denoised = data["image"].values[2]
            plt.subplot(1, 3, 1)
            plt.imshow(img_1, cmap="gray")
            plt.title("Original")
            plt.subplot(1, 3, 2)
            plt.imshow(img_1_denoised, cmap="gray")
            plt.title("Denoised")
            plt.subplot(1, 3, 3)
            plt.imshow(abs(img_1 - img_1_denoised), cmap="gray")
            plt.title("Difference")
            plt.show()

    if settings["debug"]:

        # make a plot for each slice
        # unique slices
        slices = data.slice_integer.unique()

        data_temp_old["slice_integer"] = data["slice_integer"]

        for slice_idx in slices:
            images_before_slice = data_temp_old[data_temp_old["slice_integer"] == slice_idx]["image"].values
            images_after_slice = data[data["slice_integer"] == slice_idx]["image"].values
            images_before_slice = np.stack(images_before_slice, axis=0)
            images_after_slice = np.stack(images_after_slice, axis=0)
            residuals_slice = abs(images_after_slice - images_before_slice)

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
