import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from denoise.denoise_bm4d import bm4d_denoise
from denoise.dip_denoise import dip_denoise
from denoise.patch_denoise import locally_low_rank_tucker
from dipy.core.gradients import gradient_table
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.localpca import localpca, mppca  # noqa
from dipy.denoise.noise_estimate import estimate_sigma as estimate_sigma_dipy
from dipy.denoise.non_local_means import non_local_means
from dipy.denoise.patch2self import patch2self
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm


def denoise_all_mppca(data: pd.DataFrame, settings: dict, logger: logging.Logger, info: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using a Multi-Patch PCA (MPPCA) filter.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with MPPCA")
    slices = data["slice_integer"].unique()

    image_stack = np.zeros((len(slices), info["img_size"][0], info["img_size"][1], info["n_images"]))
    for i, slice_idx in enumerate(slices):
        # get a stack of all the DWIs
        image_stack[i] = np.stack(data[data["slice_integer"] == slice_idx]["image"].values, axis=-1)
        current_entries = data.loc[data["slice_integer"] == slice_idx]

        # remove any images that have been marked to be removed
        current_entries = current_entries.loc[current_entries["to_be_removed"] == False]

        bvals = current_entries["b_value"].values
        bvecs = np.vstack(current_entries["diffusion_direction"])
        gtab = gradient_table(bvals, bvecs, atol=0.5)  # noqa

    denosing_settings = {
        "patch_radius": 5,
        # "tau_factor": None,
        # "gtab": gtab,
    }
    if "denoising_settings" in settings:
        denosing_settings.update(settings["denoising_settings"])
    # denoise_image_stack = localpca(image_stack, **denosing_settings)
    denoise_image_stack = mppca(image_stack, **denosing_settings)

    for i, slice_idx in enumerate(slices):
        # denoise the images

        denoised_slice = denoise_image_stack[i]
        denoised_slice = np.transpose(denoised_slice, (2, 0, 1))
        # save the denoised images in the original dataframe
        df_temp = pd.DataFrame.from_records(
            denoised_slice[:, None], columns=["image"], index=data[data["slice_integer"] == slice_idx].index
        )
        data.loc[data["slice_integer"] == slice_idx, "image"] = df_temp

    return data


def denoise_all_patch2self(data: pd.DataFrame, settings: dict, logger: logging.Logger, info: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using a the patch2self algorithm.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with patch2self")
    slices = data["slice_integer"].unique()

    image_stack = np.zeros((len(slices), info["img_size"][0], info["img_size"][1], info["n_images"]))
    for i, slice_idx in enumerate(slices):
        # get a stack of all the DWIs
        image_stack[i] = np.stack(data[data["slice_integer"] == slice_idx]["image"].values, axis=-1)
        current_entries = data.loc[data["slice_integer"] == slice_idx]

        # remove any images that have been marked to be removed
        current_entries = current_entries.loc[current_entries["to_be_removed"] == False]

        bvals = current_entries["b_value"].values

    denoise_image_stack = patch2self(
        image_stack,
        bvals,
        model="ols",
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=0,
    )

    for i, slice_idx in enumerate(slices):
        # denoise the images

        denoised_slice = denoise_image_stack[i]
        denoised_slice = np.transpose(denoised_slice, (2, 0, 1))
        # save the denoised images in the original dataframe
        df_temp = pd.DataFrame.from_records(
            denoised_slice[:, None], columns=["image"], index=data[data["slice_integer"] == slice_idx].index
        )
        data.loc[data["slice_integer"] == slice_idx, "image"] = df_temp

    return data


def denoise_all_nlm_dipy(data: pd.DataFrame, logger: logging.Logger, info: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using a NLM filter implemented in dipy.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with NLM dipy")
    slices = data["slice_integer"].unique()

    image_array = []
    for slice in slices:
        # get the images for this slice
        image = np.stack(data[data["slice_integer"] == slice]["image"].values)
        image_array.append(image)

    image_array = np.stack(image_array, axis=3)
    image_array = np.transpose(image_array, (1, 2, 3, 0))
    print("image_array shape", image_array.shape)

    # in this method we need to denoise each diffusion direction separately
    denoised_arr = np.zeros(image_array.shape)
    for diff_idx in range(image_array.shape[3]):
        sigma = estimate_sigma_dipy(image_array[..., diff_idx])
        mask = image_array[..., diff_idx] > 0
        den_small = non_local_means(
            image_array[..., diff_idx],
            sigma=sigma,
            mask=mask,
            patch_radius=1,
            block_radius=1,
            rician=True,
        )
        den_large = non_local_means(
            image_array[..., diff_idx],
            sigma=sigma,
            mask=mask,
            patch_radius=2,
            block_radius=1,
            rician=True,
        )
        denoised_arr[..., diff_idx] = adaptive_soft_matching(image_array[..., diff_idx], den_small, den_large, sigma)

        denoise_image_stack = np.reshape(denoised_arr, (denoised_arr.shape[0], denoised_arr.shape[1], -1))
        denoise_image_stack = np.transpose(denoise_image_stack, (2, 0, 1))

        # save the denoised images in the original dataframe
        df_temp = pd.DataFrame.from_records(denoise_image_stack[:, None])
        df_temp.index = data.index
        data["image"] = df_temp

    return data


def denoise_all_nlm(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using a Non-Local Means (NLM) filter.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with NLM")

    # get a stack of all the DWIs
    image_stack = np.stack(data["image"].values)

    # denoise with a non-local means filter
    denoise_image_stack = np.zeros(image_stack.shape)
    # NLM parameters

    patch_kw = dict(
        patch_size=10,
        patch_distance=20,
        channel_axis=None,
        preserve_range=True,
    )

    # denoise the images
    for i in tqdm(range(len(denoise_image_stack)), desc="Denoising images", unit="image"):
        sigma_est = estimate_sigma(image_stack[i], channel_axis=None)
        denoise_image_stack[i] = denoise_nl_means(
            image_stack[i], h=6 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
        )

    # save the denoised images in the original dataframe
    df_temp = pd.DataFrame.from_records(denoise_image_stack[:, None])
    data["image"] = df_temp

    return data


def denoise_all_bm4d(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using the bm4d algorithm.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with BM3D")

    # get a stack of all the DWIs
    image_stack = np.stack(data["image"].values)

    # denoise with a non-local means filter
    denoise_image_stack = np.zeros(image_stack.shape)

    # denoise the images
    for i in tqdm(range(len(denoise_image_stack)), desc="Denoising images", unit="image"):
        sigma_est = estimate_sigma_dipy(image_stack[i : i + 1], N=1)
        denoise_image_stack[i] = bm4d_denoise(image_stack[i], sigma_est)

    # save the denoised images in the original dataframe
    df_temp = pd.DataFrame.from_records(denoise_image_stack[:, None])
    data["image"] = df_temp

    return data


def denoise_all_dip(data: pd.DataFrame, settings: dict, logger: logging.Logger, info: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using the DIP algorithm.
    The denoised images replace the original images in the input DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    logger.debug("Image denoising with DIP")
    slices = data["slice_integer"].unique()

    image_stack = np.zeros((len(slices), info["img_size"][0], info["img_size"][1], info["n_images"]))
    for i, slice_idx in enumerate(slices):
        # get a stack of all the DWIs
        image_stack[i] = np.stack(data[data["slice_integer"] == slice_idx]["image"].values, axis=-1)

    denosing_settings = {
        "iterations": 1000,
        "learning_rate": 1e-4,
    }
    if "denoising_settings" in settings:
        denosing_settings.update(settings["denoising_settings"])

    denoise_stack = np.zeros(image_stack.shape)
    for slice_idx in tqdm(slices, desc="Denoising slices", unit="slice"):
        # get the images for this slice
        image_stack_slice = image_stack[slice_idx]

        # normalize the images to [0, 1]
        min_val = np.min(image_stack_slice)
        max_val = np.max(image_stack_slice)
        image_stack_slice = (image_stack_slice - min_val) / (max_val - min_val)
        denoise_image_slice = dip_denoise(image_stack_slice, **denosing_settings)
        denoise_image_slice = np.transpose(denoise_image_slice, (1, 2, 0))

        denoise_stack[slice_idx] = denoise_image_slice * (max_val - min_val) + min_val

    for i, slice_idx in enumerate(slices):
        # denoise the images

        denoised_slice = denoise_stack[i]
        denoised_slice = np.transpose(denoised_slice, (2, 0, 1))

        # save the denoised images in the original dataframe
        df_temp = pd.DataFrame.from_records(
            denoised_slice[:, None], columns=["image"], index=data[data["slice_integer"] == slice_idx].index
        )
        data.loc[data["slice_integer"] == slice_idx, "image"] = df_temp

    return data


def denoise_all_tucker(data: pd.DataFrame, logger: logging.Logger, settings: dict, info: dict) -> pd.DataFrame:

    logger.debug("Image denoising with Tucker")
    slices = data["slice_integer"].unique()

    image_stack = np.zeros((len(slices), info["img_size"][0], info["img_size"][1], info["n_images"]))
    for i, slice_idx in enumerate(slices):
        # get a stack of all the DWIs
        image_stack[i] = np.stack(data[data["slice_integer"] == slice_idx]["image"].values, axis=-1)
        current_entries = data.loc[data["slice_integer"] == slice_idx]

        # remove any images that have been marked to be removed
        current_entries = current_entries.loc[current_entries["to_be_removed"] == False]

    # normalize the images to [0, 1]
    if settings["complex_data"]:
        # normalize the magnitude and phase separately
        image_stack_mag = np.abs(image_stack)
        image_stack_angle = np.angle(image_stack)

        min_val = np.min(image_stack_mag)
        max_val = np.max(image_stack_mag)

        image_stack_mag = (image_stack_mag - min_val) / (max_val - min_val)

        # combine the normalized magnitude and phase back into a complex image
        image_stack = image_stack_mag * np.exp(1j * image_stack_angle)
    else:
        min_val = np.min(image_stack)
        max_val = np.max(image_stack)
        image_stack = (image_stack - min_val) / (max_val - min_val)

    default_settings = {
        "threshold": 0.5,
        "tau": 0.1,
        "patch_transform": "fft",
        "lambda2d": 1,
        "window_size": 5,
        "search_window": 11,
    }
    if "denoising_settings" in settings:
        default_settings.update(settings["denoising_settings"])

    denoised_slice = locally_low_rank_tucker(image_stack, **default_settings)
    # denoise_image_stack = image_stack.copy()
    # save the denoised images in the original dataframe

    for i, slice_idx in enumerate(slices):

        if settings["complex_data"]:
            # save the denoised images in the original dataframe
            denoise_image_stack_mag = np.abs(denoised_slice[i])
            denoise_image_stack_angle = np.angle(denoised_slice[i])

            denoise_image_stack_mag = denoise_image_stack_mag * (max_val - min_val) + min_val

            df_temp = pd.DataFrame.from_records(
                denoise_image_stack_mag[:, None],
                columns=["image"],
                index=data[data["slice_integer"] == slice_idx].index,
            )
            data.loc[data["slice_integer"] == slice_idx, "image"] = df_temp

            df_temp = pd.DataFrame.from_records(
                denoise_image_stack_angle[:, None],
                columns=["image_phase"],
                index=data[data["slice_integer"] == slice_idx].index,
            )
            data.loc[data["slice_integer"] == slice_idx, "image_phase"] = df_temp

        else:
            denoised_slice_stack = denoised_slice[i] * (max_val - min_val) + min_val
            denoised_slice_stack = np.transpose(denoised_slice_stack, (2, 0, 1))

            df_temp = pd.DataFrame.from_records(
                denoised_slice_stack[:, None], columns=["image"], index=data[data["slice_integer"] == slice_idx].index
            )
            data.loc[data["slice_integer"] == slice_idx, "image"] = df_temp

    return data


def image_denoising(data: pd.DataFrame, logger: logging.Logger, settings: dict, info: dict) -> pd.DataFrame:
    """Denoise all DWIs in the DataFrame using a Non-Local Means (NLM) filter.

    The denoised images replace the original images in the input DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing DWI images and related metadata.
        logger (logging.Logger): Logger instance for debug output.
        settings (dict): Dictionary of configuration settings. Must include 'debug'
            (bool) and optionally 'debug_folder' (str) if debug is True.

    Returns:
        data (pd.DataFrame): DataFrame with denoised images replacing original images.
    """

    image_stack = np.stack(data["image"].values)

    if settings["denoise_method"] == "nlm":
        data = denoise_all_nlm(data, logger)
    elif settings["denoise_method"] == "mppca":
        data = denoise_all_mppca(data, settings, logger, info)
    elif settings["denoise_method"] == "tucker":
        data = denoise_all_tucker(data, logger, settings, info)
    elif settings["denoise_method"] == "bm4d":
        data = denoise_all_bm4d(data, logger)
    elif settings["denoise_method"] == "nlm_dipy":
        data = denoise_all_nlm_dipy(data, logger, info)
    elif settings["denoise_method"] == "patch2self":
        data = denoise_all_patch2self(data, settings, logger, info)
    elif settings["denoise_method"] == "dip":
        data = denoise_all_dip(data, settings, logger, info)
    else:
        logger.error(f"Unknown denoise method: {settings['denoise_method']}")
        raise ValueError(f"Unknown denoise method: {settings['denoise_method']}")

    if settings["debug"]:

        denoise_image_stack = np.stack(data["image"].values)
        n_images = len(denoise_image_stack)

        # display some examples of the original and denoised images
        # get 5 random numbers between 0 and n_images
        random_numbers = np.random.randint(0, n_images, 5)
        # sort the random numbers
        random_numbers.sort()

        # display the images
        fig, axes = plt.subplots(3, 5, figsize=(15, 5))
        for i in range(5):
            axes[0, i].imshow(image_stack[random_numbers[i]], cmap="gray")
            axes[0, i].set_title(f"Original {random_numbers[i]}")
            axes[0, i].axis("off")
            axes[1, i].imshow(denoise_image_stack[random_numbers[i]], cmap="gray")
            axes[1, i].set_title(f"Denoised {random_numbers[i]}")
            axes[1, i].axis("off")
            axes[2, i].imshow(
                np.abs(denoise_image_stack[random_numbers[i]] - image_stack[random_numbers[i]]), cmap="gray"
            )
            axes[2, i].set_title(f"Difference {random_numbers[i]}")
            axes[2, i].axis("off")
        plt.savefig(
            os.path.join(
                settings["debug_folder"],
                "image_denoising_examples.png",
            ),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()

    return data
