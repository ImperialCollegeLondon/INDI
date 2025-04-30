import numpy as np
import os
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import pandas as pd
import logging


def image_denoising(data: pd.DataFrame, logger: logging.Logger, settings: dict) -> pd.DataFrame:
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
    for i in range(len(denoise_image_stack)):
        sigma_est = estimate_sigma(image_stack[i], channel_axis=None)
        denoise_image_stack[i] = denoise_nl_means(
            image_stack[i], h=4 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
        )

    n_images = len(denoise_image_stack)
    # save the denoised images in the original dataframe
    df_temp = pd.DataFrame.from_records(denoise_image_stack[:, None])
    data["image"] = df_temp

    if settings["debug"]:
        # display some examples of the original and denoised images
        # get 5 random numbers between 0 and n_images
        random_numbers = np.random.randint(0, n_images, 5)
        # sort the random numbers
        random_numbers.sort()

        # display the images
        fig, axes = plt.subplots(2, 5, figsize=(15, 5))
        for i, ax in enumerate(axes.flat):
            if i < 5:
                ax.imshow(image_stack[random_numbers[i]], cmap="gray")
                ax.set_title(f"Original {random_numbers[i]}")
            else:
                ax.imshow(denoise_image_stack[random_numbers[i - 5]], cmap="gray")
                ax.set_title(f"Denoised {random_numbers[i - 5]}")
            ax.axis("off")
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
