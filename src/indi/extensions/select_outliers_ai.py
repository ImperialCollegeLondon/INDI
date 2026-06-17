import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from indi.extensions.dwis_classifier import dwis_classifier
from indi.extensions.extensions import crop_pad_rotate_array


def remove_outliers_ai(
    data: pd.DataFrame,
    info: dict,
    settings: dict,
    slices: NDArray,
    logger: logging.Logger,
    threshold: float = 0.3,
) -> tuple[pd.DataFrame, NDArray]:
    """Remove bad DWI frames from the DataFrame using the AI classifier.

    Args:
        data (pd.DataFrame): DataFrame containing diffusion imaging data.
        info (dict): Metadata dict with ``"n_images"`` and ``"img_size"``
            keys.
        settings (dict): Pipeline configuration (unused here, kept for API
            consistency).
        slices (NDArray): Array of slice position strings.
        logger (logging.Logger): Logger for debug messages.
        threshold (float, optional): Probability threshold in ``[0, 1]``
            below which a frame is considered bad. Defaults to ``0.3``.

    Returns:
        data_new: DataFrame with bad frames removed.
        rows_to_drop: 1-D integer array of the dropped row indices.
    """

    # gather images from dataframe
    dwis = np.zeros([info["n_images"], info["img_size"][0], info["img_size"][1]])
    for i in range(info["n_images"]):
        # moving image
        dwis[i] = data.loc[i, "image"]

    # make sure image stack has the correct dimensions
    dwis = crop_pad_rotate_array(dwis, (info["n_images"], 256, 96), True)

    # use the AI classifier to determine which ones are bad
    frame_labels = dwis_classifier(dwis, threshold)

    # drop frames frames labeled as bad (0)
    rows_to_drop = np.where(frame_labels < 1)[0]
    data_new = data.drop(index=list(rows_to_drop))

    logger.debug("Number of images removed by AI: " + str(len(rows_to_drop)))

    return data_new, rows_to_drop
