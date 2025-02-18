import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from extensions.dwis_classifier import dwis_classifier
from extensions.extensions import crop_pad_rotate_array


def remove_outliers_ai(
    data: pd.DataFrame,
    info: dict,
    settings: dict,
    slices: NDArray,
    logger: logging.Logger,
    threshold: float = 0.3,
) -> tuple[pd.DataFrame, NDArray]:
    """Remove the bad frames from the dataframe using the AI classifier

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    info : dict
        useful info
    settings : dict
        settings
    slices : NDArray
        array with slice positions as strings
    logger : logging.Logger
        logger
    threshold : float, optional
        threshold value to consider bad in [0, 1], by default 0.3

    Returns
    -------
    Tuple[pd.DataFrame, NDArray]
        [dataframe without bad frames, array with bad frames positions]
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
