import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def phase_correction_for_complex_averaging(data: pd.DataFrame, logger: logging.Logger, settings: dict) -> pd.DataFrame:
    """Apply phase correction to complex DWI data before averaging.

    Performs the following steps for each image:

    1. Fourier-transform the complex image to k-space.
    2. Apply a 2-D Gaussian filter to extract a low-resolution phase map.
    3. Subtract the low-resolution phase from the original phase.

    This removes low-frequency motion-induced phase errors that would
    otherwise corrupt complex averaging.

    Args:
        data (pd.DataFrame): DataFrame containing at least ``"image"``
            (magnitude) and ``"image_phase"`` columns.
        logger (logging.Logger): Logger for debug messages.
        settings (dict): Configuration dict; ``debug`` and ``debug_folder``
            control optional output figures.

    Returns:
        pd.DataFrame: DataFrame with phase-corrected magnitude images in the
        ``"image"`` column.
    """

    logger.debug("Phase correction for complex averaging.")

    filter_size = 1 / 2

    def gaussian_filter(n_lin: int, n_col: int, filter_size: float = 1 / 2) -> NDArray:
        """Create a 2-D Gaussian kernel.

        The full width at tenth of maximum (FWTM) of the kernel equals
        ``filter_size`` times the corresponding FOV dimension.

        Args:
            n_lin (int): Number of rows.
            n_col (int): Number of columns.
            filter_size (float, optional): Fraction of FOV covered by the
                FWTM. Defaults to ``0.5``.

        Returns:
            NDArray: Gaussian kernel array of shape ``(n_lin, n_col)``.
        """
        # The FWTM (full width at tenth of maximum) of the Gaussian filter is:
        # 4.29193 * sigma
        fwht_factor = 4.29193
        # The sigma of the Gaussian filter is defined so that the FWTM is equal to the filter size times the FOV size. So a filter size of 1/4 means that the FWTM of the
        # Gaussian curve is 1/4 of the FOV size.
        sigma_lin = filter_size * n_lin / fwht_factor
        sigma_col = filter_size * n_col / fwht_factor

        filter = np.zeros((n_lin, n_col))
        for i in range(n_lin):
            for j in range(n_col):
                filter[i, j] = np.exp(
                    -((((i - n_lin // 2) ** 2) / (2 * sigma_lin**2)) + (((j - n_col // 2) ** 2) / (2 * sigma_col**2)))
                )

        return filter

    # get the image size
    img = data["image"].iloc[0]

    # get the Gaussian filter
    gauss_filter = gaussian_filter(img.shape[0], img.shape[1], filter_size)

    # get the magnitude and phase arrays from the dataframe
    mag = data["image"].values
    phase = data["image_phase"].values
    mag_corrected = np.zeros_like(mag)
    phase_corrected = np.zeros_like(phase)

    # loop over each image and correct the phase
    for i in range(len(mag)):
        # create complex image and iFFT back to k-space
        complex_image = mag[i] * np.exp(1j * phase[i])
        temp = np.fft.ifftshift(complex_image)
        k_space = np.fft.ifft2(temp)
        k_space = np.fft.fftshift(k_space)

        # apply the pyramid filter to k-space
        k_space = k_space * gauss_filter
        k_space = np.fft.fftshift(k_space)
        complex_image_filtered = np.fft.fft2(k_space)
        complex_image_filtered = np.fft.fftshift(complex_image_filtered)

        # subtract the low-resolution phase from the original complex image
        complex_image_with_phase_corrected = complex_image * np.exp(-1j * np.angle(complex_image_filtered))

        if settings["debug"]:
            # plot, as an example for the first image, the process of filtering the low-resolution phase
            if i == 0:
                plt.figure(figsize=(5, 5))
                plt.subplot(332)
                plt.imshow(np.abs(complex_image))
                plt.axis("off")
                plt.title("original mag")
                plt.colorbar()
                plt.subplot(333)
                plt.imshow(np.angle(complex_image))
                plt.axis("off")
                plt.title("original phase")
                plt.colorbar()
                plt.subplot(334)
                plt.imshow(np.abs(gauss_filter))
                plt.axis("off")
                plt.title("filter")
                plt.colorbar()
                plt.subplot(335)
                plt.imshow(np.abs(complex_image_filtered))
                plt.axis("off")
                plt.title("low-res mag")
                plt.colorbar()
                plt.subplot(336)
                plt.imshow(np.angle(complex_image_filtered))
                plt.axis("off")
                plt.title("low-res phase")
                plt.colorbar()
                plt.subplot(338)
                plt.imshow(np.abs(complex_image_with_phase_corrected))
                plt.axis("off")
                plt.title("corrected mag")
                plt.colorbar()
                plt.subplot(339)
                plt.imshow(np.angle(complex_image_with_phase_corrected))
                plt.axis("off")
                plt.title("corrected phase")
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        settings["debug_folder"],
                        "phase_correction_filter_example.png",
                    ),
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()

        # update the magnitude and phase arrays
        mag_corrected[i] = np.ascontiguousarray(np.abs(complex_image_with_phase_corrected))
        phase_corrected[i] = np.ascontiguousarray(np.angle(complex_image_with_phase_corrected))

    # update the dataframe with the new complex data
    data["image"] = mag_corrected
    data["image_phase"] = phase_corrected

    return data
