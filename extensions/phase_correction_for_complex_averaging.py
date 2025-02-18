import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def phase_correction_for_complex_averaging(data: pd.DataFrame, logger: logging.Logger, settings: dict) -> pd.DataFrame:
    """
    Performs phase correction for complex averaging:
    1. Fourier transforms the complex data
    2. Apply a pyramid filter to the k-space data creating a low resolution image
    3. Subtract the low resolution phase of the original phase

    This should remove motion induced phase errors in the complex data. We expect the motion induced phase errors to be low frequency.

    Parameters
    ----------
    data: dataframe with data
    logger: logger
    settings: settings dictionary

    Returns
    -------
    dataframe data phase corrected

    """

    logger.debug("Phase correction for complex averaging.")

    # pyramid filter size x: the pyramid is the size of 1/x of the original image
    filter_factor = 4

    def pyramid(n_lin, n_col, factor=1):
        """
        Create a pyramid filter for k-space data.

        Parameters
        ----------
        n_lin: int
            Number of lines in the image.
        n_col: int
            Number of columns in the image.
        factor: int
            Pyramid filter size.

        Returns
        -------
        pyr: np.array
            Pyramid filter.

        """

        # number of line and columns in the pyramid
        n_lin_pyr = n_lin // factor
        n_col_pyr = n_col // factor

        # create the pyramid filter
        r_lin = np.arange(n_lin_pyr)
        r_col = np.arange(n_col_pyr)
        d_lin = np.minimum(r_lin, r_lin[::-1])
        d_col = np.minimum(r_col, r_col[::-1])
        pyr = np.minimum.outer(d_lin, d_col)
        # normalise the pyramid filter
        pyr = pyr / np.max(pyr)

        # The pyramid filter can be larger or bigger than the original image
        # so we need to pad or crop the pyramid filter to the original image size
        if factor > 1:
            # pad the pyramid to the original size
            pyr = np.pad(
                pyr,
                [(round((n_lin - (n_lin // factor)) // 2),), (round((n_col - (n_col // factor)) // 2),)],
                mode="constant",
            )

        else:
            delta_lin = n_lin_pyr - n_lin
            delta_col = n_col_pyr - n_col

            pyr = pyr[
                round(delta_lin // 2) : round(delta_lin // 2 + n_lin),
                round(delta_col // 2) : round(delta_col // 2 + n_col),
            ]

        # make sure pyramid filter is the same size as the input image
        if not (n_lin, n_col) == pyr.shape:
            if n_lin < pyr.shape[0]:
                pyr = pyr[:n_lin, :]
            elif n_lin > pyr.shape[0]:
                delta = n_lin - pyr.shape[0]
                pyr = np.pad(pyr, ((delta, 0), (0, 0)), mode="constant")
            if n_col < pyr.shape[1]:
                pyr = pyr[:, :n_col]
            elif n_col > pyr.shape[1]:
                delta = n_col - pyr.shape[1]
                pyr = np.pad(pyr, ((0, 0), (delta, 0)), mode="constant")

        return pyr

    # get the image size
    img = data["image"].iloc[0]

    # get the pyramid filter
    pyr_filter = pyramid(img.shape[0], img.shape[1], filter_factor)

    # get the magnitude and phase arrays from the dataframe
    mag = data["image"].values
    phase = data["image_phase"].values

    # loop over each image and correct the phase
    for i in range(len(mag)):
        # create complex image and iFFT back to k-space
        complex_image = mag[i] * np.exp(1j * phase[i])
        temp = np.fft.ifftshift(complex_image)
        k_space = np.fft.ifft2(temp)
        k_space = np.fft.fftshift(k_space)

        # apply the pyramid filter to k-space
        k_space = k_space * pyr_filter
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
                plt.imshow(np.abs(pyr_filter))
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
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()

        # update the magnitude and phase arrays
        mag[i] = np.ascontiguousarray(np.abs(complex_image_with_phase_corrected))
        phase[i] = np.ascontiguousarray(np.angle(complex_image_with_phase_corrected))

    # update the dataframe with the new complex data
    data["image"] = mag
    data["image_phase"] = phase

    return data
