import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from scipy import stats


def plot_residuals_plot(residuals: NDArray, slice_idx: int, settings: dict, prefix: str = ""):
    """
    Plot the tensor residuals averaged per image

    Parameters
    ----------
    residuals: residuals per image
    slice_idx: slice position string
    settings: dictionary with useful info
    prefix: prefix for the file name

    """
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(residuals)
    plt.xlabel("image #")
    plt.ylabel("residuals")
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.title("Tensor residuals (average of all myocardial voxels)", fontsize=7)
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "tensor_residuals_" + prefix + "_slice_" + str(slice_idx).zfill(2) + ".png",
        ),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def plot_residuals_map(residuals: NDArray, slice_idx: int, settings: dict, prefix: str = ""):
    """
    Plot the tensor residuals averaged per pixel

    Parameters
    ----------
    residuals: residuals per pixel
    slice: slice position string
    settings: dictionary with useful info
    prefix: prefix for the file name

    """
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(residuals)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)
    plt.title("Tensor residuals (average of all DWIs)", fontsize=7)
    plt.axis("off")
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "tensor_residuals_map_" + prefix + "_slice_" + str(slice_idx).zfill(2) + ".png",
        ),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def get_residual_z_scores(residuals: NDArray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Get z-scores and outliers from residuals

    Parameters
    ----------
    residuals: array with residual values

    Returns
    -------
    z_scores: z-scores of the residuals for each image
    outliers: outliers in descending order if any based on a z-score threshold
    outliers_pos: position of any outliers

    """
    # find outliers from the average residual of each image
    # get z-score and threshold
    z_scores = stats.zscore(residuals)
    # Select data points with z-scores above or below 3
    outliers = np.abs(z_scores) > 3.0
    # store outliers by descending z-score
    idxs = np.nonzero(outliers)[0]
    ascending_order = z_scores[idxs].argsort()
    outliers_pos = np.flip(idxs[ascending_order])

    return z_scores, outliers, outliers_pos


def plot_tensor_components(D: NDArray, slices: NDArray, settings: dict):
    """
    Plot tensor components

    Parameters
    ----------
    D: diffusion tensor array
    slices: array with slice positions
    settings: dictionary with useful info

    Returns
    -------

    """
    for slice_idx in slices:
        # imshow the tensor
        plt.figure(figsize=(15, 15))
        plt.subplot(3, 3, 1)
        plt.imshow(D[slice_idx, :, :, 0, 0], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxx")

        plt.subplot(3, 3, 5)
        plt.imshow(D[slice_idx, :, :, 1, 1], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dyy")

        plt.subplot(3, 3, 9)
        plt.imshow(D[slice_idx, :, :, 2, 2], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Dzz")
        plt.axis("off")

        plt.subplot(3, 3, 2)
        plt.imshow(D[slice_idx, :, :, 0, 1], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxy")

        plt.subplot(3, 3, 3)
        plt.imshow(D[slice_idx, :, :, 0, 2], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxz")

        plt.subplot(3, 3, 6)
        plt.imshow(D[slice_idx, :, :, 1, 2], vmin=D.min(), vmax=D.max())
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Dyz")
        plt.axis("off")
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(
                settings["debug_folder"],
                "tensor_components_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=200,
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def dipy_tensor_fit(
    slices: NDArray,
    data: pd.DataFrame,
    info: dict,
    settings: dict,
    mask_3c: NDArray,
    logger: logging.Logger,
    method: str = "NLLS",
    quick_mode=False,
):
    """

    Fit tensor to data in dataframe. The fitting methods are
    from DiPy:
    - LS: Linear Least Squares
    - WLS: Weighted Linear Least Squares
    - NLLS: Non-Linear Least Squares
    - RESTORE: RESTORE method

    Parameters
    ----------
    slices: array of strings with slice positions
    data: dataframe with all the diffusion information
    info: dictionary with general information
    settings: dictionary with general options
    mask_3c: segmentation mask
    logger: logger messages
    method: string with the fitting method
    quick_mode: boolean to speed up the function

    Returns
    -------

    Tensor array and info dictionary

    """
    import dipy.denoise.noise_estimate as ne
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table

    logger.info("Starting tensor fitting with method: " + method)

    tensor = np.empty([info["img_size"][0], info["img_size"][1], 3, 3, len(slices)])
    residuals_img = {}
    residuals_map = {}

    myo_mask = np.copy(mask_3c.reshape(mask_3c.shape[0], mask_3c.shape[1] * mask_3c.shape[2]))
    myo_mask[myo_mask > 1] = 0

    # I need to do this per slice, because gtab might differ from slice to slice
    info["tensor fitting sigma"] = {}
    for slice_idx in slices:
        current_entries = data.loc[data["slice_integer"] == slice_idx]
        bvals = current_entries["b_value"].values
        bvecs = np.vstack(current_entries["direction"])
        gtab = gradient_table(bvals, bvecs, atol=0.5)

        image_data = np.stack(current_entries["image"])
        image_data = image_data[..., np.newaxis]
        image_data = image_data.transpose(1, 2, 3, 0)
        if not quick_mode:
            sigma = ne.estimate_sigma(image_data)
            info["tensor fitting sigma"][str(slice_idx).zfill(2)] = (
                "%.2f" % np.nanmean(sigma) + " +/- " + "%.2f" % np.nanstd(sigma)
            )
            logger.debug(
                "Mean sigma for slice "
                + str(slice_idx).zfill(2)
                + ": "
                + str("%.2f" % np.nanmean(sigma) + " +/- " + "%.2f" % np.nanstd(sigma))
            )

        if method == "NLLS" or method == "RESTORE":
            tenmodel = dti.TensorModel(gtab, fit_method=method, sigma=sigma, return_S0_hat=True)
        else:
            tenmodel = dti.TensorModel(gtab, fit_method=method, return_S0_hat=True)

        tenfit = tenmodel.fit(image_data)
        tensor[..., slice_idx] = np.squeeze(tenfit.quadratic_form)

        if not quick_mode:
            if method != "RESTORE":
                # calculate tensor residuals
                # Predict a signal given tensor parameters.
                s_est = dti.tensor_prediction(tenfit.model_params, gtab, S0=tenfit.S0_hat)
                res = np.abs(image_data - s_est)

                # estimate res in the myocardium per diffusion image
                myo_pxs = np.flatnonzero(myo_mask[slice_idx])
                res_img = np.squeeze(np.reshape(res, [res.shape[0] * res.shape[1], res.shape[2], res.shape[3]]))
                res_img = np.nanmean(res_img[myo_pxs, :], axis=0)
                residuals_img[slice] = res_img
                # estimate res per voxel
                res_map = np.nanmean(np.squeeze(res), axis=2)
                residuals_map[slice] = res_map

                z_scores, outliers, outliers_pos = get_residual_z_scores(res_img)

                if settings["debug"]:
                    plot_residuals_plot(res_img, slice_idx, settings, prefix="")
                    plot_residuals_map(res_map, slice_idx, settings, prefix="")

            else:
                residuals_img = []
                residuals_map = []
        else:
            residuals_img = []
            residuals_map = []

    # reorder tensor to: [slice, lines, cols, 3x3 tensor]
    tensor = tensor.transpose(4, 0, 1, 2, 3)

    if not quick_mode:
        if settings["debug"]:
            plot_tensor_components(tensor, slices, settings)

    return tensor, residuals_img, residuals_map, info
