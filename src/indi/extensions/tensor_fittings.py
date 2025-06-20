import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from scipy import stats
from tqdm import tqdm

# import time


def plot_residuals_plot(residuals: NDArray, slice_idx: int, settings: dict, prefix: str = ""):
    """
    Plot the tensor residuals averaged per image and save the figure.

    Args:
        residuals (NDArray): Residuals per image (averaged over myocardial voxels).
        slice_idx (int): Index of the slice being plotted.
        settings (dict): Dictionary with configuration and output paths.
        prefix (str, optional): Prefix for the output file name. Defaults to "".

    Returns:
        None
    """
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(residuals)
    plt.xlabel("image #")
    plt.ylabel("residuals")
    plt.tick_params(axis="both", which="major")
    plt.title("Tensor residuals (average of all myocardial voxels)")
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["results"],
            "results_b",
            "tensor_residuals" + prefix + "_slice_" + str(slice_idx).zfill(2) + ".png",
        ),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def plot_residuals_map(
    residuals: NDArray, average_images: NDArray, mask_3c: NDArray, slice_idx: int, settings: dict, prefix: str = ""
):
    """
    Plot the tensor residuals averaged per pixel and save the figure.

    Args:
        residuals (NDArray): Residuals per pixel.
        average_images (NDArray): Array of average images for each slice.
        mask_3c (NDArray): 3D segmentation mask array.
        slice_idx (int): Index of the slice being plotted.
        settings (dict): Dictionary with configuration and output paths.
        prefix (str, optional): Prefix for the output file name. Defaults to "".

    Returns:
        None
    """
    alphas_whole_heart = np.copy(mask_3c[slice_idx])
    alphas_whole_heart[alphas_whole_heart > 0.1] = 1
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(average_images[slice_idx], cmap="Greys_r")
    plt.imshow(residuals, alpha=alphas_whole_heart)
    plt.colorbar(fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=5)
    plt.title("Tensor residuals (average of all DWIs)")
    plt.axis("off")
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["results"],
            "results_b",
            "tensor_residuals_map" + prefix + "_slice_" + str(slice_idx).zfill(2) + ".png",
        ),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def get_residual_z_scores(residuals: NDArray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Compute z-scores and identify outliers from residuals.

    Args:
        residuals (NDArray): Array with residual values.

    Returns:
        tuple:
            z_scores (NDArray): Z-scores of the residuals for each image.
            outliers (NDArray): Boolean array indicating outliers (z-score > 3).
            outliers_pos (NDArray): Indices of outlier positions, sorted by descending z-score.
    """
    z_scores = stats.zscore(residuals)
    # Select data points with z-scores above or below 3
    outliers = np.abs(z_scores) > 3.0
    # store outliers by descending z-score
    idxs = np.nonzero(outliers)[0]
    ascending_order = z_scores[idxs].argsort()
    outliers_pos = np.flip(idxs[ascending_order])

    return z_scores, outliers, outliers_pos


def plot_tensor_components(D: NDArray, average_images: NDArray, mask_3c: NDArray, slices: NDArray, settings: dict):
    """
    Plot and save the six unique tensor components for each slice.

    Args:
        D (NDArray): Diffusion tensor array, shape [n_slices, rows, cols, 3, 3].
        average_images (NDArray): Array with average images for each slice.
        mask_3c (NDArray): 3D segmentation mask array.
        slices (NDArray): Array of slice indices or positions.
        settings (dict): Dictionary with configuration and output paths.

    Returns:
        None
    """
    mask = np.copy(mask_3c)
    mask[mask == 2] = 0

    myo_tensor = np.copy(D)
    myo_tensor[mask == 0] = np.nan

    tensor_mean = np.nanmean(myo_tensor)
    tensor_std = np.nanstd(myo_tensor)
    vmin = tensor_mean - 3 * tensor_std
    vmax = tensor_mean + 3 * tensor_std

    for slice_idx in slices:
        alphas_whole_heart = np.copy(mask_3c[slice_idx])
        alphas_whole_heart[alphas_whole_heart > 0.1] = 1

        plt.figure(figsize=(15, 15))
        plt.subplot(3, 3, 1)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 0, 0], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxx")

        plt.subplot(3, 3, 5)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 1, 1], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dyy")

        plt.subplot(3, 3, 9)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 2, 2], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Dzz")
        plt.axis("off")

        plt.subplot(3, 3, 2)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 0, 1], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxy")

        plt.subplot(3, 3, 3)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 0, 2], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.title("Dxz")

        plt.subplot(3, 3, 6)
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(D[slice_idx, :, :, 1, 2], vmin=vmin, vmax=vmax, alpha=alphas_whole_heart)
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
    average_images: NDArray,
    logger: logging.Logger,
    method: str = "NLLS",
    quick_mode=False,
):
    """
    Fit a diffusion tensor model to the data using DiPy.

    Supports multiple fitting methods: 'LS' (Linear Least Squares), 'WLS' (Weighted Linear Least Squares),
    'NLLS' (Non-Linear Least Squares), and 'RESTORE'. Optionally computes and plots residuals and tensor components.

    Args:
        slices (NDArray): Array of slice indices or positions.
        data (pd.DataFrame): DataFrame containing diffusion-weighted image data and metadata.
        info (dict): Dictionary with general scan and image information.
        settings (dict): Dictionary with processing and output settings.
        mask_3c (NDArray): 3D segmentation mask array.
        average_images (NDArray): Array of average images for each slice.
        logger (logging.Logger): Logger for status and debug messages.
        method (str, optional): Tensor fitting method. Defaults to "NLLS".
        quick_mode (bool, optional): If True, skips residual calculations and plotting. Defaults to False.

    Returns:
        tuple:
            tensor (NDArray): Fitted diffusion tensor array, shape [n_slices, rows, cols, 3, 3].
            s0 (NDArray): Estimated S0 images, shape [n_slices, rows, cols].
            residuals_img (dict or list): Residuals per image for each slice, or empty list if not computed.
            residuals_map (dict or list): Residuals per voxel for each slice, or empty list if not computed.
            info (dict): Updated info dictionary with fitting statistics.
    """
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table

    logger.info("Starting tensor fitting with method: " + method)

    tensor = np.zeros([info["img_size"][0], info["img_size"][1], 3, 3, mask_3c.shape[0]])
    s0 = np.zeros([info["img_size"][0], info["img_size"][1], info["n_slices"]])
    residuals_img = {}
    residuals_map = {}

    myo_mask = np.copy(mask_3c.reshape(mask_3c.shape[0], mask_3c.shape[1] * mask_3c.shape[2]))
    myo_mask[myo_mask > 1] = 0

    # I need to do this per slice, because gtab might differ from slice to slice
    info["tensor fitting sigma"] = {}

    for i, slice_idx in enumerate(tqdm(slices, desc="Tensor fitting")):
        current_entries = data.loc[data["slice_integer"] == slice_idx]

        # remove any images that have been marked to be removed
        current_entries = current_entries.loc[current_entries["to_be_removed"] == False]

        if not current_entries["bmatrix"].isnull().all():
            # If the b-matrix is present, we should use it instead of b-values and b-vectors
            if i == 0:
                message_tensor_fitting_flag = 0
            bmatrix = np.stack(current_entries["bmatrix"].values)
            vals, vectors = np.linalg.eigh(bmatrix)
            idx = vals.argsort(axis=1)[:, ::-1]
            gradient = np.array([vectors[i, :, idx[i, 0]] for i in range(vals.shape[0])])
            bvals = np.trace(bmatrix, axis1=-1, axis2=-2)
            gtab = gradient_table(bvals, bvecs=gradient, btens=bmatrix)
        else:
            if i == 0:
                message_tensor_fitting_flag = 1
            bvals = current_entries["b_value"].values
            bvecs = np.vstack(current_entries["diffusion_direction"])
            gtab = gradient_table(bvals, bvecs=bvecs, atol=0.5)

        image_data = np.stack(current_entries["image"])
        image_data = image_data[..., np.newaxis]
        image_data = image_data.transpose(1, 2, 3, 0)
        # t0 = time.time()

        if method == "NLLS" or method == "RESTORE":
            tenmodel = dti.TensorModel(gtab, fit_method=method, return_S0_hat=True)
        else:
            tenmodel = dti.TensorModel(gtab, fit_method=method, return_S0_hat=True)

        tenfit = tenmodel.fit(image_data)
        tensor[..., slice_idx] = np.squeeze(tenfit.quadratic_form)
        s0[..., slice_idx] = np.squeeze(tenfit.S0_hat)

        # t1 = time.time()
        # total = t1 - t0
        # logger.info(f"Slice {slice_idx}: Time for tensor fitting: {total = :.3f} seconds")

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
                residuals_img[slice_idx] = res_img
                # estimate res per voxel
                res_map = np.nanmean(np.squeeze(res), axis=2)
                residuals_map[slice_idx] = res_map

                z_scores, outliers, outliers_pos = get_residual_z_scores(res_img)

                if settings["debug"]:
                    plot_residuals_plot(res_img, slice_idx, settings, prefix="")
                    plot_residuals_map(res_map, average_images, mask_3c, slice_idx, settings, prefix="")

            else:
                residuals_img = []
                residuals_map = []
        else:
            residuals_img = []
            residuals_map = []

    if message_tensor_fitting_flag == 0:
        logger.info("Tensor fitting used: b-values and b-matrix")
    elif message_tensor_fitting_flag == 1:
        logger.info("Tensor fitting used: b-values and b-vecs")

    # reorder tensor to: [slice, lines, cols, 3x3 tensor]
    tensor = tensor.transpose(4, 0, 1, 2, 3)
    s0 = s0.transpose(2, 0, 1)

    if not quick_mode:
        if settings["debug"]:
            plot_tensor_components(tensor, average_images, mask_3c, slices, settings)

    return tensor, s0, residuals_img, residuals_map, info
