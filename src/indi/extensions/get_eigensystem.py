import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_eigenvalues_histograms(eigenvalues: NDArray, settings: dict, mask_3c: NDArray):
    """Plot histograms of eigenvalues for the LV myocardium.

    Args:
        eigenvalues (NDArray): Array with shape ``[slice, row, col, 3]``
            ordered from smallest to largest.
        settings (dict): Configuration dictionary; must include
            ``debug_folder`` with the output path.
        mask_3c (NDArray): Three-class heart segmentation mask.
    """

    myo_mask = np.copy(mask_3c)
    myo_mask[myo_mask == 2] = 0
    myo_mask = myo_mask[..., np.newaxis]
    eigenvalues_myo = eigenvalues * myo_mask
    myo_values_ev1 = eigenvalues_myo[:, :, :, 0].flatten() * 1e3
    myo_values_ev1 = myo_values_ev1[myo_values_ev1 > np.spacing(1)]
    myo_values_ev2 = eigenvalues_myo[:, :, :, 1].flatten() * 1e3
    myo_values_ev2 = myo_values_ev2[myo_values_ev2 > np.spacing(1)]
    myo_values_ev3 = eigenvalues_myo[:, :, :, 2].flatten() * 1e3
    myo_values_ev3 = myo_values_ev3[myo_values_ev3 > np.spacing(1)]
    w = 0.04
    start_val = 0
    end_val = 4
    plt.figure(figsize=(5, 5))
    plt.hist(
        x=myo_values_ev1,
        bins=np.arange(start_val, end_val + w, w),
        alpha=0.8,
        rwidth=0.85,
        label="L3",
    )
    plt.hist(
        x=myo_values_ev2,
        bins=np.arange(start_val, end_val + w, w),
        alpha=0.8,
        rwidth=0.85,
        label="L2",
    )
    plt.hist(
        x=myo_values_ev3,
        bins=np.arange(start_val, end_val + w, w),
        alpha=0.8,
        rwidth=0.85,
        label="L1",
    )
    plt.legend(loc="upper right", fontsize=7)
    plt.title("Eigenvalues", fontsize=7)
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(settings["debug_folder"], "LV_eigenvalues_histograms.png"),
        dpi=100,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def plot_eigenvector_maps(
    eigenvectors: NDArray,
    average_images: NDArray,
    mask_3c: NDArray,
    slices: NDArray,
    settings: dict,
) -> None:
    """Plot eigenvector component maps overlaid on average images.

    Args:
        eigenvectors (NDArray): Array of eigenvectors with shape
            ``[slices, rows, cols, xyz, order]``.
        average_images (NDArray): Normalised average image per slice.
        mask_3c (NDArray): Three-class heart segmentation mask used for
            alpha masking.
        slices (NDArray): Slice indices to iterate over.
        settings (dict): Configuration; must include ``debug_folder``.
    """
    # plot the eigenvectors
    direction_str = ["x", "y", "z"]
    order_str = ["tertiary", "secondary", "primary"]
    for slice_idx in slices:
        alphas_whole_heart = np.copy(mask_3c[slice_idx])
        alphas_whole_heart[alphas_whole_heart > 0.1] = 1
        fig, ax = plt.subplots(3, 3)
        for idx, eig_order in enumerate(range(2, -1, -1)):
            for direction in range(3):
                ax[idx, direction].imshow(average_images[slice_idx], cmap="Greys_r")
                i = ax[idx, direction].imshow(
                    eigenvectors[slice_idx, :, :, direction, eig_order],
                    vmin=-1,
                    vmax=1,
                    alpha=alphas_whole_heart,
                    cmap="RdYlBu",
                )
                ax[idx, direction].set_title(order_str[eig_order] + ": " + direction_str[direction], fontsize=7)
                ax[idx, direction].axis("off")
                plt.tick_params(axis="both", which="major", labelsize=5)
                cbar = plt.colorbar(i)
                cbar.ax.tick_params(labelsize=5)
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(
                settings["debug_folder"],
                "eigenvector_components_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def make_eigenvectors_z_positive(eigenvectors: NDArray) -> NDArray:
    """Flip eigenvectors so that their z-component is always positive.

    This normalises orientation ambiguity for easier visual debugging.

    Args:
        eigenvectors (NDArray): Array of eigenvectors with shape
            ``[slices, rows, cols, xyz, order]``.

    Returns:
        NDArray: Eigenvector array with z-components flipped to be positive.
    """
    temp = eigenvectors.swapaxes(3, 4)
    temp = np.reshape(
        temp,
        [temp.shape[0] * temp.shape[1] * temp.shape[2] * temp.shape[3], temp.shape[4]],
    )
    test2 = temp[:, 2] < 0
    temp[test2, :] = -temp[test2, :]
    temp = np.reshape(
        temp,
        [
            eigenvectors.shape[0],
            eigenvectors.shape[1],
            eigenvectors.shape[2],
            eigenvectors.shape[4],
            eigenvectors.shape[3],
        ],
    )
    temp = temp.swapaxes(3, 4)
    eigenvectors = temp

    return eigenvectors


def get_negative_eigenvalues_map(
    eigenvalues: NDArray,
    slices: NDArray,
    info: dict,
    average_images: NDArray,
    settings: dict,
    mask_3c: NDArray,
) -> NDArray:
    """Compute and plot a map of negative-eigenvalue voxels per slice.

    Args:
        eigenvalues (NDArray): Eigenvalue array with shape
            ``[slices, rows, cols, 3]``.
        slices (NDArray): Slice indices to process and save.
        info (dict): Metadata dictionary containing ``n_slices`` and
            ``img_size``.
        average_images (NDArray): Normalised average image per slice, used
            as the underlay for the overlay plot.
        settings (dict): Configuration; must include ``results`` path.
        mask_3c (NDArray): Three-class heart segmentation mask.

    Returns:
        NDArray: Negative-eigenvalue count map with shape
        ``[n_slices, rows, cols]``, where each voxel value indicates how many
        of the three eigenvalues are negative.
    """

    background_mask = np.copy(mask_3c)
    background_mask[background_mask > 0] = 1
    negative_eig_map = np.zeros([info["n_slices"], info["img_size"][0], info["img_size"][1]])
    for slice_idx in slices:
        eig_1 = eigenvalues[slice_idx, :, :, 0] < 0
        eig_2 = eigenvalues[slice_idx, :, :, 1] < 0
        eig_3 = eigenvalues[slice_idx, :, :, 2] < 0
        eig_1 = eig_1.astype(int) * background_mask[slice_idx]
        eig_2 = eig_2.astype(int) * background_mask[slice_idx]
        eig_3 = eig_3.astype(int) * background_mask[slice_idx]

        negative_eig_map[slice_idx] = eig_1 + eig_2 + eig_3

    cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps.get_cmap("Set3").colors[1:4])
    for slice_idx in slices:
        alphas = np.copy(negative_eig_map[slice_idx])
        alphas[alphas > 0.1] = 1
        plt.figure(figsize=(5, 5))
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(negative_eig_map[slice_idx], alpha=alphas, vmin=1, vmax=3, cmap=cmap)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        # cbar.ax.tick_params(labelsize=5)
        cbar.set_ticks([4 / 3, 2, 8 / 3])
        cbar.set_ticklabels(["1", "2", "3"])
        plt.title("negative eigenvalues")
        plt.tick_params(axis="both", which="major")
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "negative_eigenvalues_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

    return negative_eig_map


def get_eigensystem(
    dti: dict,
    slices: NDArray,
    info: dict,
    average_images: NDArray,
    settings: dict,
    mask_3c: NDArray,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Compute eigenvalues and eigenvectors of the diffusion tensor.

    Decomposes the symmetric diffusion tensor stored in ``dti["tensor"]``,
    orients all eigenvectors so that z is positive, replaces negative
    eigenvalues with a small epsilon, and optionally saves diagnostic plots.

    Args:
        dti (dict): DTI data dictionary; must contain ``"tensor"`` and will be
            populated with ``"eigenvalues"``, ``"eigenvectors"``, and
            ``"negative_eigenvalues"``.
        slices (NDArray): Slice indices to process.
        info (dict): Metadata dictionary; will be updated with
            ``n_negative_eigenvalues`` and ``percentage_negative_eigenvalues``.
        average_images (NDArray): Normalised average image per slice.
        settings (dict): Configuration dictionary; ``debug`` toggles
            diagnostic plots.
        mask_3c (NDArray): Three-class heart segmentation mask.
        logger (logging.Logger): Logger for progress and diagnostic messages.

    Returns:
        tuple[dict, dict]: Updated ``dti`` dictionary and updated ``info``
        dictionary.
    """
    # we need to mask the nans from the tensor array
    dti["eigenvalues"], dti["eigenvectors"] = np.linalg.eigh(
        dti["tensor"]
    )  # use the fact that dti["tensor"] is symmetric to speed up the process

    # get info on the number of negative eigenvalues in the myocardium
    vals = dti["eigenvalues"][mask_3c == 1]
    neg_vals = vals[vals < 0]

    info["n_negative_eigenvalues"] = int(len(neg_vals))
    info["percentage_negative_eigenvalues"] = str(int((len(neg_vals) / vals.size) * 100))
    logger.debug(
        "Number of negative eigenvalues: "
        + str(info["n_negative_eigenvalues"])
        + " ("
        + str(int((len(neg_vals) / vals.size) * 100))
        + "%)"
    )

    # export negative dti["eigenvalues"] map
    dti["negative_eigenvalues"] = get_negative_eigenvalues_map(
        dti["eigenvalues"], slices, info, average_images, settings, mask_3c
    )
    # make dti["eigenvectors"] point z positive for easier debugging
    dti["eigenvectors"] = make_eigenvectors_z_positive(dti["eigenvectors"])

    # remove negative dti["eigenvalues"] by converting to a very small number
    dti["eigenvalues"][dti["eigenvalues"] < 0] = np.spacing(1)

    # plot histograms of the dti["eigenvalues"]
    # and also maps for the eigenvectors if debug is True
    if settings["debug"]:
        plot_eigenvalues_histograms(dti["eigenvalues"], settings, mask_3c)
        plot_eigenvector_maps(dti["eigenvectors"], average_images, mask_3c, slices, settings)

    return dti, info
