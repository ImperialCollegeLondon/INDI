import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_eigenvalues_histograms(eigenvalues: NDArray, settings: dict, mask_3c: NDArray):
    """
    Plot eigenvalues histograms

    Parameters
    ----------
    eigenvalues: NDArray with eigenvalues [slice, row, col, eigenvalue_order]
    settings: dictionary with the path to the debug folder
    mask_3c: U-Net segmentation
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
        os.path.join(settings["results"], "LV_eigenvalues_histograms.png"),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def plot_eigenvector_maps(eigenvectors, slices, settings):
    # plot the eigenvectors
    direction_str = ["x", "y", "z"]
    order_str = ["tertiary", "secondary", "primary"]
    for slice_idx in slices:
        fig, ax = plt.subplots(3, 3)
        for idx, eig_order in enumerate(range(2, -1, -1)):
            for direction in range(3):
                i = ax[idx, direction].imshow(eigenvectors[slice_idx, :, :, direction, eig_order], vmin=-1, vmax=1)
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
            dpi=200,
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def make_eigenvectors_z_positive(eigenvectors: NDArray) -> NDArray:
    """
    Flip all the eigenvectors to have the same z orientation for debugging

    Parameters
    ----------
    eigenvectors: array with eigenvectors

    Returns
    -------
    flipped eigenvectors
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
):
    """
    Save the negative eigenvalues map

    Parameters
    ----------
    eigenvalues: array with negative eigenvalues
    slices: array with slice positions
    info: useful info
    average_images: array with the average image for each slice
    settings: useful info
    mask_3c: U-Net segmentation
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

    alphas = np.copy(negative_eig_map[slice_idx])
    alphas[alphas > 0.1] = 1
    cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Set3").colors[1:4])
    for slice_idx in slices:
        plt.figure(figsize=(5, 5))
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(negative_eig_map[slice_idx], alpha=alphas, vmin=1, vmax=3, cmap=cmap)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_ticks([4 / 3, 2, 8 / 3])
        cbar.set_ticklabels(["1", "2", "3"])
        plt.title("negative eigenvalues slice: " + str(slice_idx).zfill(2), fontsize=7)
        plt.tick_params(axis="both", which="major", labelsize=5)
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "negative_eigenvalues_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=200,
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def get_eigensystem(
    dti: dict,
    slices: NDArray,
    info: dict,
    average_images: NDArray,
    settings: dict,
    mask_3c: NDArray,
    logger: logging.Logger,
) -> [dict, dict]:
    """

    Calculate eigenvalues and eigenvectors of the DTI tensor

    Parameters
    ----------
    dti : dict
        dictionary with DTI variables
    slices : NDArray
        array with slice strings
    info : dict
    average_images : NDArray
        average normalised images
    settings : dict
    mask_3c : NDArray
        segmentation mask
    options : dict
    logger : logging.Logger

    Returns
    -------
    [dict, dict]

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
    get_negative_eigenvalues_map(dti["eigenvalues"], slices, info, average_images, settings, mask_3c)
    # make dti["eigenvectors"] point z positive for easier debugging
    dti["eigenvectors"] = make_eigenvectors_z_positive(dti["eigenvectors"])

    # remove negative dti["eigenvalues"] by converting to a very small number
    dti["eigenvalues"][dti["eigenvalues"] < 0] = np.spacing(1)

    # plot histograms of the dti["eigenvalues"]
    # and also maps for the eigenvectors if debug is True
    plot_eigenvalues_histograms(dti["eigenvalues"], settings, mask_3c)
    if settings["debug"]:
        plot_eigenvector_maps(dti["eigenvectors"], slices, settings)

    return dti, info
