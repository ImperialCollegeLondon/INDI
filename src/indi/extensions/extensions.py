import copy
import csv
import logging
import math
import os
import pickle
import subprocess
import sys
from typing import Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters
import skimage.measure
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage
from skimage import morphology
from skimage.measure import label, regionprops_table

from indi.extensions.export_vectors_tensors_vtk import export_vectors_tensors_vtk, save_vtk_file


def clean_image(
    img: NDArray, slices: NDArray, factor: float = 0.5, blur: bool = False
) -> tuple[NDArray, NDArray, NDArray]:
    """Clean the image background by thresholding the image using Otsu's method.

    Args:
        img: image, array of floats scaled [0 1]
        slices: array with slice integers
        factor: Threshold reduction factor [0 1]. 1 means no reduction. Defaults to 0.5.)
        blur: option to blur the image

    Returns:
        clean_img: cleaned image
        mask: threshold mask
        thresh: threshold value used = Otsu's x factor

    """

    n_slices = img.shape[0]
    clean_img = np.zeros(img.shape)
    mask = np.zeros(img.shape)
    thresh = np.zeros(n_slices)

    for slice_idx in slices:
        if blur:
            # blur the image to denoise
            img[slice_idx] = skimage.filters.gaussian(img[slice_idx], sigma=2.0)

        # threshold based on Otsu's method
        thresh[slice_idx] = factor * (skimage.filters.threshold_otsu(img[slice_idx]))
        thresh_img = img[slice_idx].copy()
        thresh_img[thresh_img < thresh[slice_idx]] = 0
        mask[slice_idx] = np.multiply(thresh_img > thresh[slice_idx], 1)

        clean_img[slice_idx] = img[slice_idx] * mask[slice_idx]

    return clean_img, mask, thresh


def close_small_holes(mask: NDArray) -> NDArray:
    """Close small holes in the mask and add them to the rest of heart mask

    Args:
      mask: mask with 0: background, 1: LV, 2: rest of heart

    Returns:
        mask: NDArray: mask with holes filled

    """
    # convert mask to binary
    binary_mask = mask.astype(bool)
    binary_mask = binary_mask.astype(int)

    # close small holes in mask with size up to a square of 4x4
    new_mask = ndimage.binary_closing(binary_mask, structure=np.ones((4, 4))).astype(int)
    # get only the new pixels that should be filled
    diff_mask = new_mask - binary_mask
    # add those pixels to the rest of the heart label (value = 2).
    mask[diff_mask == 1] = 2

    return mask


def get_cylindrical_coordinates_short_axis(mask: NDArray) -> dict:
    """
    Function to calculate an approximate version of the local cardiac coordinates for a short-axis plane
    (radial, circumferential, and longitudinal vectors). They will be cylindrical coordinates with the
    centre on the FOV centre and the z-axis perpendicular to the image plane.

    Args:
        mask: mask to where to calculate the vectors

    Returns:
        heart_coordinates: a dictionary with radi, circ, long arrays
    """

    # the three orthogonal vectors
    long = np.zeros((mask.shape + (3,)))
    circ = np.zeros((mask.shape + (3,)))
    radi = np.zeros((mask.shape + (3,)))
    phi_matrix = np.zeros(mask.shape)

    # centre of image as we don't know yet the centre of the LV
    # we jut hope the two are close
    centre_of_images = np.array([mask.shape[1] / 2, mask.shape[2] / 2])
    coords = np.where(mask == 1)
    centre_of_images_coords = centre_of_images[..., np.newaxis]
    centre_of_images_coords = np.repeat(centre_of_images_coords, len(coords[0]), axis=1)
    centre_of_images_coords = np.vstack((coords[0], centre_of_images_coords))
    n_points = len(coords[0])
    phi_matrix[coords] = -np.arctan2(coords[1] - centre_of_images_coords[1], coords[2] - centre_of_images_coords[2])

    long[coords] = [0, 0, 1]

    circ[coords] = np.array(
        [
            np.sin(phi_matrix[coords]),
            -np.cos(phi_matrix[coords]),
            np.repeat(0, n_points),
        ]
    ).T

    radi[coords] = np.array(
        [
            np.cos(phi_matrix[coords]),
            np.sin(phi_matrix[coords]),
            np.repeat(0, n_points),
        ]
    ).T

    # output variable as a dictionary with all 3 vectors
    local_cylindrical_coordinates = {"long": long, "circ": circ, "radi": radi}

    # if settings["debug"]:
    #     # plot the cardiac coordinates maps
    #     direction_str = ["x", "y", "z"]
    #     order_keys = ["long", "circ", "radi"]
    #     for slice_idx, slice_str in enumerate(slices):
    #         fig, ax = plt.subplots(3, 3)
    #         for idx in range(3):
    #             for direction in range(3):
    #                 i = ax[idx, direction].imshow(
    #                     local_cylindrical_coordinates[order_keys[idx]][slice_idx, :, :, direction], vmin=-1, vmax=1
    #                 )
    #                 ax[idx, direction].set_title(order_keys[idx] + ": " + direction_str[direction], fontsize=7)
    #                 ax[idx, direction].axis("off")
    #                 plt.tick_params(axis="both", which="major", labelsize=5)
    #                 cbar = plt.colorbar(i)
    #                 cbar.ax.tick_params(labelsize=5)
    #         plt.tight_layout(pad=1.0)
    #         plt.savefig(
    #             os.path.join(
    #                 settings["debug_folder"],
    #                 "cardiac_coordinates_slice_" + slice_str + ".png",
    #             ),
    #             dpi=100,
    #             pad_inches=0,
    #             transparent=False,
    #         )
    #         plt.close()

    # maps = {"mag": mag_image, "mask": mask}
    # lcc = copy.deepcopy(local_cylindrical_coordinates)
    # save_vtk_file(lcc, {}, maps, info, "cylindrical_coordinates", settings["debug_folder"])

    return local_cylindrical_coordinates


def get_cardiac_coordinates_short_axis(
    mask: NDArray,
    segmentation: dict,
    slices: NDArray,
    n_slices: int,
    settings,
    dti: dict,
    average_images: NDArray,
    info: dict,
) -> tuple[dict, dict]:
    """
    Function to calculate the local cardiac coordinates for a short-axis plane
    (radial, circumferential, and longitudinal vectors)

    Args:
        mask: hearts masks
        segmentation: dict with segmentation info
        slices: array with slice integers
        n_slices: int with number of slices
        settings: dict
        dti: dictionary with DTI maps
        average_images: normalised average image per slice
        info: dict

    Returns:
        heart_coordinates: as a dictionary with radi, circ, long arrays
        lv_centres: dictionary with the LV centres for each slice
        phi_matrix: dictionary with the phi matrix for each slice
    """
    lv_centres = np.zeros([n_slices, 2], dtype=int)

    # the three orthogonal vectors
    long = np.zeros((mask.shape + (3,)))
    circ = np.zeros((mask.shape + (3,)))
    circ_adjusted = np.zeros((mask.shape + (3,)))
    radi_adjusted = np.zeros((mask.shape + (3,)))

    phi_matrix = {}

    for slice_idx in slices:
        # number of slices and get LV mask from mask
        # (contains LV myocardium and other heart structures)

        lv_mask = np.zeros(mask[slice_idx].shape)
        lv_mask[mask[slice_idx] == 1] = 1

        coords = np.where(lv_mask == 1)
        n_points = len(coords[0])
        # find the LV centre
        count = (lv_mask == 1).sum()
        x_center, y_center = np.round(np.argwhere(lv_mask == 1).sum(0) / count)
        lv_centres[slice_idx, :] = [x_center, y_center]

        phi_matrix[slice_idx] = np.zeros(lv_mask.shape)

        phi_matrix[slice_idx][coords] = -np.arctan2(coords[0] - x_center, coords[1] - y_center)

        long[slice_idx][coords] = [0, 0, 1]

        circ[slice_idx][coords] = np.array(
            [
                np.sin(phi_matrix[slice_idx][coords]),
                -np.cos(phi_matrix[slice_idx][coords]),
                np.repeat(0, n_points),
            ]
        ).T

        epi_points = np.flip(segmentation[slice_idx]["epicardium"])

        # remove last point because it is the same as the first
        epi_points = epi_points[:-1, :]
        # epi_points = np.unique(np.round(epi_points), axis=1)

        for idx in range(n_points):
            c_point = [coords[0][idx], coords[1][idx]]
            # distance of array to this point
            dist = np.sqrt((epi_points[:, 0] - c_point[0]) ** 2 + (epi_points[:, 1] - c_point[1]) ** 2)
            # get the index of the closest point
            closest_point_idx = np.argmin(dist)

            closest_wall_vec = np.array(
                [epi_points[closest_point_idx, 0], epi_points[closest_point_idx, 1]]
            ) - np.array([epi_points[closest_point_idx - 1, 0], epi_points[closest_point_idx - 1, 1]])

            # normalise and add z
            closest_wall_vec = np.divide(closest_wall_vec, np.linalg.norm(closest_wall_vec))
            closest_wall_vec = np.append(closest_wall_vec, 0)

            # convert angle from line col to usual xy directions
            closest_wall_vec = np.array([closest_wall_vec[1], -closest_wall_vec[0], 0])

            # angle with circ
            angle = np.rad2deg(np.arccos(np.dot(closest_wall_vec, circ[slice_idx][c_point[0], c_point[1]])))

            if angle > 90:
                closest_wall_vec = -closest_wall_vec

            circ_adjusted[slice_idx][c_point[0], c_point[1]] = np.array([closest_wall_vec[0], closest_wall_vec[1], 0])
            radi_adjusted[slice_idx][c_point[0], c_point[1]] = -np.cross(
                circ_adjusted[slice_idx][c_point[0], c_point[1]], long[slice_idx][c_point[0], c_point[1]]
            )

    # output variable as a dictionary with all 3 vectors
    local_cardiac_coordinates = {"long": long, "circ": circ_adjusted, "radi": radi_adjusted}

    if settings["debug"]:
        # plot the cardiac coordinates maps
        direction_str = ["x", "y", "z"]
        order_keys = ["long", "circ", "radi"]
        for slice_idx in slices:
            alphas_whole_heart = np.copy(mask[slice_idx])
            alphas_whole_heart[alphas_whole_heart > 0.1] = 1
            fig, ax = plt.subplots(3, 3)
            for idx in range(3):
                for direction in range(3):
                    ax[idx, direction].imshow(average_images[slice_idx], cmap="Greys_r")
                    i = ax[idx, direction].imshow(
                        local_cardiac_coordinates[order_keys[idx]][slice_idx, :, :, direction],
                        vmin=-1,
                        vmax=1,
                        alpha=alphas_whole_heart,
                        cmap="RdYlBu",
                    )
                    ax[idx, direction].set_title(order_keys[idx] + ": " + direction_str[direction], fontsize=7)
                    ax[idx, direction].axis("off")
                    plt.tick_params(axis="both", which="major", labelsize=5)
                    cbar = plt.colorbar(i)
                    cbar.ax.tick_params(labelsize=5)
            plt.tight_layout(pad=1.0)
            plt.savefig(
                os.path.join(
                    settings["debug_folder"],
                    "cardiac_coordinates_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        # save local_cardiac_coordinates to a vtk file
        maps = {"FA": dti["fa"], "MD": dti["md"], "mask": mask, "mean_img": average_images}
        # dictionaries and lists are mutable, so they will be modified also outside the function
        # so here, to prevent local_cardiac_coordinates dict to be modified I am creating a
        # deep copy.
        lcc = copy.deepcopy(local_cardiac_coordinates)
        if settings["debug"]:
            save_vtk_file(lcc, {}, maps, info, "cardiac_coordinates", settings["debug_folder"], save_mesh=True)

    return local_cardiac_coordinates, lv_centres, phi_matrix


def create_2d_montage(img_stack: NDArray) -> NDArray:
    """Creates a 2D montage of a 3D array of images

    Args:
        img_stack: 3d image stack

    Returns:
        montage: 2D image montage
    """
    n_images = img_stack.shape[0]
    n_rows = img_stack.shape[1]
    n_cols = img_stack.shape[2]

    # number of tiles in the column direction
    n_tiles_cols = 10

    # fill with empty image positions in order to make a big tile of images
    rounded_10 = math.ceil(n_images / n_tiles_cols) * n_tiles_cols
    empty_tiles = rounded_10 - n_images
    empty_tiles_array = np.zeros((empty_tiles, n_rows, n_cols))
    img_stack = np.concatenate((img_stack, empty_tiles_array), axis=0)

    # Number of tiles in the row direction
    n_tiles_rows = int(rounded_10 / 10)

    # create montage of images
    montage = (
        np.reshape(img_stack, (n_tiles_rows, n_tiles_cols, n_rows, n_cols))
        .swapaxes(1, 2)
        .reshape(n_rows * n_tiles_rows, n_cols * n_tiles_cols)
    )

    return montage


def get_colourmaps() -> dict:
    """
    Load the custom colormap RGB values

    Returns:
        Dictionary with Listed Colormaps

    """
    script_path = os.path.dirname(__file__)
    colormaps = {}
    MD = np.loadtxt(os.path.join(script_path, "colourmaps_data", "MD.txt"))
    md_cmap = matplotlib.colors.ListedColormap(MD)
    colormaps["MD"] = md_cmap
    ################
    FA = np.loadtxt(os.path.join(script_path, "colourmaps_data", "FA.txt"))
    fa_cmap = matplotlib.colors.ListedColormap(FA)
    colormaps["FA"] = fa_cmap
    ################
    E2A = np.loadtxt(os.path.join(script_path, "colourmaps_data", "abs_E2A.txt"))
    abs_e2a_cmap = matplotlib.colors.ListedColormap(E2A)
    colormaps["abs_E2A"] = abs_e2a_cmap

    HA = np.loadtxt(os.path.join(script_path, "colourmaps_data", "HA.txt"))
    ha_cmap = matplotlib.colors.ListedColormap(HA)
    colormaps["HA"] = ha_cmap

    return colormaps


def rad_to_mag(img: NDArray, max_value: int = 4096) -> NDArray:
    """Convert image array from radians to magnitude

    Args:
      img: input image in radians
      max_value: max value of the image

    Returns:
        img: image in magnitude
    """

    img = max_value * img / np.pi
    return img


def mag_to_rad(img: NDArray, max_value: int = 4096) -> NDArray:
    """Convert image array from magnitude to radians

    Args:
        img: input image in magnitude
        max_value: max value of the image

    Returns:
        img: image in radians

    """
    img = np.pi * img / max_value
    return img


def clean_mask(mask: NDArray) -> NDArray:
    """Clean a mask to leave only the biggest blob

    Args:
        mask: input mask

    Returns:
        clean_mask: mask with only the biggest blob

    """
    img_size = np.shape(mask)
    clean_mask = np.zeros([img_size[0], img_size[1], img_size[2]])

    for idx in range(img_size[0]):
        slice_mask = mask[idx]

        # find the biggest blob
        label_img = label(slice_mask)
        props = regionprops_table(label_img, properties=("area", "centroid"))
        table = pd.DataFrame(props)
        max_area = table["area"].max()

        # remove any blob with area smaller than the largest
        clean_mask[idx, :, :] = morphology.remove_small_objects(
            slice_mask.astype(bool), min_size=max_area, connectivity=2
        ).astype(int)
        clean_mask[idx, :, :] = clean_mask[idx, :, :] * slice_mask

    return clean_mask


def get_snr_maps(
    data: pd.DataFrame,
    mask_3c: NDArray,
    average_images: NDArray,
    slices: NDArray,
    settings: dict,
    logger: logging.Logger,
    info: dict,
) -> Tuple[dict, NDArray, dict, dict]:
    """Save the SNR maps.

    Args:
        data: dataframe with the diffusion images and info
        mask: U-Net mask of the heart
        average_images: array with average images
        slices: array with slice integers
        settings: dictionary with useful info
        logger: logger object
        info: dictionary with useful info

    Returns:
        snr: dictionary for each slice with nested dictionaries for each diffusion config
        noise: dictionary for each slice with nested dictionaries for each diffusion config
        snr_b0_lv: dictionary with the median and IQR of the SNR for b0 images
        info: updated dictionary with useful info
    """

    img_size = data.loc[0, "image"].shape

    # snr will be a dictionary for each slice with nested dictionaries for each
    # diffusion config
    snr = {}
    snr_b0_lv = {}
    noise = {}
    for slice_idx in slices:
        snr[slice_idx] = {}
        noise[slice_idx] = {}

    for slice_idx in slices:
        # dataframe for each slice
        current_entries = data.loc[data["slice_integer"] == slice_idx].copy()
        # how many diffusion configs do we have for this slice?
        current_entries["diffusion_direction"] = [tuple(lst_in) for lst_in in current_entries["diffusion_direction"]]
        diffusion_configs_table = (
            current_entries.groupby(["b_value_original", "diffusion_direction"]).size().reset_index(name="Freq")
        )
        # loop over each config, if 5 or more repetitions,
        # then calculate SNR map
        for i, row in diffusion_configs_table.iterrows():
            if row["Freq"] > 4:
                temp = current_entries[
                    (current_entries["b_value_original"] == row["b_value_original"])
                    & (current_entries["diffusion_direction"] == row["diffusion_direction"])
                ]
                temp = temp.reset_index(drop=True)
                # loop over all images and stack them
                img_stack = np.empty([row["Freq"], img_size[0], img_size[1]])
                for i2, row2 in temp.iterrows():
                    img_stack[i2, :, :] = row2["image"]

                # define key for the snr dictionary based on
                # the b-value and direction
                key_tuple = (
                    round(row["b_value_original"]),
                    round(row["diffusion_direction"][0], 2),
                    round(row["diffusion_direction"][1], 2),
                    round(row["diffusion_direction"][2], 2),
                )
                delimiter = "_"
                key_string = delimiter.join([str(value) for value in key_tuple])

                # avoid division by zeros in the background
                std_array = np.std(img_stack, axis=0)
                std_array[mask_3c[slice_idx] == 0] = np.nan

                snr[slice_idx][key_string] = np.divide(np.mean(img_stack, axis=0), std_array)
                noise[slice_idx][key_string] = std_array

                if round(row["b_value_original"]) == 0:
                    snr_values = snr[slice_idx][key_string]
                    snr_b0_lv[slice_idx] = {}
                    snr_b0_lv[slice_idx]["median"] = np.nanmedian(snr_values[mask_3c[slice_idx] == 1])
                    snr_b0_lv[slice_idx]["iqr"] = [
                        np.nanpercentile(snr_values[mask_3c[slice_idx] == 1], 25),
                        np.nanpercentile(snr_values[mask_3c[slice_idx] == 1], 75),
                    ]

    # we can only store these values if we have b0 data
    if bool(snr_b0_lv):
        # add to logger mean SNR for each slice for b0
        info["LV SNR b0"] = {}
        for slice_idx in slices:
            if slice_idx in snr_b0_lv:
                logger.debug(
                    "LV SNR for slice "
                    + str(slice_idx).zfill(2)
                    + " b0 images = "
                    + "%.2f" % snr_b0_lv[slice_idx]["median"]
                    + " ["
                    + "%.2f" % snr_b0_lv[slice_idx]["iqr"][0]
                    + ", "
                    + "%.2f" % snr_b0_lv[slice_idx]["iqr"][1]
                    + "]"
                )
            else:
                logger.debug(
                    "LV SNR for slice " + str(slice_idx).zfill(2) + " b0 images = " + "Not enough repetitions."
                )

    if settings["debug"]:
        for slice_idx in slices:
            alphas_whole_heart = np.copy(mask_3c[slice_idx])
            alphas_whole_heart[alphas_whole_heart > 0.1] = 1
            if len(snr[slice_idx]) > 0:
                plt.figure(figsize=(3 * len(snr[slice_idx]), 3))
                for idx, key in enumerate(snr[slice_idx]):
                    plt.subplot(1, len(snr[slice_idx]), idx + 1)
                    plt.imshow(average_images[slice_idx], cmap="Greys_r")
                    plt.imshow(snr[slice_idx][key], vmin=0, vmax=20, cmap="magma", alpha=alphas_whole_heart)
                    plt.axis("off")
                    plt.title(key, fontsize=7)
                    cbar = plt.colorbar(fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=7)
                plt.tight_layout(pad=1.0)
                plt.savefig(
                    os.path.join(settings["debug_folder"], "snr_maps_slice_" + str(slice_idx).zfill(2) + ".png"),
                    dpi=100,
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()

    return snr, noise, snr_b0_lv, info


def get_window(img: NDArray, mask: NDArray) -> tuple[float, float]:
    """Get the window for the image as 3 std from the mean

    Args:
        img: input image
        mask: mask to get the window

    Returns:
        vmin
        vmax
    """

    # check if mask is not empty
    if mask.size == 0:
        mask = np.ones_like(img)
    # get all the non-background pixel values from image
    px_values = img[mask == 1]
    # get mean and std
    img_mean = np.nanmean(px_values)
    img_std = np.nanstd(px_values)
    # define the values
    vmin = img_mean - 3 * img_std
    vmax = img_mean + 3 * img_std
    # cutoff at 0 in case vmin is negative
    if vmin < 0:
        vmin = 0

    return vmin, vmax


def crop_pad_rotate_array(img: NDArray, correct_size: list, allow_rotation: bool = False) -> NDArray:
    """Crop or pad array to the required size

    Args:
        img: input image
        correct_size: desired size
        allow_rotation: allow rotation of the image by  90 deg to match the size

    Returns:
        img: croped or padded image
    """

    def crop_and_pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
        """Crop or pad array to desired length along specific axis

        Args:
            array: input array
            target_length: desired length
            axis: axis to pad or crop, by default 0

        Returns
            array: cropped or padded array

        """
        # target length and array length need to be both even or both odd
        # if not, then pad array with 0s in this dimension
        if np.mod(target_length, 2) == 0 and np.mod(array.shape[axis], 2) == 1:
            npad = [(0, 0)] * array.ndim
            npad[axis] = tuple(map(sum, zip(npad[axis], (0, 1))))
            array = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

        if np.mod(target_length, 2) == 1 and np.mod(array.shape[axis], 2) == 0:
            npad = [(0, 0)] * array.ndim
            npad[axis] = tuple(map(sum, zip(npad[axis], (1, 0))))
            array = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

        # determine size of padding
        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            # if negative then we need to crop instead of padding
            include_range = (
                int((array.shape[axis] - target_length) * 0.5),
                int((array.shape[axis] + target_length) * 0.5),
            )
            array = array.take(indices=range(include_range[0], include_range[1]), axis=axis)
        else:
            npad = [(0, 0)] * array.ndim
            npad[axis] = (int(pad_size * 0.5), int(pad_size * 0.5))
            array = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

        return array

    img_dims_original = img.shape

    # if allow rotation is True, then before padding and cropping
    # we can do a 90 deg rotation to try and match the dimensions
    if allow_rotation:
        slices, lines, cols = img_dims_original
        if cols > lines:
            # rotate
            img = np.rot90(img, 1, axes=(1, 2))
            img_dims_original = img.shape

    # loop over each dimension
    n_dims = len(img_dims_original)
    for dim in range(n_dims):
        if img_dims_original[dim] != correct_size[dim]:
            img = crop_and_pad_along_axis(img, correct_size[dim], axis=dim)

    return img


def reshape_tensor_from_6_to_3x3(D6: NDArray) -> NDArray:
    """Reshape tensor from 6 to 3x3

    Args:
        D6: tensor with 6 components

    Returns:
        D33: tensor with 3x3 components
    """

    D33 = np.zeros((D6.shape[0], D6.shape[1], D6.shape[2], 3, 3))
    D33[:, :, :, 0, 0] = D6[:, :, :, 0]
    D33[:, :, :, 0, 1] = D6[:, :, :, 1]
    D33[:, :, :, 0, 2] = D6[:, :, :, 2]
    D33[:, :, :, 1, 0] = D6[:, :, :, 1]
    D33[:, :, :, 1, 1] = D6[:, :, :, 3]
    D33[:, :, :, 1, 2] = D6[:, :, :, 4]
    D33[:, :, :, 2, 0] = D6[:, :, :, 2]
    D33[:, :, :, 2, 1] = D6[:, :, :, 4]
    D33[:, :, :, 2, 2] = D6[:, :, :, 5]

    return D33


def plot_results_maps(
    slices: NDArray,
    mask_3c: NDArray,
    average_images: dict,
    dti: dict,
    segmentation: dict,
    colormaps: dict,
    settings: dict,
    folder_id: str,
):
    """Plots the main montage of results

    Args:
        slices: array with slice position as strings
        mask_3c: segmentation mask
        average_images: average image of each slice
        dti: dictionary DTI holds DTI parameters
        segmentation: LV segmentation info
        colormaps: DTI maps colormaps
        settings: settings
        folder_id: folder id string to use on the filename

    """

    params = {}

    params["MD"] = {}
    params["MD"]["var_name"] = "md"
    params["MD"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["MD"]["cmap"] = colormaps["MD"]
    params["MD"]["title"] = "Mean diffusivity"
    params["MD"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MD"]["scale"] = 1000
    params["MD"]["abs"] = False

    params["MAG_ANISOTROPY"] = {}
    params["MAG_ANISOTROPY"]["var_name"] = "mag_anisotropy"
    params["MAG_ANISOTROPY"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["MAG_ANISOTROPY"]["cmap"] = "viridis"
    params["MAG_ANISOTROPY"]["title"] = "Mag anisotropy"
    params["MAG_ANISOTROPY"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MAG_ANISOTROPY"]["scale"] = 1000
    params["MAG_ANISOTROPY"]["abs"] = False

    params["MODE"] = {}
    params["MODE"]["var_name"] = "mode"
    params["MODE"]["vmin_max"] = [-1, 1]
    params["MODE"]["cmap"] = "plasma"
    params["MODE"]["title"] = "Tensor mode"
    params["MODE"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MODE"]["scale"] = 1
    params["MODE"]["abs"] = False

    params["FA"] = {}
    params["FA"]["var_name"] = "fa"
    params["FA"]["vmin_max"] = [0, 1]
    params["FA"]["cmap"] = colormaps["FA"]
    params["FA"]["title"] = "Fractional anisotropy"
    params["FA"]["units"] = "[]"
    params["FA"]["scale"] = 1
    params["FA"]["abs"] = False

    params["FROB_NORM"] = {}
    params["FROB_NORM"]["var_name"] = "frob_norm"
    params["FROB_NORM"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["FROB_NORM"]["cmap"] = "inferno"
    params["FROB_NORM"]["title"] = "Frobenius norm"
    params["FROB_NORM"]["units"] = "10^{-3} mm^2s^{-1}"
    params["FROB_NORM"]["scale"] = 1000
    params["FROB_NORM"]["abs"] = False

    params["HA"] = {}
    params["HA"]["var_name"] = "ha"
    params["HA"]["vmin_max"] = [-90, 90]
    params["HA"]["cmap"] = colormaps["HA"]
    params["HA"]["title"] = "Helix angle"
    params["HA"]["units"] = "degrees"
    params["HA"]["scale"] = 1
    params["HA"]["abs"] = False

    params["TA"] = {}
    params["TA"]["var_name"] = "ta"
    params["TA"]["vmin_max"] = [-90, 90]
    params["TA"]["cmap"] = "twilight_shifted"
    params["TA"]["title"] = "Transverse angle"
    params["TA"]["units"] = "degrees"
    params["TA"]["scale"] = 1
    params["TA"]["abs"] = False

    params["E2A"] = {}
    params["E2A"]["var_name"] = "e2a"
    params["E2A"]["vmin_max"] = [-90, 90]
    params["E2A"]["cmap"] = "twilight_shifted"
    params["E2A"]["title"] = "Sheetlet angle"
    params["E2A"]["units"] = "degrees"
    params["E2A"]["scale"] = 1
    params["E2A"]["abs"] = False

    params["abs_E2A"] = {}
    params["abs_E2A"]["var_name"] = "e2a"
    params["abs_E2A"]["vmin_max"] = [0, 90]
    params["abs_E2A"]["cmap"] = colormaps["abs_E2A"]
    params["abs_E2A"]["title"] = "Absolute sheetlet angle"
    params["abs_E2A"]["units"] = "degrees"
    params["abs_E2A"]["scale"] = 1
    params["abs_E2A"]["abs"] = True

    # # plot montage 1, histograms and maps for all metrics
    # for slice_idx in slices:
    #     alphas_whole_heart = np.copy(mask_3c[slice_idx])
    #     alphas_whole_heart[alphas_whole_heart > 0.1] = 1

    #     alphas_myocardium = np.copy(mask_3c[slice_idx])
    #     alphas_myocardium[alphas_myocardium == 2] = 0
    #     alphas_myocardium[alphas_myocardium > 0.1] = 1

    #     plt.figure(figsize=(15, 5))

    #     for idx, param in enumerate(params):
    #         # plot maps
    #         plt.subplot(2, len(params), idx + 1)
    #         plt.imshow(average_images[slice_idx], cmap="Greys_r")
    #         if params[param]["abs"]:
    #             plt.imshow(
    #                 np.abs(dti[params[param]["var_name"]][slice_idx] * params[param]["scale"]),
    #                 alpha=alphas_whole_heart,
    #                 vmin=params[param]["vmin_max"][0],
    #                 vmax=params[param]["vmin_max"][1],
    #                 cmap=params[param]["cmap"],
    #             )
    #         else:
    #             plt.imshow(
    #                 dti[params[param]["var_name"]][slice_idx] * params[param]["scale"],
    #                 alpha=alphas_whole_heart,
    #                 vmin=params[param]["vmin_max"][0],
    #                 vmax=params[param]["vmin_max"][1],
    #                 cmap=params[param]["cmap"],
    #             )
    #         plt.colorbar(fraction=0.046, pad=0.04)
    #         plt.axis("off")
    #         plt.title(params[param]["title"])

    #         # plot histograms
    #         plt.subplot(2, len(params), idx + 1 + len(params))
    #         if params[param]["abs"]:
    #             vals = abs(dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"])
    #         else:
    #             vals = dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"]
    #         bins = np.linspace(params[param]["vmin_max"][0], params[param]["vmin_max"][1], 40)
    #         weights = np.ones_like(vals) / len(vals)
    #         n, bins, patches = plt.hist(vals, bins=bins, weights=weights, rwidth=0.95)
    #         bin_centers = 0.5 * (bins[:-1] + bins[1:])
    #         # scale values to interval [0,1]
    #         col = bin_centers - min(bin_centers)
    #         col /= max(col)
    #         cm = plt.cm.get_cmap(params[param]["cmap"])
    #         for c, p in zip(col, patches):
    #             plt.setp(p, "facecolor", cm(c))
    #         plt.title(params[param]["title"])

    #     plt.tight_layout(pad=1.0)
    #     fname = os.path.join(
    #         settings["results"],
    #         "tensor_parameter_maps_" + folder_id + "_slice_" + str(slice_idx).zfill(2) + ".png",
    #     )
    #     # On windows there is a maximun path lenght of ~256 characters
    #     if os.name == "nt" and len(os.path.abspath(fname)) > 250:
    #         fname = os.path.join(
    #             settings["results"],
    #             "tensor_parameter_maps_" + "_slice_" + str(slice_idx).zfill(2) + ".png",
    #         )
    #     plt.savefig(fname, dpi=300, pad_inches=0, transparent=False)
    #     plt.close()

    # Also plots maps individually
    for slice_idx in slices:
        alphas_whole_heart = np.copy(mask_3c[slice_idx])
        alphas_whole_heart[alphas_whole_heart > 0.1] = 1

        alphas_myocardium = np.copy(mask_3c[slice_idx])
        alphas_myocardium[alphas_myocardium == 2] = 0
        alphas_myocardium[alphas_myocardium > 0.1] = 1

        for param in params:
            plt.figure(figsize=(5, 5))
            plt.imshow(average_images[slice_idx], cmap="Greys_r")
            if params[param]["abs"]:
                plt.imshow(
                    np.abs(dti[params[param]["var_name"]][slice_idx] * params[param]["scale"]),
                    alpha=alphas_whole_heart,
                    vmin=params[param]["vmin_max"][0],
                    vmax=params[param]["vmin_max"][1],
                    cmap=params[param]["cmap"],
                )
            else:
                plt.imshow(
                    dti[params[param]["var_name"]][slice_idx] * params[param]["scale"],
                    alpha=alphas_whole_heart,
                    vmin=params[param]["vmin_max"][0],
                    vmax=params[param]["vmin_max"][1],
                    cmap=params[param]["cmap"],
                )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout(pad=1.0)
            plt.axis("off")
            plt.title(params[param]["title"])
            plt.savefig(
                os.path.join(
                    settings["results"],
                    "results_a",
                    "maps_" + param + "_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

            plt.figure(figsize=(5, 5))
            if params[param]["abs"]:
                vals = abs(dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"])
            else:
                vals = dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"]
            bins = np.linspace(params[param]["vmin_max"][0], params[param]["vmin_max"][1], 40)
            weights = np.ones_like(vals) / len(vals)
            n, bins, patches = plt.hist(vals, bins=bins, weights=weights, rwidth=0.95)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            # scale values to interval [0,1]
            col = bin_centers - min(bin_centers)
            col /= max(col)
            cm = plt.cm.get_cmap(params[param]["cmap"])
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cm(c))
            plt.title(params[param]["title"])
            plt.savefig(
                os.path.join(
                    settings["results"],
                    "results_a",
                    "histograms_" + param + "_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        # plot LV 12 sectors
        plt.figure(figsize=(5, 5))
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(
            dti["lv_sectors"][slice_idx],
            alpha=alphas_whole_heart * 0.5,
            vmin=1,
            vmax=12,
            cmap=matplotlib.colors.ListedColormap(matplotlib.colormaps.get_cmap("tab20c").colors[0:12]),
        )
        if segmentation[slice_idx]["anterior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["anterior_ip"][0],
                segmentation[slice_idx]["anterior_ip"][1],
                "2",
                color="tab:orange",
                markersize=20,
                alpha=1.0,
            )
        if segmentation[slice_idx]["inferior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["inferior_ip"][0],
                segmentation[slice_idx]["inferior_ip"][1],
                "1",
                color="tab:orange",
                markersize=20,
                alpha=1.0,
            )
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        ticklabels = [str(i) for i in range(1, 12 + 1)]
        tickpos = np.linspace(1.5, 12 - 0.5, 12)
        cbar.set_ticks(tickpos, labels=ticklabels)
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.title("LV sectors")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "lv_segments_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

        # plot S0 map
        plt.figure(figsize=(5, 5))
        # plt.imshow(average_images[slice_idx], cmap="Blues_r", vmin=0, vmax=1)
        vmin, vmax = get_window(dti["s0"][slice_idx], mask_3c[slice_idx])
        plt.imshow(dti["s0"][slice_idx], cmap="Greys_r", vmin=vmin, vmax=vmax)
        if segmentation[slice_idx]["epicardium"].size != 0:
            plt.plot(
                segmentation[slice_idx]["epicardium"][:, 0],
                segmentation[slice_idx]["epicardium"][:, 1],
                ".",
                color="tab:blue",
                markersize=0.5,
                alpha=1.0,
            )
        if segmentation[slice_idx]["endocardium"].size != 0:
            plt.plot(
                segmentation[slice_idx]["endocardium"][:, 0],
                segmentation[slice_idx]["endocardium"][:, 1],
                ".",
                color="tab:red",
                markersize=0.5,
                alpha=1.0,
            )
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.title("S0")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "maps_s0_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def get_xarray(info: dict, dti: dict, crop_mask: NDArray, slices: NDArray):
    """Create an xarray dataset with the DTI maps

    Args:
        info: dictionary with useful info
        dti: DTI maps
        crop_mask: crop mask used to get the pixel coordinates taking into account the cropping
        slices: slice strings array

    Returns:
        ds: xarray dataset
    """
    # create coordinates
    rows = np.linspace(1, info["original_img_size"][0], info["original_img_size"][0], dtype=int)
    rows_crop = np.linspace(
        info["crop_corner"][0], info["crop_corner"][0] + info["img_size"][0], info["img_size"][0], dtype=int
    )
    cols = np.linspace(1, info["original_img_size"][1], info["original_img_size"][1], dtype=int)
    cols_crop = np.linspace(
        info["crop_corner"][1], info["crop_corner"][1] + info["img_size"][1], info["img_size"][1], dtype=int
    )

    pos = np.linspace(1, 3, 3, dtype=int)
    vectors_xyz = np.linspace(1, 3, 3, dtype=int)

    ds = xr.Dataset(
        data_vars=dict(
            FA=(["slice", "row_crop", "cols_crop"], dti["fa"], {"units": "[]"}),
            MD=(["slice", "row_crop", "cols_crop"], dti["md"], {"units": "mm2/sec"}),
            eigenvalues=(
                ["slice", "row_crop", "cols_crop", "ev_order"],
                dti["eigenvalues"],
                {"units": "mm2/sec"},
            ),
            eigenvectors=(
                ["slice", "row_crop", "cols_crop", "vector_xyz", "ev_order"],
                dti["eigenvectors"],
                {"units": "[]"},
            ),
        ),
        coords=dict(
            slice=(["slice"], slices),
            row=(["row"], rows),
            col=(["col"], cols),
            row_crop=(["row_crop"], rows_crop),
            col_crop=(["col_crop"], cols_crop),
            ev_order=(["ev_order"], pos),
            vector_xyz=(["vector_xyz"], vectors_xyz),
        ),
        attrs=dict(description="diffusion scalar measurements"),
    )

    # add HA and E2A measures to dataset
    ds["HA"] = (["slice", "row_crop", "cols_crop"], dti["ha"], {"units": "degrees"})
    ds["E2A"] = (["slice", "row_crop", "cols_crop"], dti["e2a"], {"units": "degrees"})
    ds["TA"] = (["slice", "row_crop", "cols_crop"], dti["ta"], {"units": "degrees"})
    ds["abs_E2A"] = (["slice", "row_crop", "cols_crop"], np.abs(dti["e2a"]), {"units": "degrees"})
    ds["LV_sectors"] = (["slice", "row_crop", "cols_crop"], dti["lv_sectors"], {"units": "[]"})

    # add crop mask to dataset
    ds["crop_mask"] = (["row", "col"], crop_mask, {"units": "[]"})

    return ds


def export_to_hdf5(dti: dict, mask_3c: NDArray, settings: dict):
    """Export DTI maps to HDF5

    Args:
        dti: dict with DTI maps
        mask_3c: segmentation mask
        settings: dict with settings

    """
    with h5py.File(os.path.join(settings["results"], "data", "DTI_maps" + ".h5"), "w") as hf:
        for name, key in dti.items():
            if isinstance(key, np.ndarray):
                hf.create_dataset(name, data=key)
            elif isinstance(key, dict):
                for subname, subkey in key.items():
                    if isinstance(subkey, np.ndarray):
                        hf.create_dataset(name + "_" + str(subname), data=subkey)
                    elif isinstance(key, dict):
                        for subsubname, subsubkey in subkey.items():
                            if isinstance(subsubkey, np.ndarray):
                                hf.create_dataset(name + "_" + str(subname) + "_" + subsubname, data=subsubkey)
        hf.create_dataset("mask", data=mask_3c)

    # # to read a map example
    # with h5py.File(os.path.join(settings["results"], "data", "DTI_maps" + ".h5"), "r") as hf:
    #     data = hf["ha"][:]


def export_results(
    data: pd.DataFrame,
    dti: dict,
    info: dict,
    settings: dict,
    mask_3c: NDArray,
    slices: NDArray,
    average_images: NDArray,
    segmentation: dict,
    colormaps: dict,
    logger: logging.Logger,
):
    """Export results to disk: VTK, PNGs, HDF5, pickled dictionary and YAML.

    Args:
        data: dataframe with the diffusion images and info
        dti: DTI maps
        info: dictionary with useful info
        settings: dict with settings
        mask_3c: heart segmentation mask
        slices: array with slice position
        average_images: normalised averaged images
        segmentation: LV segmentation info on LV borders and insertion points
        colormaps: DTI tailored colormaps
        logger: logger object

    """

    logger.info("Exporting results to disk...")

    # create a string with the folder names
    folder_id = settings["work_folder"]
    folder_id = os.path.normpath(folder_id)
    folders = folder_id.split(os.sep)
    folder_id = folders[-4:-1]
    folder_id = "_".join(folder_id)

    # plot eigenvectors and tensor in VTK format
    export_vectors_tensors_vtk(dti, info, settings, mask_3c, average_images)

    # plot results maps
    plot_results_maps(slices, mask_3c, average_images, dti, segmentation, colormaps, settings, folder_id)

    # save xarray to NetCDF
    # ds.to_netcdf(os.path.join(settings["results"], "data", "DTI_maps.nc"))

    # save results to h5 file
    export_to_hdf5(dti, mask_3c, settings)

    # save results summary to a csv table
    export_summary_table(dti, settings, slices)

    # save to disk dti dictionary
    dti["info"] = info
    with open(os.path.join(settings["results"], "data", "DTI_data.dat"), "wb") as handle:
        pickle.dump(dti, handle)

    # save the final diffusion database
    # save the dataframe and the info dict
    data.attrs["mask"] = mask_3c
    save_path = os.path.join(settings["results"], "data", "database.zip")
    data.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})

    # get git commit hash
    def get_git_revision_hash():
        try:
            return (
                subprocess.check_output(
                    ["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))
                )
                .strip()
                .decode()
            )
        except:
            return "No Git repository found!"

    info["git_hash"] = get_git_revision_hash()

    # save to disk basic info in a csv file
    with open(os.path.join(settings["results"], "data", "DTI_data.csv"), "w", newline="") as handle1:
        w = csv.writer(handle1)
        w.writerows(info.items())

    # do final montage with image magick
    for slice_idx in slices:
        if os.name == "nt":
            os.system('powershell.exe "Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process')
            run_command = (
                "powershell.exe "
                + os.path.join(os.path.dirname(__file__), "montage_script_windows.ps1")
                + " "
                + os.path.abspath(settings["results"])
                + " "
                + str(slice_idx).zfill(2)
                + " "
                + folder_id
            )
        else:
            run_command = (
                "bash "
                + os.path.join(os.path.dirname(__file__), "montage_script.sh")
                + " "
                + os.path.abspath(settings["results"])
                + " "
                + str(slice_idx).zfill(2)
                + " "
                + folder_id
            )
        os.system(run_command)


def export_summary_table(dti: dict, settings: dict, slices: NDArray):
    """Export summary of DTI values to a csv table

    Args:
        dti: dictionary with DTI maps
        settings: dictionary with settings
        slices: array with slice position

    """
    # get absolute E2A and TA
    dti["abs_e2a"] = np.abs(dti["e2a"])
    dti["abs_ta"] = np.abs(dti["ta"])
    dti["md_1e3"] = dti["md"] * 1000
    dti["frob_norm_1e3"] = dti["frob_norm"] * 1000
    dti["mag_anisotropy_1e3"] = dti["mag_anisotropy"] * 1000
    var_list = ["fa", "mode", "md_1e3", "frob_norm_1e3", "mag_anisotropy_1e3", "ha", "e2a", "ta", "abs_e2a", "abs_ta"]
    str_list = ["FA", "MODE", "MD", "NORM", "MAG_ANISOTROPY", "HA", "E2A", "TA", "|E2A|", "|TA|"]
    # global values
    table_global = []
    table_global.append(["Global", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    table_global.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    for idx, var in enumerate(var_list):
        table_global.append(
            [
                str_list[idx],
                np.nanmean(dti[var]),
                np.nanstd(dti[var]),
                np.nanmedian(dti[var]),
                np.nanquantile(dti[var], 0.25),
                np.nanquantile(dti[var], 0.75),
                np.nanmin(dti[var]),
                np.nanmax(dti[var]),
            ]
        )

    # also calculate global values for SNR
    # gather all keys found in all slices
    keys = []
    for slice_idx in slices:
        keys.extend(dti["snr"][slice_idx].keys())
    keys = list(set(keys))
    # sort keys
    keys.sort()
    # loop over each key and collect all values from all slices
    for key in keys:
        values = []
        for slice_idx in slices:
            if key in dti["snr"][slice_idx]:
                c_vals = dti["snr"][slice_idx][key]
                c_vals = c_vals[~np.isnan(c_vals)]
                c_vals = c_vals[np.isfinite(c_vals)]
                values.extend(c_vals)
        table_global.append(
            [
                "SNR_" + key,
                np.nanmean(values),
                np.nanstd(values),
                np.nanmedian(values),
                np.nanquantile(values, 0.25),
                np.nanquantile(values, 0.75),
                np.nanmin(values),
                np.nanmax(values),
            ]
        )

    if len(slices) > 1:
        # per slice values
        table_per_slice = []
        table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        table_per_slice.append(["Per slice", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for idx, var in enumerate(var_list):
            table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            for slice_idx in slices:
                table_per_slice.append(
                    [
                        str_list[idx] + "_slice_" + str(slice_idx).zfill(2),
                        np.nanmean(dti[var][slice_idx]),
                        np.nanstd(dti[var][slice_idx]),
                        np.nanmedian(dti[var][slice_idx]),
                        np.nanquantile(dti[var][slice_idx], 0.25),
                        np.nanquantile(dti[var][slice_idx], 0.75),
                        np.nanmin(dti[var][slice_idx]),
                        np.nanmax(dti[var][slice_idx]),
                    ]
                )
        # do the same for SNR
        table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for slice_idx in slices:
            for key in dti["snr"][slice_idx]:
                c_vals = dti["snr"][slice_idx][key]
                c_vals = c_vals[~np.isnan(c_vals)]
                c_vals = c_vals[np.isfinite(c_vals)]
                table_per_slice.append(
                    [
                        "SNR_" + key + "_slice_" + str(slice_idx).zfill(2),
                        np.nanmean(c_vals),
                        np.nanstd(c_vals),
                        np.nanmedian(c_vals),
                        np.nanquantile(c_vals, 0.25),
                        np.nanquantile(c_vals, 0.75),
                        np.nanmin(c_vals),
                        np.nanmax(c_vals),
                    ]
                )

        # add to the global table
        table_global.extend(table_per_slice)

    # convert table to dataframe
    table_global = pd.DataFrame(
        table_global,
        columns=["Parameter", "Mean", "Std", "Median", "Q25", "Q75", "Min", "Max"],
    )
    table_global = table_global.round(decimals=2)
    # export dataframe csv
    table_global.to_csv(os.path.join(settings["results"], "results_table.csv"), index=False)


def get_lv_segments(
    segmentation: dict,
    phi_matrix: dict,
    mask_3c: NDArray,
    lv_centres: NDArray,
    slices: NDArray,
    logger: logging.Logger,
) -> NDArray:
    """
    Get array with the LV segments

    Args:
        segmentation: dict containing the LV borders and insertion points
        phi_matrix: dict containing numpy arrays with the
            angles with respect to the centre of the lV (counter-clockwise)
        mask_3c: 3D numpy array with the LV segmentation mask
        lv_centres: Dictionary containing the (y_centre, x_centre) tuple for the lv, for each slice
        slices: array with slice integers
        logger: logging.Logger

    Returns:
        the corresponding 12 segments DataArray for the left ventricle
    """

    LV_free_wall_n_segs = 8
    LV_septal_wall_n_segs = 4
    # n_slices = mask_3c.shape[0]

    # dictionary with keys: Tuple[slice_idx, segment_number(i.e. 1 to 12)]
    segments_and_points = {}

    # loop over slices
    for slice_idx in slices:
        if segmentation[slice_idx]["anterior_ip"].size != 0 and segmentation[slice_idx]["inferior_ip"].size != 0:
            lv_mask = np.copy(mask_3c[slice_idx])
            lv_mask[lv_mask == 2] = 0
            phi_matrix[slice_idx][lv_mask == 0] = np.nan
            phi_matrix[slice_idx] = -(phi_matrix[slice_idx] - np.pi)

            y_center, x_center = lv_centres[slice_idx]

            # superior and inferior insertion points must be defined in this order
            anterior_ins_pt = segmentation[slice_idx]["anterior_ip"]
            inferior_ins_pt = segmentation[slice_idx]["inferior_ip"]

            (
                anterior_ins_pt_col,
                anterior_ins_pt_row,
            ) = anterior_ins_pt
            (
                inferior_ins_pt_col,
                inferior_ins_pt_row,
            ) = inferior_ins_pt

            # angles of the insertion points [0 2pi] clockwise
            theta_ant_ins_pt = -(-np.arctan2(anterior_ins_pt_row - y_center, anterior_ins_pt_col - x_center) - np.pi)
            theta_inf_ins_pt = -(-np.arctan2(inferior_ins_pt_row - y_center, inferior_ins_pt_col - x_center) - np.pi)

            # ====================
            # Segments free-wall
            # ====================
            # angular span of the free wall
            if theta_ant_ins_pt < theta_inf_ins_pt:
                angle_span_free_wall = theta_inf_ins_pt - theta_ant_ins_pt
            else:
                angle_span_free_wall = 2 * np.pi - (theta_ant_ins_pt - theta_inf_ins_pt)
            # divide the span by 8 segments
            angle_span_free_wall /= LV_free_wall_n_segs

            # loop and gather the points for each lateral wall segment
            for segment_idx in range(LV_free_wall_n_segs):
                theta_start = theta_ant_ins_pt + segment_idx * angle_span_free_wall
                if theta_start > 2 * np.pi:
                    theta_start -= 2 * np.pi
                theta_end = theta_ant_ins_pt + (segment_idx + 1) * angle_span_free_wall
                if theta_end > 2 * np.pi:
                    theta_end -= 2 * np.pi

                if theta_end > theta_start:
                    points = np.argwhere((phi_matrix[slice_idx] >= theta_start) & (phi_matrix[slice_idx] < theta_end))
                else:
                    points = np.argwhere((phi_matrix[slice_idx] >= theta_start) | (phi_matrix[slice_idx] < theta_end))

                segments_and_points[(slice_idx, segment_idx + 1)] = points

            # ====================
            # Segments septal-wall
            # ====================
            # angular span of the free wall
            if theta_ant_ins_pt < theta_inf_ins_pt:
                angle_span_septal_wall = 2 * np.pi - (theta_inf_ins_pt - theta_ant_ins_pt)
            else:
                angle_span_septal_wall = theta_ant_ins_pt - theta_inf_ins_pt
            # divide the span by 4 segments
            angle_span_septal_wall /= LV_septal_wall_n_segs

            # loop and gather the points for each lateral wall segment
            for idx, segment_idx in enumerate(
                range(LV_free_wall_n_segs, (LV_free_wall_n_segs + LV_septal_wall_n_segs))
            ):
                theta_start = theta_inf_ins_pt + idx * angle_span_septal_wall
                if theta_start > 2 * np.pi:
                    theta_start -= 2 * np.pi
                theta_end = theta_inf_ins_pt + (idx + 1) * angle_span_septal_wall
                if theta_end > 2 * np.pi:
                    theta_end -= 2 * np.pi

                if theta_end > theta_start:
                    points = np.argwhere((phi_matrix[slice_idx] >= theta_start) & (phi_matrix[slice_idx] < theta_end))
                else:
                    points = np.argwhere((phi_matrix[slice_idx] >= theta_start) | (phi_matrix[slice_idx] < theta_end))

                segments_and_points[(slice_idx, segment_idx + 1)] = points

    # prepare the output
    segments_mask = np.zeros(mask_3c.shape)
    segments_mask[:] = np.nan
    for slice_idx in slices:
        if segmentation[slice_idx]["anterior_ip"].size != 0 and segmentation[slice_idx]["inferior_ip"].size != 0:
            for curr_segment in range(1, (LV_free_wall_n_segs + LV_septal_wall_n_segs + 1)):
                if (slice_idx, curr_segment) in segments_and_points.keys():
                    segments_mask[
                        slice_idx,
                        segments_and_points[slice_idx, curr_segment][:, 0],
                        segments_and_points[slice_idx, curr_segment][:, 1],
                    ] = curr_segment

    logger.debug("LV segmentation in sectors done.")

    return segments_mask


def query_yes_no(question, default="no"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def image_histogram_equalization(image: NDArray, number_bins: int = 256):
    """
    Equalize histogram in numpy array image for better visualisation
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    Args:
        image: NDArray with grayscale image
        number_bins: number of histogram bins

    Returns:
        image_equilized: NDArray with equalized image

    """

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def remove_slices(
    data: pd.DataFrame, slices: NDArray, segmentation: dict, logger: logging
) -> tuple[pd.DataFrame, NDArray, dict]:
    """Remove slices that are marked as to be removed for all entries

    Args:
        data: dataframe with the diffusion images and info
        slices: array with slice position
        segmentation: LV segmentation info on LV borders and insertion points
        logger: logger object

    Returns:
        data: dataframe with slices removed
        slices: array with new slice position
        segmentation: segmentation info with slices removed

    """

    for slice_idx in slices:
        c_data = data[data.slice_integer == slice_idx]
        # check if to_be_removed column are all true
        if c_data.to_be_removed.all():
            # if so, then remove all rows for this column
            data = data[data.slice_integer != slice_idx]

    original_n_slices = len(slices)

    # slices is going to be a list of all the integers
    slices = data.slice_integer.unique()

    # remove slices from segmentation
    deepcopy_segmentation = copy.deepcopy(segmentation)
    for slice_idx in deepcopy_segmentation:
        if slice_idx not in slices:
            segmentation.pop(slice_idx)

    n_slices = len(slices)
    if n_slices != original_n_slices:
        logger.info(f"Number of slices reduced from {original_n_slices} to {n_slices}")

    return data, slices, segmentation


def remove_outliers(data: pd.DataFrame, info: dict, settings: dict) -> tuple[pd.DataFrame, dict]:
    """
    Remove outlier images from the dataframe.

    Images that are marked for removal (i.e., with `to_be_removed == True`) will be excluded from the dataframe.
    The function also updates the info dictionary with the new number of images. Optionally, it creates montages
    of removed and retained images for debugging if enabled in settings.

    Args:
        data (pd.DataFrame): DataFrame containing diffusion images and associated information.
        info (dict): Dictionary with metadata and processing information.
        settings (dict): Dictionary with configuration and debug options.

    Returns:
        tuple:
            pd.DataFrame: DataFrame with outlier images removed.
            dict: Updated info dictionary with the number of remaining images.
    """

    # this code can be used to double check if the images that are
    # set to be removed are in fact the correct ones
    # if settings["debug"]:
    #     # montage all the images that are to be removed or not removed
    #     create_image_montage(data, settings, flag=True)
    #     create_image_montage(data, settings, flag=False)

    # remove the rejected images from the dataframe
    data = data.loc[data["to_be_removed"] == False]
    data.reset_index(drop=True, inplace=True)
    info["n_images"] = len(data)

    return data, info


# def create_image_montage(data: pd.DataFrame, settings: dict, flag: bool = True):
#     """
#     Create a montage of images that are either marked for removal or retention.

#     This function generates and saves a montage of images from the dataframe, based on whether they are flagged
#     for removal (`to_be_removed == True`) or retention (`to_be_removed == False`). The montage is saved as a PNG
#     file in the debug folder specified in settings.

#     Args:
#         data (pd.DataFrame): DataFrame containing diffusion images and associated information. Must include an 'image' column and a 'to_be_removed' boolean column.
#         settings (dict): Dictionary with configuration and debug options. Must include a 'debug_folder' key.
#         flag (bool, optional): If True, creates a montage of images to be removed. If False, creates a montage of images to be retained. Defaults to True.
#     """

#     selected_images = data.loc[data["to_be_removed"] == flag]
#     if not selected_images.empty:
#         # create montage of the images to be removed
#         montage_images = []
#         for _, row in selected_images.iterrows():
#             c_img = row["image"]
#             # scale the image to 0-255
#             c_img = (c_img - np.min(c_img)) / (np.max(c_img) - np.min(c_img)) * 255
#             c_img = c_img.astype(np.uint8)
#             # add the image to the montage list
#             montage_images.append(c_img)

#         montage_images = np.array(montage_images)
#         test = skimage.util.montage(montage_images)

#         save_path = os.path.join(settings["debug_folder"], "outliers_" + str(flag) + ".png")

#         # save montage as a PNG file
#         plt.figure(figsize=(10, 10))
#         plt.imshow(test, cmap="Greys_r")
#         plt.axis("off")
#         plt.tight_layout(pad=1.0)
#         plt.savefig(
#             save_path,
#             dpi=100,
#             pad_inches=0,
#             transparent=False,
#         )
#         plt.close()
