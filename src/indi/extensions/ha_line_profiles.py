"""
Calculates transmural gradient information from the helix angle maps
Also generates useful distance maps and bullseye maps
"""

import logging
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from indi.extensions.manual_lv_segmentation import get_epi_contour, get_sa_contours
from indi.extensions.polygon_selector import spline_interpolate_contour


def get_ha_line_profiles_and_distance_maps(
    HA: NDArray,
    lv_centres: dict,
    slices: NDArray,
    mask_3c: NDArray,
    segmentation: dict,
    settings: dict,
    info: dict,
    average_images: NDArray,
    logger: logging.Logger,
    ventricle: str = "LV",
) -> tuple[dict, dict, NDArray, NDArray, NDArray, NDArray, dict]:
    """
    Compute helix angle (HA) line profiles, wall thickness, bullseye maps, and distance maps for each slice.

    This function extracts transmural HA line profiles from the LV center to the epicardium, computes wall thickness,
    and generates bullseye maps and distance maps (endo, epi, and transmural) for each slice. It also calculates
    summary statistics for the transmural HA profile.

    Args:
        HA (NDArray): Array of helix angle maps for each slice.
        lv_centres (dict): Dictionary mapping slice indices to LV center coordinates.
        slices (NDArray): Array or list of slice indices to process.
        mask_3c (NDArray): 3-class U-Net segmentation mask (background, LV myocardium, RV).
        segmentation (dict): Dictionary with segmentation information for LV contours.
        settings (dict): Configuration and output settings.
        info (dict): Additional information, including pixel spacing.
        average_images (NDArray): Array of average images for each slice.
        logger (logging.Logger): Logger for debug and information messages.
        ventricle (str, optional): Ventricle name to process. Defaults to "LV".

    Returns:
        tuple:
            ha_lines_profiles (dict): Dictionary with HA line profiles for each slice.
            wall_thickness (dict): Dictionary with wall thickness values for each slice.
            bullseye_maps (NDArray): Array of bullseye maps for all slices.
            distance_endo_maps (NDArray): Array of endocardium distance maps for all slices.
            distance_epi_maps (NDArray): Array of epicardium distance maps for all slices.
            distance_transmural_maps (NDArray): Array of transmural (relative) distance maps for all slices.
            ha_lines_profiles_2 (dict): Dictionary with transmural HA profile statistics for each slice.
    """
    # lenth of line profile interpolation (from endo to epi)
    interp_len = 50

    # dictionary to store the data in a dict for each slice
    ha_lines_profiles = {}
    ha_lines_profiles_2 = {}
    wall_thickness = {}

    # arrays to store the bullseye and distance maps
    bullseye_maps = np.zeros(mask_3c.shape)
    distance_endo_maps = np.zeros(mask_3c.shape)
    distance_epi_maps = np.zeros(mask_3c.shape)
    distance_transmural_maps = np.zeros(mask_3c.shape)

    # loop over each slice
    for slice_idx in slices:

        ha_lines_profiles[slice_idx] = {}
        wall_thickness[slice_idx] = {}
        ha_lines_profiles_2[slice_idx] = {}

        # current HA map
        c_HA = np.copy(HA[slice_idx])
        # current U-Net mask
        c_mask = np.copy(mask_3c[slice_idx])
        # make the mask binary, remove RV and all non LV myocardium is 0
        c_mask[c_mask == 2] = 0
        # HA map background is nan
        c_HA[c_mask == 0] = np.nan
        # make mask uint8 to be used with cv2
        c_mask = np.array(c_mask * 255, dtype=np.uint8)

        # get the contours of the epicardium and endocardium
        if segmentation[slice_idx]["endocardium"].size != 0:
            epi_contour, endo_contour = get_sa_contours(c_mask)
        else:
            # if we don't have the endocardium segmented, there is no reason
            # to get the HA line profile information
            epi_contour = get_epi_contour(c_mask)
            # endo_contour is going to be the centroid of the LV mask
            endo_contour = np.array(
                [
                    [int(lv_centres[slice_idx][1]), int(lv_centres[slice_idx][0])],
                ]
            )

        # gather the HA values from the centre of the LV to each epicardial point
        # and also the wall thickness
        lp_matrix = np.empty((len(epi_contour), interp_len))
        wt = []
        # loop over each epicardial point
        for point_idx in range(len(epi_contour)):
            # Extract the line profiles from the centre to each point of the epicardium
            # These are in _pixel_ coordinates!!
            # centre
            y0, x0 = [int(x) for x in lv_centres[slice_idx]]
            # epicardial point
            x1, y1 = epi_contour[point_idx]

            # length of the line
            length = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

            # points along the line
            x, y = np.linspace(x0, x1, length + 1), np.linspace(y0, y1, length + 1)

            # remove background points
            # find coordinates where HA is nan
            lp = c_HA[y.astype(int), x.astype(int)]
            x = x[~np.isnan(lp)]
            y = y[~np.isnan(lp)]

            # remove the first and last 10% of the line to minimise partial volume effects
            # at least one pixel will be removed from each end
            trim_amount = int(np.ceil(0.1 * len(x)))
            x_trim = x[trim_amount:-trim_amount]
            y_trim = y[trim_amount:-trim_amount]

            # Extract the HA values along the line
            lp = c_HA[y_trim.astype(int), x_trim.astype(int)]
            lp_all = c_HA[y.astype(int), x.astype(int)]

            # get wall thickness in mm (Assuming square pixels with pixel spacing the same in x and y!)
            c_wt = len(lp_all) * info["pixel_spacing"][0]
            wt.append(c_wt)

            if len(lp) < 2:
                # not enough points to create a line profile
                lp = np.full((interp_len,), np.nan)
                lp_matrix[point_idx, :] = lp

            else:
                # The function fix_angle_wrap is not used anymore, as it can be dangerous and remove
                # too much data, or mask disease. It will also misbehave with unusual data like phantoms.
                # # fix angle wrap
                # lp_wrap_fix = fix_angle_wrap(lp, 45)

                # interpolate line profile valid points to interp_len
                lp = np.interp(np.linspace(0, len(lp), interp_len), np.linspace(0, len(lp), len(lp)), lp)
                # store line profile in a matrix
                lp_matrix[point_idx, :] = lp

        # store HA line profile matrix and wall thickness in dictionaries
        ha_lines_profiles[slice_idx]["lp_matrix"] = lp_matrix
        wall_thickness[slice_idx]["wt"] = wt

        # fit a line to the mean line profile
        average_lp = np.nanmean(lp_matrix, axis=0)

        std_lp = np.nanstd(lp_matrix, axis=0)
        model = LinearRegression()
        x = np.linspace(0.1, 0.9, len(lp)).reshape(-1, 1)
        model.fit(x, average_lp.reshape(-1, 1))
        r_sq = model.score(x, average_lp.reshape(-1, 1))
        y_pred = model.predict(x)
        slope = model.coef_[0, 0]

        # store all this info in the dictionary
        ha_lines_profiles[slice_idx]["average_lp"] = average_lp
        ha_lines_profiles[slice_idx]["std_lp"] = std_lp
        ha_lines_profiles[slice_idx]["r_sq"] = r_sq
        ha_lines_profiles[slice_idx]["slope"] = slope
        ha_lines_profiles[slice_idx]["y_pred"] = y_pred

        # ================================================================
        # ================================================================
        # plot HA line profiles v1
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        if not np.isnan(x).all():
            plt.plot(x, lp_matrix.T, color="green", alpha=0.03)
            # plt.plot(average_lp, linewidth=2, color="black", label="mean")
            plt.errorbar(x, average_lp, std_lp, linewidth=2, color="black", label="mean", elinewidth=0.5)
            plt.plot(x, y_pred, linewidth=1, color="red", linestyle="--", label="fit")
        plt.xlabel("normalised wall from endo to epi")
        plt.ylabel("HA (degrees)")
        plt.tick_params(axis="both", which="major")
        plt.title(
            "Linear fit (Rsq = "
            + "%.2f" % r_sq
            + " slope = "
            + "%.2f" % slope
            + " std = "
            + "%.2f" % np.mean(std_lp)
            + ")",
            fontsize=8,
        )
        plt.ylim(-90, 90)
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "HA_line_profiles_" + "slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

        # ================================================================
        # ================================================================
        # bulls eye map with 3 segments
        epi_contour_interp = spline_interpolate_contour(epi_contour, 1000, join_ends=False)
        if endo_contour.shape[0] > 4:
            endo_contour_interp = spline_interpolate_contour(endo_contour, 1000, join_ends=False)
        else:
            endo_contour_interp = endo_contour.copy()

        # order the countour points by the angle with the centre of the LV
        y_center, x_center = lv_centres[slice_idx]

        # loop over slices and get the phi_matrix for each slice
        # get the angle of each point in the epi and endo contours
        phi_matrix_epi = []
        phi_matrix_endo = []
        for coords in epi_contour_interp:
            phi_matrix_epi.append(-np.arctan2(coords[0] - x_center, coords[1] - y_center))
        for coords in endo_contour_interp:
            phi_matrix_endo.append(-np.arctan2(coords[0] - x_center, coords[1] - y_center))

        # reorder the contours according to the phi_matrix
        pos = np.argsort(phi_matrix_epi)
        epi_contour_interp = epi_contour_interp[pos]
        pos = np.argsort(phi_matrix_endo)
        endo_contour_interp = endo_contour_interp[pos]

        # create the new contours
        endo_mid = (epi_contour_interp - endo_contour_interp) * (1 / 3) + endo_contour_interp
        epi_mid = (epi_contour_interp - endo_contour_interp) * (2 / 3) + endo_contour_interp

        # create mask from the contours
        binary_mask = np.copy(mask_3c[slice_idx])
        binary_mask[binary_mask != 1] = 0
        ring_1 = np.zeros(binary_mask.shape)
        ring_1 = cv2.fillPoly(ring_1, [np.array(epi_mid, np.int32)], 1)
        ring_1[binary_mask == 0] = np.nan

        ring_2 = np.zeros(binary_mask.shape)
        ring_2 = cv2.fillPoly(ring_2, [np.array(endo_mid, np.int32)], 1)
        ring_2[binary_mask == 0] = np.nan

        bull_map = binary_mask * 3 - ring_1 - ring_2

        # calculate three distance maps
        # distance_map_endo: distance from the endocardium
        # distance_map_epi: distance from the epicardium
        # distance_map_transmural: relative distance radially from the endocardium to the epicardium
        distance_map_endo = np.zeros(binary_mask.shape)
        distance_map_epi = np.zeros(binary_mask.shape)
        distance_map_transmural = np.zeros(binary_mask.shape)

        # get coordinates of the pixels in the binary mask
        coords = np.where(binary_mask != 0)
        for point_idx in range(len(coords[0])):
            # get the closest point from endo_contour
            x1, y1 = coords[1][point_idx], coords[0][point_idx]
            dist = np.sqrt((endo_contour_interp[:, 0] - x1) ** 2 + (endo_contour_interp[:, 1] - y1) ** 2)
            distance = np.min(dist)
            distance_map_endo[y1, x1] = distance
            # get the closest point from epi_contour
            dist = np.sqrt((epi_contour_interp[:, 0] - x1) ** 2 + (epi_contour_interp[:, 1] - y1) ** 2)
            distance = np.min(dist)
            distance_map_epi[y1, x1] = distance
            # calculate the normalised transmural distance
            distance_map_transmural[y1, x1] = (distance_map_endo[y1, x1]) / (
                distance_map_epi[y1, x1] + distance_map_endo[y1, x1]
            )

        distance_map_endo[binary_mask == 0] = np.nan
        distance_map_epi[binary_mask == 0] = np.nan
        distance_map_transmural[binary_mask == 0] = np.nan

        # store the bullseye map
        bullseye_maps[slice_idx] = bull_map
        distance_endo_maps[slice_idx] = distance_map_endo
        distance_epi_maps[slice_idx] = distance_map_epi
        distance_transmural_maps[slice_idx] = distance_map_transmural

        # bin the ha values per distance from the endocardium
        # start with 17 bins, and reduce until we have at least 5 values in each bin
        n_bins = 17 + 1
        n_empty_bins = 1
        while n_empty_bins > 0:
            n_bins -= 1
            distance_bins = np.linspace(0.1, 0.9, n_bins)
            # bin_centres = (distance_bins[:-1] + distance_bins[1:]) / 2
            delta_bins = distance_bins[1] - distance_bins[0]
            c_values_y = HA[slice_idx]
            c_values_x = distance_map_transmural

            # Create binned values for HA based on distance bins
            norm_binned_ha = [
                c_values_y[(c_values_x > low) & (c_values_x <= high)]
                for low, high in zip(distance_bins[:-1], distance_bins[1:])
            ]

            # count number of bins with less than 5 values
            n_empty_bins = [1 for bin_values in norm_binned_ha if len(bin_values) < 5]
            n_empty_bins = len(n_empty_bins)

        # Calculate median and percentiles for each bin
        ha_transmural_median = np.array([np.nanmedian(bin_values) for bin_values in norm_binned_ha])
        ha_transmural_25 = np.array([np.nanpercentile(bin_values, 25) for bin_values in norm_binned_ha])
        ha_transmural_75 = np.array([np.nanpercentile(bin_values, 75) for bin_values in norm_binned_ha])
        ha_transmural_iqr = ha_transmural_75 - ha_transmural_25

        # fit a line to the median HA
        model = LinearRegression()
        x = distance_bins[:-1] + (delta_bins * 0.5).reshape(-1, 1)
        x = np.transpose(x, (1, 0))
        model.fit(x, ha_transmural_median)
        r_sq = model.score(x, ha_transmural_median.reshape(-1, 1))
        y_pred = model.predict(x)
        slope = model.coef_[0]

        # store all this info in the dictionary
        ha_lines_profiles_2[slice_idx]["median"] = ha_transmural_median
        ha_lines_profiles_2[slice_idx]["q25"] = ha_transmural_25
        ha_lines_profiles_2[slice_idx]["q75"] = ha_transmural_75
        ha_lines_profiles_2[slice_idx]["iqr"] = ha_transmural_iqr
        ha_lines_profiles_2[slice_idx]["r_sq"] = r_sq
        ha_lines_profiles_2[slice_idx]["slope"] = slope
        ha_lines_profiles_2[slice_idx]["y_pred"] = y_pred

        # ================================================================
        # ================================================================
        # plot HA line profiles v2
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.plot(x, ha_transmural_median.T, color="green", alpha=0.03)
        plt.errorbar(
            x,
            ha_transmural_median,
            yerr=[
                np.array(ha_transmural_median) - np.array(ha_transmural_25),
                np.array(ha_transmural_75) - np.array(ha_transmural_median),
            ],
            linewidth=2,
            color="black",
            label="mean",
            elinewidth=0.5,
        )
        plt.plot(x, y_pred, linewidth=1, color="red", linestyle="--", label="fit")
        plt.xlabel("normalised wall from endo to epi")
        plt.ylabel("HA (degrees)")
        plt.tick_params(axis="both", which="major")
        plt.title(
            "Linear fit (Rsq = "
            + "%.2f" % r_sq
            + " slope = "
            + "%.2f" % slope
            + " mean_iqr = "
            + "%.2f" % np.mean(ha_transmural_iqr)
            + ")",
            fontsize=8,
        )
        plt.ylim(-90, 90)
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout(pad=1.0)
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "HA_line_profiles_2_" + "slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

        # ================================================================
        # ================================================================
        # plot the bulls eye maps
        if settings["debug"]:
            cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps.get_cmap("Set1").colors[0:3])
            alphas_whole_heart = np.copy(mask_3c[slice_idx])
            alphas_whole_heart[alphas_whole_heart > 0.1] = 1
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(average_images[slice_idx], cmap="Greys_r")
            i = ax[0, 0].imshow(bull_map, alpha=alphas_whole_heart * 0.7, cmap=cmap, vmin=1, vmax=3)
            cbar = plt.colorbar(i, fraction=0.046, pad=0.04)
            cbar.set_ticks([4 / 3, 2, 8 / 3])
            cbar.set_ticklabels(["1", "2", "3"])
            cbar.ax.tick_params(labelsize=5)
            ax[0, 0].axis("off")
            ax[0, 0].set_title("Bullseye")
            ax[0, 1].imshow(average_images[slice_idx], cmap="Greys_r")
            i = ax[0, 1].imshow(distance_map_transmural, alpha=alphas_whole_heart * 0.7, cmap="Reds", vmin=0, vmax=1)
            cbar = plt.colorbar(i, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=5)
            ax[0, 1].axis("off")
            ax[0, 1].set_title("Transmural relative distance")

            ax[1, 0].imshow(average_images[slice_idx], cmap="Greys_r")
            i = ax[1, 0].imshow(distance_map_endo, alpha=alphas_whole_heart * 0.7, cmap="Reds")
            cbar = plt.colorbar(i, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=5)
            ax[1, 0].axis("off")
            ax[1, 0].set_title("Distance from endo")

            ax[1, 1].imshow(average_images[slice_idx], cmap="Greys_r")
            i = ax[1, 1].imshow(distance_map_epi, alpha=alphas_whole_heart * 0.7, cmap="Reds")
            cbar = plt.colorbar(i, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=5)
            ax[1, 1].axis("off")
            ax[1, 1].set_title("Distance from epi")
            plt.tight_layout(pad=1.0)
            plt.savefig(
                os.path.join(
                    settings["debug_folder"],
                    f"{ventricle}_bullseye_map_" + "slice_" + str(slice_idx).zfill(3) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

    return (
        ha_lines_profiles,
        wall_thickness,
        bullseye_maps,
        distance_endo_maps,
        distance_epi_maps,
        distance_transmural_maps,
        ha_lines_profiles_2,
    )


def fix_angle_wrap(lp: NDArray, angle_jump=90) -> NDArray:
    """
    Remove points and all subsequent points where the line profile differs from 90 deg
    or more from the previous point.

    The line profile is divided in two, and the analyses start from the meso point
    towards the endo and epi separately. Angle wrap is more common towards the
    endo and epi borders.

    Args:
        lp: line profile before fix
        angle_jump: angle jump threshold

    Returns:
        lp_new: line profile after fix

    """

    # NEW METHOD
    diff = np.diff(lp)
    pos = diff > angle_jump
    if pos.any():
        pos = np.argmax(diff > angle_jump)
        lp_new = np.copy(lp)
        lp_new[pos + 1 :] = np.nan
    else:
        lp_new = np.copy(lp)

    return lp_new
