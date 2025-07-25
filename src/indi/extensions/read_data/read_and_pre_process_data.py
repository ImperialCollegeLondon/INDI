import glob
import json
import logging
import math
import os
import re
import shutil
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import py7zr
import scipy.ndimage
from dotenv import dotenv_values
from numpy.typing import NDArray

from indi.extensions.read_data.dicom_to_h5_csv import (
    add_missing_columns,
    check_global_info,
    check_rows_and_columns,
    get_data_from_dicoms,
    interpolate_dicom_pixel_values,
    scale_dicom_pixel_values,
    tweak_directions_and_b_values,
)


def data_summary_plots_and_logs(data: pd.DataFrame, settings: dict, info: dict, logger: logging.Logger):
    """
    Summarises the data

    Produces a plot showing the b-value, direction, and slice
    Also prints a summary of the data to the log file

    Parameters:
        data: dataframe with all the dwi data
        settings: dictionary with useful info
        info: dictionary with useful info
        logger: logger for console and file
    """

    # get directions order as a numeric vector
    data["diffusion_direction"] = data["diffusion_direction"].apply(lambda x: tuple(x))
    unique_dirs = data["diffusion_direction"].unique().tolist()
    idxs = [item for item in range(0, len(unique_dirs) + 1)]
    dir_keys = {unique_dirs[i]: idxs[i] for i in range(len(unique_dirs))}

    direction_list = data["diffusion_direction"].tolist()
    direction_idxs = [dir_keys[direction_list[i]] for i in range(len(direction_list))]

    # save bar plots in a montage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(data.b_value_original.values, ".")
    plt.title("b-values", fontsize=7)
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.grid(linewidth=0.5, alpha=0.3, linestyle="--")

    plt.subplot(1, 3, 2)
    plt.plot(data.slice_integer.values, ".")
    plt.title("slice positions", fontsize=7)
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.grid(linewidth=0.5, alpha=0.3, linestyle="--")

    plt.subplot(1, 3, 3)
    plt.plot(direction_idxs, ".")
    plt.title("directions", fontsize=7)
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.grid(linewidth=0.5, alpha=0.3, linestyle="--")

    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["results"],
            "data_summary.png",
        ),
        dpi=100,
        pad_inches=0,
        transparent=False,
    )
    plt.close()

    # also print a summary of the data
    # print the number of images
    logger.debug("Number of images: " + str(info["n_images"]))

    # print the slice range
    unique_slices = data["slice_integer"].unique()
    logger.debug(
        "Number of slices: "
        + str(len(unique_slices))
        + " ["
        + str(unique_slices[0])
        + " : "
        + str(unique_slices[-1])
        + "]"
    )

    # image dimensions
    logger.debug("Image size: " + str(info["img_size"]))
    # resolution
    logger.debug(f"Slice spacing: {info["slice_spacing"]:0.2f}")
    logger.debug("Pixel spacing: " + str(info["pixel_spacing"]))

    # print the diffusion protocol in detail if not too complicated
    configs_table = data[["b_value_original", "slice_integer", "diffusion_direction_original"]]

    # configuration summary for slice X
    configs_table_first_slice = get_diffusion_summary_for_slice(logger, unique_slices, configs_table, 0)

    # check if there are other slices, if so then check if they have the same config
    # remove slice integer column
    configs_table_first_slice = configs_table_first_slice.drop(columns=["slice_integer"])

    # loop through all slices and check if directions and b-values are the same
    for slice_idx in unique_slices:
        c_table = configs_table.loc[configs_table["slice_integer"] == slice_idx]
        # check if b-values and directions are the same, if not print summary for this slice
        if not np.array_equal(
            np.sort(c_table["b_value_original"].values), np.sort(configs_table_first_slice["b_value_original"].values)
        ) or not np.array_equal(
            np.sort(c_table["diffusion_direction_original"].values),
            np.sort(configs_table_first_slice["diffusion_direction_original"].values),
        ):
            logger.debug("Different diffusion protocol for slice " + str(slice_idx))
            _ = get_diffusion_summary_for_slice(logger, unique_slices, configs_table, slice_idx)


def get_diffusion_summary_for_slice(
    logger: logging.Logger, slices: np.ndarray, configs_table: pd.DataFrame, slice_idx: int = 0
) -> pd.DataFrame:
    """Log the diffusion protocol for a given slice

    This function will log the diffusion protocol for a given slice
    and return a table with the same protocol.

    Parameters:
        logger: logger for console and file
        slices: array with the slice numbers
        configs_table: table with the diffusion protocols for all slices
        slice_idx: slice integer to return protocol and table, by default 0

    Returns:
        pd.DataFrame: table with the diffusion protocol for the given slice
    """
    configs_table_this_slice = configs_table.loc[configs_table["slice_integer"] == slices[slice_idx]]
    # different b-values
    b_values = configs_table_this_slice["b_value_original"].unique()

    if len(b_values) < 10:
        logger.debug("Diffusion protocol for slice: " + str(slice_idx))
        for bval in b_values:
            logger.debug(f"b-value = {bval}")
            c_table = configs_table_this_slice.loc[configs_table_this_slice["b_value_original"] == bval]
            logger.debug(f"   images: {len(c_table)}")
            # check if column contains lists or tuples
            if isinstance(c_table["diffusion_direction_original"].iloc[0], (list, tuple)):
                # convert to tuples if they are lists
                c_table.loc[:, "diffusion_direction_original"] = c_table["diffusion_direction_original"].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )
            c_table["diffusion_direction_original"].unique()
            logger.debug(f"   directions: {len(c_table["diffusion_direction_original"].unique())}")
            logger.debug(f"   repetitions: {len(c_table)/len(c_table["diffusion_direction_original"].unique())}")

    return configs_table_this_slice


def sort_by_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataframe by acquisition date and time

    Parameters
    ----------
    df: dataframe with diffusion database

    Returns
    -------
    dataframe with sorted values
    """
    # create a new column with date and time, drop the previous two columns
    # if this column doesn't exist already
    if "acquisition_date_time" in df.columns:
        try:
            df["acquisition_date_time"] = pd.to_datetime(df["acquisition_date_time"], format="%Y%m%d%H%M%S.%f")
        except ValueError:
            # Fallback if microseconds are missing
            df["acquisition_date_time"] = pd.to_datetime(df["acquisition_date_time"], format="%Y%m%d%H%M%S")

        df = df.sort_values(["acquisition_date_time"], ascending=True)
    else:
        df["acquisition_date_time"] = df["acquisition_date"] + " " + df["acquisition_time"]

        # check if acquisition date and time information exist
        if not (df["acquisition_date"] == "None").all():
            df["acquisition_date_time"] = pd.to_datetime(df["acquisition_date_time"], format="%Y%m%d %H%M%S.%f")
            # sort by date and time
            df = df.sort_values(["acquisition_date_time"], ascending=True)
        else:
            df["acquisition_date_time"] = "None"
            # if we don't have the acquisition time and date, then sort by series number? Not for now.
            # df = df.sort_values(["series_number"], ascending=True)

        # drop these two columns as we now have a single column with time and date
        df = df.drop(columns=["acquisition_date", "acquisition_time"])

    df = df.reset_index(drop=True)

    return df


def get_nii_pixel_array(nii_px_array: NDArray, c_slice_idx: int, c_frame_idx: int) -> NDArray:
    """
    Get the pixel array from a nii file

    Parameters
    ----------
    nii_px_array
    c_slice_idx
    c_frame_idx

    Returns
    -------
    pixel array

    """

    img = np.rot90(nii_px_array[:, :, c_slice_idx, c_frame_idx], k=1, axes=(0, 1))
    # check if largest dimension is lower than 192
    # if so, then interpolate array by a factor of two
    larger_dim = max(img.shape)
    interp_img = True if larger_dim <= 192 else False
    if interp_img:
        img = scipy.ndimage.zoom(img, 2, order=3)
        # zero potential negative values from the interpolation
        img[img < 0] = 0
    return img


def get_nii_diffusion_direction(dir: NDArray, settings: dict) -> list:
    """
    Get diffusion directions from a nii file
    Parameters
    ----------
    dir: NDArray
    settings: dict

    Returns
    -------
    directions: list
        list with the diffusion directions
    """
    if not np.any(dir):
        if settings["sequence_type"] == "steam":
            return list(1 / np.sqrt(3) * np.array([1.0, -1.0, 1.0]))
        else:
            return list(np.array([0.0, 0.0, 0.0]))
    else:
        return list(dir)


def get_nii_series_description(json_header: dict) -> str:
    """
    Get series description from the JSON header

    Parameters
    ----------
    json_header

    Returns
    -------
    series description

    """
    if "SeriesDescription" in json_header.keys():
        result = (json_header["SeriesDescription"].replace(" ", "_"),)
    else:
        result = "None"
    return result


def read_and_process_niis(
    list_nii: list, settings: dict, info: dict, logger: logging.Logger
) -> Tuple[pd.DataFrame, dict]:
    """
    Get diffusion and other parameters from the NIFTI files

    Parameters
    ----------
    list_nii
    settings
    info
    logger

    Returns
    -------
    dataframe with all the gathered information plus info dict with some global information

    """
    # opening first nii file
    first_nii = nib.load(os.path.join(settings["dicom_folder"], list_nii[0]))
    first_nii_header = first_nii.header

    # Opening first JSON file
    json_file = list_nii[0].replace(".nii", ".json")
    json_file = os.path.join(settings["dicom_folder"], json_file)
    with open(json_file) as _json_file:
        first_json_header = json.load(_json_file)

    # collect some global header info in a dictionary
    header_info = {}
    # get image comments, if not in the dictionary then add empty string
    if "ImageComments" in first_json_header.keys():
        header_info["image_comments"] = first_json_header["ImageComments"]
    else:
        header_info["image_comments"] = "None"
    # get image orientation, if not in the dictionary then add identity matrix
    if "ImageOrientationPatientDICOM" in first_json_header.keys():
        header_info["image_orientation_patient"] = first_json_header["ImageOrientationPatientDICOM"]
    else:
        header_info["image_orientation_patient"] = "None"
    # slice thickness
    if "SliceThickness" in first_json_header.keys():
        header_info["slice_thickness"] = first_json_header["SliceThickness"]
    else:
        header_info["slice_thickness"] = "None"

    temp_list = list(first_nii_header["pixdim"][1:3])
    header_info["pixel_spacing"] = [float(i) for i in temp_list]
    header_info["slice_distance"] = float(first_nii_header["pixdim"][3])

    # how many images per nii file
    n_images_per_file = first_nii.shape[3]
    n_slices_per_file = first_nii.shape[2]

    # start building a list for each image
    df = []
    for idx, nii_file in enumerate(list_nii):
        # load nii file and pixel array
        nii = nib.load(os.path.join(settings["dicom_folder"], nii_file))
        nii_px_array = np.array(nii.get_fdata())
        # load json file
        json_file = nii_file.replace(".nii", ".json")
        json_file = os.path.join(settings["dicom_folder"], json_file)
        with open(json_file) as _json_file:
            json_header = json.load(_json_file)
        # load b-value file
        bval_file = nii_file.replace(".nii", ".bval")
        bval_file = os.path.join(settings["dicom_folder"], bval_file)
        with open(bval_file) as _bval_file:
            bval = _bval_file.read()
            bval = bval.split()
            bval = [float(b) for b in bval]
        bval = np.array(bval)
        bval = np.repeat(bval[np.newaxis, :], n_slices_per_file, axis=0)
        bval_original = np.copy(bval)
        # load bvec file
        bvec_file = nii_file.replace(".nii", ".bvec")
        bvec_file = os.path.join(settings["dicom_folder"], bvec_file)
        with open(bvec_file) as _bvec_file:
            bvec = _bvec_file.read()
            bvec = bvec.split()
            bvec = [float(b) for b in bvec]
        bvec = np.array(bvec)
        bvec = bvec.reshape((3, int(len(bvec) / 3)))
        bvec = np.repeat(bvec[np.newaxis, :, :], n_slices_per_file, axis=0)

        # if file exists load rr interval csv file as a dataframe
        b_values_file = nii_file.replace(".nii", ".csv")
        b_values_file = os.path.join(settings["dicom_folder"], b_values_file)
        if os.path.exists(b_values_file):
            b_values_table = pd.read_csv(b_values_file)
            if idx == 0:
                logger.debug("Found file with adjusted b-values.")
        else:
            b_values_table = pd.DataFrame()
            if idx == 0:
                logger.debug("Did not find file with adjusted b-values.")

        # replace current array of b-values for this series
        if not b_values_table.empty:
            for slice_idx in range(bval.shape[0]):
                for img_idx in range(bval.shape[1]):
                    bval[slice_idx, img_idx] = b_values_table.loc[
                        (
                            (b_values_table["slice_dim_idx"] == slice_idx)
                            & (b_values_table["frame_dim_idx"] == img_idx)
                        ),
                        "b_value",
                    ].iloc[0]

        # if empty then return a table with nan values
        if b_values_table.empty:
            b_values_table = pd.DataFrame(
                index=np.arange(n_slices_per_file * n_images_per_file),
                columns=["nominal_interval_(msec)", "acquisition_time", "acquisition_date"],
            )
            b_values_table["nominal_interval"] = "None"
            b_values_table["acquisition_date_time"] = "None"
            b_values_table["slice_dim_idx"] = np.repeat(np.arange(n_slices_per_file), n_images_per_file)
            b_values_table["frame_dim_idx"] = np.tile(np.arange(n_images_per_file), n_slices_per_file)

        # loop over each slice and each image
        for slice_idx in range(n_slices_per_file):
            for img_idx in range(n_images_per_file):
                # append values (will be a row in the dataframe)
                df.append(
                    (
                        # file name
                        nii_file,
                        # array of pixel values
                        get_nii_pixel_array(nii_px_array, slice_idx, img_idx),
                        # b-value
                        bval[slice_idx, img_idx],
                        # b-value original
                        bval_original[slice_idx, img_idx],
                        # diffusion directions
                        get_nii_diffusion_direction(bvec[slice_idx, :, img_idx], settings),
                        # image position
                        (0, 0, first_nii_header["pixdim"][3] * slice_idx),
                        # nominal interval
                        b_values_table.loc[
                            (b_values_table["slice_dim_idx"] == slice_idx)
                            & (b_values_table["frame_dim_idx"] == img_idx),
                            "nominal_interval",
                        ].values[0],
                        # acquisition date and time
                        None,
                        # b_values_table.loc[
                        #     (b_values_table["slice_dim_idx"] == slice_idx)
                        #     & (b_values_table["frame_dim_idx"] == img_idx),
                        #     "acquisition_date_time",
                        # ].values[0],
                        # False if diffusion direction is a field
                        True,
                        # series description
                        get_nii_series_description(json_header),
                        # series number
                        json_header["SeriesNumber"],
                        # dictionary with header fields
                        json_header,
                    )
                )

    df = pd.DataFrame(
        df,
        columns=[
            "file_name",
            "image",
            "b_value",
            "b_value_original",
            "diffusion_direction",
            "image_position",
            "nominal_interval",
            "acquisition_date_time",
            "dir_in_image_plane",
            "series_description",
            "series_number",
            "header",
        ],
    )

    # convert acquisition date and time to a datetime object
    if not (df["acquisition_date_time"] == "None").all():
        df["acquisition_date_time"] = pd.to_datetime(df["acquisition_date_time"], format="%Y-%m-%d %H:%M:%S.%f")

    # merge header info into info
    info = {**info, **header_info}

    return df, info


def estimate_rr_interval(data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    This function will estimate the RR interval from the DICOM header
    and add it to the dataframe

    # if no nominal interval values are in the headers, then we will adjust
    # the b-values according to the RR interval by getting the time delta between images
    # convert time strings to microseconds

    Parameters
    ----------
    data: dataframe with diffusion database
    settings: dict

    Returns:
    -------
    dataframe with added estimated RR interval column

    """

    # check if we have acquisition date and time values
    # if so then estimate RR interval
    # if not then just copy the assumed values
    if not (data["acquisition_date_time"] == "None").all():
        # convert time to miliseconds
        time_stamps = data["acquisition_date_time"].astype(np.int64) / int(1e6)

        if settings["sequence_type"] == "steam":
            # get half the time delta between images
            # TODO account here for the number of slices per DICOM?
            time_delta = np.diff(time_stamps) * 0.5
        elif settings["sequence_type"] == "se":
            # get the time delta between images
            time_delta = np.diff(time_stamps)
        # prepend nan to the time delta
        time_delta = np.insert(time_delta, 0, np.nan)
        # get median time delta, and replace values above 4x the median with nan
        median_time = np.nanmedian(time_delta)
        time_delta[time_delta > 4 * median_time] = np.nan
        # add time delta to the dataframe
        data["estimated_rr_interval"] = time_delta
        # replace nans with the next non-nan value
        data["estimated_rr_interval"] = data["estimated_rr_interval"].bfill()

    else:
        data["estimated_rr_interval"] = "None"
        data["nominal_interval"] = "None"

    return data


def plot_b_values_adjustment(data: pd.DataFrame, settings: dict):
    """
    Plot b_values before and after adjustment, and the estimated RR intervals

    Parameters
    ----------
    data
    settings

    Returns
    -------

    """

    b_vals_original = data["b_value_original"].values
    b_vals = data["b_value"].values
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(b_vals_original, alpha=0.8)
    plt.plot(b_vals, alpha=0.8)
    plt.xlabel("image #")
    plt.ylabel("b-value")
    plt.legend(["original", "adjusted"])
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.title("b-values", fontsize=7)
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "data_b_values.png",
        ),
        dpi=100,
        pad_inches=0,
        transparent=False,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    if "nominal_interval" in data:
        plt.plot(data["nominal_interval"], alpha=0.8)
    if settings["sequence_type"] == "steam":
        plt.plot(data["estimated_rr_interval"], alpha=0.8)
    plt.legend(["nominal", "adjusted RR"])
    plt.xlabel("image #")
    plt.ylabel("nominal intervals")
    plt.ylim([0, 2000])
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.title("nominal intervals", fontsize=7)
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "nominal_and_rr_intervals.png",
        ),
        dpi=100,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def adjust_b_val_and_dir(
    data: pd.DataFrame,
    settings: dict,
    info: dict,
    logger: logging.Logger,
    data_type: str,
) -> pd.DataFrame:
    """
    This function will adjust:
    . b-values according to the recorded RR interval
    . diffusion-directions rotate header directions to the image plane

    data: dataframe with diffusion database
    settings: dict
    info: dict
    data: dataframe with diffusion database
    logger: logger for console and file
    data_type: str with the type of data (dicom or nii)

    Returns
    -------
    dataframe with adjusted b-values and diffusion directions
    """

    n_entries, _ = data.shape

    # copy the b-values to another column to save the original prescribed
    # b-value
    # this is not done for NIFTI data because we already have this columns
    if data_type == "dicom" or data_type == "pandas":
        data["b_value_original"] = data["b_value"]
    data["diffusion_direction_original"] = data["diffusion_direction"]

    data = estimate_rr_interval(data, settings)

    # ========================================================================
    # ========================================================================
    # Adjusting b-values for STEAM
    # ========================================================================
    # adjust b-values if STEAM sequence and DICOM or pandas data

    # ========================================================================
    # read the assumed RR interval and the calculated real b0 value
    # from the image comments header field
    if settings["sequence_type"] == "steam" and (data_type == "dicom" or data_type == "pandas"):
        if info["image_comments"]:
            logger.debug("Dicom header comment found: " + info["image_comments"])
            # get all numbers from comment field
            m = re.findall(r"[-+]?(?:\d*\.*\d+)", info["image_comments"])
            m = [float(m) for m in m]
            if len(m) > 2:
                logger.debug("Header comment field is corrupted!")
                assumed_rr_int = settings["assumed_rr_interval"]
                calculated_real_b0 = settings["calculated_real_b0"]
            if len(m) == 2:
                logger.debug("Both b0 and RR interval found in header.")
                calculated_real_b0 = m[0]
                assumed_rr_int = m[1]
            elif len(m) == 1:
                logger.debug("Only b0 found in header.")
                calculated_real_b0 = m[0]
                assumed_rr_int = settings["assumed_rr_interval"]
            else:
                # incomplete info --> hard code numbers with the most
                # likely values
                logger.debug("No dicom header comment found!")
                assumed_rr_int = settings["assumed_rr_interval"]
                calculated_real_b0 = settings["calculated_real_b0"]
        else:
            logger.debug("No dicom header comment found!")
            # no info --> hard code numbers with the most
            # likely values
            assumed_rr_int = settings["assumed_rr_interval"]
            calculated_real_b0 = settings["calculated_real_b0"]

        logger.debug("calculated_real_b0: " + str(calculated_real_b0))
        logger.debug("assumed_rr_interval: " + str(assumed_rr_int))

        # ========================================================================
        # adjust the b-values
        logger.debug("STEAM sequence and DICOM data: adjusting b-values")

        # first adjust the b0 value
        # if b-value == 0, then replace it with the calculated real b0 value
        # this small b-value is due to a small diffusion weighting given by the
        # spoiler gradients
        data["b_value"] = data["b_value"].apply(lambda x: calculated_real_b0 if x == 0 else x)

        # for in-vivo, also adjust all b-values relative to the assumed RR interval
        if not settings["ex_vivo"]:

            def adjust_b_values(val, nom_interval, ass_rr_int, est_rr_int):
                if nom_interval != 0.0 and nom_interval != "None":
                    return val * (nom_interval * 1e-3) / (ass_rr_int * 1e-3)
                else:
                    return val * (est_rr_int * 1e-3) / (ass_rr_int * 1e-3)

            data["b_value"] = data.apply(
                lambda x: adjust_b_values(
                    x["b_value"], x["nominal_interval"], assumed_rr_int, x["estimated_rr_interval"]
                ),
                axis=1,
            )

    else:
        assumed_rr_int = None
        calculated_real_b0 = None
        logger.debug("SE sequence or NIFTI data: not adjusting b-values")

    # ========================================================================
    # Adjusting diffusion directions to the image plane if DICOM or pandas
    # ========================================================================
    if data_type == "dicom" or data_type == "pandas":
        logger.debug("DICOM or Pandas data: rotating directions to the image plane.")

        # get the rotation matrix
        if info["manufacturer"] == "siemens" or info["manufacturer"] == "philips" or info["manufacturer"] == "uih":
            first_column = np.array(info["image_orientation_patient"][0:3])
            second_column = np.array(info["image_orientation_patient"][3:6])
            third_column = np.cross(first_column, second_column)
            rotation_matrix = np.stack((first_column, second_column, third_column), axis=-1)

        elif info["manufacturer"] == "ge":
            # for GE the directions are in the MR physics coordinates
            # (Frequency, Phase, Slice), so we need to rotate them
            # to the image plane
            # TODO this is not working 100% for GE data, sometimes directions need to be inverted.
            # This could be because we need more information from the header or there is a bug in the sequence
            # code. GE team investigating...
            rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        if settings["sequence_type"] == "steam":
            # For STEAM we need to tweak the direction in the b0 images.
            # we already adjusted b=0 to a small b-value given by the spoiler.
            # The spoiler diffusion direction is in the image plane,
            # so we need to set dir_in_image_plane to True for these cases.
            data.loc[(data["diffusion_direction"] == (0.0, 0.0, 0.0)), "dir_in_image_plane"] = True

            # for these same cases we need to change the direction to the normalised (1, 1, 1)
            data["diffusion_direction"] = data["diffusion_direction"].apply(
                lambda x: (
                    (
                        1.0 / math.sqrt(3),
                        1.0 / math.sqrt(3),
                        1.0 / math.sqrt(3),
                    )
                    if x == (0.0, 0.0, 0.0)
                    else x
                )
            )

        # For both SE and STEAM, rotate all directions where dir_in_image_plane is False
        def rotate_dir(val, in_plane_bool, rot_matrix):
            if not in_plane_bool:
                return list(np.matmul(val, rot_matrix))
            else:
                return val

        def rotate_bmatrix(val, in_plane_bool, rot_matrix):
            if not in_plane_bool:
                return np.dot(np.linalg.inv(rot_matrix), np.dot(val, rot_matrix))
            else:
                return val

        data["diffusion_direction"] = data.apply(
            lambda x: rotate_dir(x["diffusion_direction"], x["dir_in_image_plane"], rotation_matrix),
            axis=1,
        )

        if "bmatrix" in data.columns:
            data["bmatrix"] = data.apply(
                lambda x: rotate_bmatrix(x["bmatrix"], x["dir_in_image_plane"], rotation_matrix),
                axis=1,
            )

    # the DICOM standard for directions:
    # x positive is left to right
    # y positive top to bottom
    # so invert the Y direction in order to have the conventional cartesian
    # orientations from the start:
    # (x positive right to left, y positive bottom to top, z positive away from you)
    if data_type == "dicom" or data_type == "pandas":
        data["diffusion_direction"] = data["diffusion_direction"].apply(lambda x: np.multiply(x, [1, -1, 1]))

        if "bmatrix" in data.columns:
            rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            data["bmatrix"] = data.apply(
                lambda x: rotate_bmatrix(x["bmatrix"], x["dir_in_image_plane"], rot_mat),
                axis=1,
            )

    if settings["debug"]:
        plot_b_values_adjustment(data, settings)

    return data


def create_2d_montage_from_database(
    data: pd.DataFrame,
    b_value_column_name: str,
    direction_column_name: str,
    settings: dict,
    info: dict,
    slices: list,
    filename: str,
    save_path: str,
    list_to_highlight: list = [],
    segmentation: dict = {},
    print_series: bool = False,
    image_label: str = "image",
):
    """
    Create a grid with all DWIs for each slice

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    b_value_column_name : str
        string of the column to use as the b-value
    direction_column_name: str with direction
    settings : dict
        settings from YAML file
    info : dict
        useful info
    slices : list
        list of slices
    filename : str
        filename to save the montage
    save_path: str
    where to save the image
    list_to_highlight : list, optional
        list of indices to highlight, by default []
    segmentation: dict with segmentation information
    print_series: bool print series description switch
    image_label: "image" or "image_phase"

    """

    # loop over the slices
    for slice_int in slices:
        # dataframe with current slice
        c_df = data.loc[data["slice_integer"] == slice_int].copy()

        # initiate maximum number of images found for each b-val and dir combination
        max_number_of_images = 0

        # convert list of directions to a tuple
        c_df[direction_column_name].apply(tuple)

        # get unique b-values
        b_vals = c_df[b_value_column_name].unique()
        b_vals.sort()

        # initiate the stacks for the images and the highlight masks
        c_img_stack = {}
        c_img_stack_series_description = {}
        c_highlight_stack = {}

        # loop over sorted b-values
        for b_val in b_vals:
            # dataframe with current slice and current b-value
            c_df_b = c_df.loc[c_df[b_value_column_name] == b_val].copy()

            # now loop over directions
            c_df_b[direction_column_name] = c_df_b[direction_column_name].apply(tuple)

            # unique directions
            dirs = c_df_b[direction_column_name].unique()
            dirs.sort()

            # loop over directions
            for dir_idx, dir in enumerate(dirs):
                # dataframe with current slice, current b-value, and current direction images
                c_df_b_d = c_df_b.loc[c_df_b[direction_column_name] == dir].copy()

                # for each b_val and each dir collect all images
                c_img_stack[int(b_val), dir_idx] = np.stack(c_df_b_d[image_label].values, axis=0)
                c_img_stack_series_description[int(b_val), dir_idx] = c_df_b_d.series_description.values

                # create a mask with 0s and 1s to highlight images in certain positions of the dataframe
                # these mask will be of the same shape as the images stack
                c_indices = c_df_b_d.index.to_numpy()
                mask = np.isin(c_indices, list_to_highlight)
                mask = mask[..., np.newaxis, np.newaxis]
                mask = np.repeat(mask, c_img_stack[int(b_val), dir_idx].shape[1], axis=1)
                mask = np.repeat(mask, c_img_stack[int(b_val), dir_idx].shape[2], axis=2)
                c_highlight_stack[int(b_val), dir_idx] = mask

                # record n_images if bigger than the values stored
                n_images = c_df_b_d.shape[0]
                if n_images > max_number_of_images:
                    max_number_of_images = n_images

        # now we have 2 dictionaries:
        # with all the images for each b_val and dir combination
        # and the respective masks to highlight certain images

        # so now we need to create 2 montages with all stacks [len(c_img_stack), max_number_of_images]
        # initiate montages
        montage = np.zeros((len(c_img_stack) * info["img_size"][0], max_number_of_images * info["img_size"][1]))
        montage_mask = np.zeros((len(c_img_stack) * info["img_size"][0], max_number_of_images * info["img_size"][1]))
        # loop over the dictionaries and add to montages
        for idx, key in enumerate(c_img_stack):
            cc_img_stack = c_img_stack[key]
            cc_img_stack = cc_img_stack.transpose(1, 2, 0)
            cc_img_stack = np.reshape(
                cc_img_stack, (cc_img_stack.shape[0], cc_img_stack.shape[1] * cc_img_stack.shape[2]), order="F"
            )
            montage[idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_img_stack.shape[1]] = (
                cc_img_stack
            )

            if list_to_highlight:
                # repeat for mask
                cc_mask_stack = c_highlight_stack[key]
                cc_mask_stack = cc_mask_stack.transpose(1, 2, 0)
                cc_mask_stack = np.reshape(
                    cc_mask_stack, (cc_mask_stack.shape[0], cc_mask_stack.shape[1] * cc_mask_stack.shape[2]), order="F"
                )
                montage_mask[idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_mask_stack.shape[1]] = (
                    cc_mask_stack
                )

        # create RGB image of montage and increase brightness
        if image_label == "image":
            montage = montage / np.max(montage)
            montage = 2 * montage
            montage[montage > 1] = 1
        elif image_label == "image_phase":
            montage = montage + np.pi
            montage = montage / np.max(montage)
            montage[montage > 1] = 1
            montage[montage < 0] = 0
        # add RGB channels
        montage = np.stack([montage, montage, montage], axis=-1)

        # we need to add the segmentation, therefore the montages need to be with colour
        if segmentation:
            # create montage with segmentation
            seg_img = np.zeros((info["img_size"][0], info["img_size"][1], 3))
            pts = np.array(segmentation[slice_int]["epicardium"], dtype=int)
            seg_img[pts[:, 1], pts[:, 0]] = [1.0, 1.0, 0.33]
            if segmentation[slice_int]["endocardium"].size != 0:
                pts = np.array(segmentation[slice_int]["endocardium"], dtype=int)
                seg_img[pts[:, 1], pts[:, 0]] = [1.0, 1.0, 0.33]
            # repeat image for the entire stack
            seg_img = np.tile(seg_img, (len(c_img_stack), max_number_of_images, 1))

            # create a transparency mask as a 4th channel
            seg_mask = seg_img[:, :, :] == [0, 0, 0]
            seg_mask = seg_mask.all(axis=2)
            seg_mask = np.invert(seg_mask) * 0.5
            seg_img = np.dstack([seg_img, seg_mask])

        if list_to_highlight:
            # repeat slightly different for the rejected images mask
            montage_mask = montage_mask / np.max(montage_mask)
            montage_mask = np.stack([montage_mask, 0 * montage_mask, 0 * montage_mask], axis=-1)
            montage_mask = np.dstack([montage_mask, montage_mask[:, :, 0] * 0.2])

        # check if fov is vertical, if so text needs to be rotated
        if info["img_size"][0] < info["img_size"][1]:
            text_rotation = 0
        else:
            text_rotation = 90
        # save montages in a figure
        fig = plt.figure(figsize=(len(c_img_stack), max_number_of_images))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(montage)
        if settings["print_series_description"]:
            if print_series:
                for idx, key in enumerate(c_img_stack_series_description):
                    for iidx, label in enumerate(c_img_stack_series_description[key]):
                        x_pos = 5 + iidx * info["img_size"][1]
                        y_pos = 10 + idx * info["img_size"][0]
                        plt.text(
                            x_pos,
                            y_pos,
                            label,
                            fontsize=3,
                            color="tab:orange",
                            horizontalalignment="left",
                            verticalalignment="top",
                            bbox=dict(facecolor="black", pad=0, edgecolor="none"),
                            rotation=text_rotation,
                        )
        if segmentation:
            plt.imshow(seg_img)
        if list_to_highlight:
            plt.imshow(montage_mask)
        ax.set_yticks(
            range(
                round(info["img_size"][0] * 0.5),
                round(info["img_size"][0] * len(c_img_stack)),
                round(info["img_size"][0]),
            )
        )
        lbls = list(c_img_stack.keys())
        lbls = [str(i) for i in lbls]
        ax.set_yticklabels(lbls)
        ax.set_xticks([])
        plt.savefig(
            os.path.join(
                save_path,
                filename + "_slice_" + str(slice_int).zfill(2) + ".png",
            ),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def reorder_by_slice(
    data: pd.DataFrame, settings: dict, info: dict, logger: logging
) -> [pd.DataFrame, dict, list, int]:
    """
    Reorder data by slice and remove slices if needed

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    settings : dict
        settings from YAML file
    info : dict
        useful info
    logger : logging
        logger for console

    Returns
    -------
    pd.DataFrame
        dataframe with reordered data
    dict
        info about the data
    list
        list of slices
    int
        number of slices

    """
    # round the image position values (sometimes there are small differences)
    # round image position to the first significant digit decimal place in the slice thickness
    try:
        slice_thickness = float(info["slice_thickness"])
        decimal_place = abs(int(math.log10(abs(slice_thickness)))) + 1
        data["image_position"] = data["image_position"].apply(lambda x: tuple(np.round(x, decimal_place)))
    except KeyError:
        # No slice thickness in the header
        data["image_position"] = data["image_position"].apply(lambda x: tuple(x))

    # determine if we can use z, or y or x to sort the slices
    unique_positions = np.array(list(data.image_position.unique()))
    n_positions = len(unique_positions)
    n_positions_x = len(np.unique(unique_positions[:, 0]))
    n_positions_y = len(np.unique(unique_positions[:, 1]))
    n_positions_z = len(np.unique(unique_positions[:, 2]))
    if n_positions_z == n_positions:
        image_position_label_idx = 2
    elif n_positions_y == n_positions:
        image_position_label_idx = 1
    elif n_positions_x == n_positions:
        image_position_label_idx = 0

    # sort the slices by the new column
    image_position_label = np.array(list(data.image_position.values))[:, image_position_label_idx]
    data["image_position_label"] = pd.Series(image_position_label)
    data = data.sort_values(["image_position_label", "acquisition_date_time"], ascending=[False, True])
    data = data.reset_index(drop=True)

    # how many slices do we have and encode them with an integer
    slices = data.image_position.unique()
    n_slices = len(slices)

    # create dictionaries to go from image position to integer and vice versa
    info["integer_to_image_positions"] = {}
    for idx, slice in enumerate(slices):
        info["integer_to_image_positions"][idx] = slice
    info["image_positions_to_integer"] = dict((v, k) for k, v in info["integer_to_image_positions"].items())
    # create a new column in the data table with the slice integer for that slice
    list_of_tuples = data.image_position.values
    slice_integer = [info["image_positions_to_integer"][i] for i in list_of_tuples]
    data["slice_integer"] = pd.Series(slice_integer)

    # Do we need to remove slices?
    if settings["remove_slices"]:
        logger.debug("Original number of slices: " + str(n_slices))

    slices_to_remove = settings["remove_slices"]
    logger.debug("Removing slices: " + str(slices_to_remove))
    for slice_idx in slices_to_remove:
        data = data[data.slice_integer != slice_idx]
        data = data.reset_index(drop=True)

    # slices is going to be a list of all the integers
    slices = data.slice_integer.unique()
    # n_slices = len(slices)
    # leave this with the original number of slices
    info["n_slices"] = n_slices

    # calculate distances between slices
    if n_slices > 1:
        # To sort the positions we assume that the positions follow a line in 3D space. We calculate the distance to the
        # middle point of the array of positions and sort the positions by this signed distance. The sign of the distance
        # is given by the dot product of the normal to the plane defined by the first and last position and the vector.
        # This plane is perpendicular to the line defined by the positions.

        def sign(x):
            return 2 * (x >= 0) - 1

        normal = unique_positions[0] - unique_positions[-1]
        mid_pos_idx = len(unique_positions) // 2
        d = unique_positions[mid_pos_idx]
        distance = sign(np.dot(unique_positions - d, normal)) * np.linalg.norm(unique_positions - d, axis=1)
        unique_positions = unique_positions[np.argsort(distance), :]
        spacing_z = np.linalg.norm(unique_positions[1:] - unique_positions[:-1], axis=1)
        slice_spacing = np.mean(spacing_z)
        info["slice_spacing"] = slice_spacing
    else:
        info["slice_spacing"] = 0

    return data, info, slices, n_slices


def read_data(settings: dict, info: dict, logger: logging) -> tuple[pd.DataFrame, dict, NDArray]:
    """

    Read DTCMR data

    Parameters
    ----------
    settings : dict
        settings from YAML file
    info : dict
        useful info
    logger : logging
        logger for console

    Returns
    -------
    pd.DataFrame
        dataframe with all the data
    info
        useful info
    slices
        array with slices

    """

    # initiate variables
    data_type = None
    settings["complex_data"] = False

    # gather possible dicoms, or nii file paths into a list
    data_type, list_dicoms, list_dicoms_phase, list_nii = list_files(data_type, logger, settings)

    # If DICOMs are present, then we will read them and build our diffusion database.
    # If DICOMs are not present but NIFTI files are, then we will read them and build our diffusion database.
    # If neither DICOMs nor NIFTI files are present, then we will read the pre-saved database
    if data_type == "dicom":
        data, data_phase, info = read_and_process_dicoms(info, list_dicoms, list_dicoms_phase, logger, settings)

    elif data_type == "nii":
        # read nii files
        data, info = read_and_process_niis(list_nii, settings, info, logger)

    else:
        # read pandas
        data_type = "pandas"
        data, data_phase, info = read_and_process_pandas(logger, settings)

    # now that we loaded the images and headers we need to organise it as
    # we cannot assume that the files are in any particular order
    # sort the dataframe by date and time, this is needed in case we need to adjust
    # the b-values by the DICOM timings
    data = sort_by_date_time(data)
    if settings["complex_data"]:
        data_phase = sort_by_date_time(data_phase)

        # copy the image column to the data table
        data["image_phase"] = data_phase["image"]
        # discard the data_phase dataframe
        del data_phase

        # translate phase magnitude values to angles in radians
        data["image_phase"] = np.pi * (data["image_phase"]) / 4096

    # =========================================================
    # adjust b-values and diffusion directions to image
    # =========================================================
    data = adjust_b_val_and_dir(data, settings, info, logger, data_type)

    # =========================================================
    # set b-values to be removed if list not empty in settings
    # =========================================================
    if "to_be_removed" not in data:
        data["to_be_removed"] = False
    if settings["remove_b_values"]:
        for bval in settings["remove_b_values"]:
            data.loc[data.b_value_original == bval, "to_be_removed"] = True

    # Add B-matrix column if not present in siemens data
    if "bmatrix" not in data.columns:
        data["bmatrix"] = None

    # =========================================================
    # re-order data again, this time by slice first, then by date-time
    # also potentially remove slices
    # =========================================================
    data, info, slices, n_slices = reorder_by_slice(data, settings, info, logger)

    # number of dicom files
    info["n_images"] = data.shape[0]
    # image size
    if data_type == "nii":
        info["img_size"] = data.image[0].shape
    else:
        info["img_size"] = (info["Rows"], info["Columns"])

    data_summary_plots_and_logs(data, settings, info, logger)

    # =========================================================
    # display all DWIs in a montage
    # =========================================================
    if settings["debug"]:
        create_2d_montage_from_database(
            data,
            "b_value_original",
            "diffusion_direction_original",
            settings,
            info,
            slices,
            "dwis_original_dicoms",
            settings["debug_folder"],
            [],
            {},
            True,
            "image",
        )
        if settings["complex_data"]:
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "diffusion_direction_original",
                settings,
                info,
                slices,
                "dwis_original_dicoms_phase",
                settings["debug_folder"],
                [],
                {},
                True,
                "image_phase",
            )

    # also save some diffusion info to a csv file
    save_path = os.path.join(settings["dicom_folder"], "diff_info_sorted.csv")

    # drop the image and image_phase columns from data and export to csv
    data_without_imgs = data.drop(columns=["image"])
    if settings["complex_data"]:
        data_without_imgs = data_without_imgs.drop(columns=["image_phase"])

    data_without_imgs.to_csv(save_path)
    del data_without_imgs

    # plot the b-values (before and after adjustment)
    if settings["debug"] and settings["sequence_type"] == "steam":
        plot_b_values_adjustment(data, settings)

    return data, info, slices


def read_and_process_pandas(logger: logging, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Read pandas dataframe

    Parameters
    ----------
    logger
    settings

    Returns
    -------
    dataframe with diffusion info
    dataframe with phase diffusion info (if any)
    info dict

    """
    logger.debug("No DICOM or Nii files found. Checking for diffusion database files previously created.")
    # check if subfolders "mag" and "phase" exist
    mag_folder = os.path.join(settings["dicom_folder"], "mag")
    phase_folder = os.path.join(settings["dicom_folder"], "phase")
    if os.path.exists(mag_folder):
        settings["complex_data"] = True
        settings["dicom_folder"] = mag_folder
        if os.path.exists(phase_folder):
            settings["dicom_folder_phase"] = phase_folder
        else:
            raise FileNotFoundError("No phase data folder found!")

        logger.debug("Magnitude and phase folders found.")
        logger.debug("Complex averaging on.")
    # =========================================================
    # read the dataframe
    # =========================================================
    save_path = os.path.join(settings["dicom_folder"], "data.gz")
    # check if files exist
    if not os.path.exists(save_path):
        logger.error("No magnitude database files found!")
        raise FileNotFoundError("No magnitude database files found!")
    data = pd.read_pickle(save_path)
    info = data.attrs["info"]
    if settings["complex_data"]:
        save_path = os.path.join(settings["dicom_folder_phase"], "data.gz")
        # check if files exist
        if not os.path.exists(save_path):
            logger.error("No phase database files found!")
            raise FileNotFoundError("No phase database files found!")
        data_phase = pd.read_pickle(save_path)
    else:
        data_phase = pd.DataFrame()
    # =========================================================
    # read the pixel arrays from the h5 file and add them to the dataframe
    # =========================================================
    save_path = os.path.join(settings["dicom_folder"], "images.h5")
    with h5py.File(save_path, "r") as hf:
        pixel_values = hf["pixel_values"][:]
    assert (
        len(pixel_values) == data.shape[0]
    ), "Number of pixel slices does not match the number of entries in the dataframe."
    data["image"] = pd.Series([x for x in pixel_values])
    if settings["complex_data"]:
        save_path = os.path.join(settings["dicom_folder_phase"], "images.h5")
        with h5py.File(save_path, "r") as hf:
            pixel_values_phase = hf["pixel_values"][:]
        assert (
            len(pixel_values_phase) == data_phase.shape[0]
        ), "Number of pixel slices does not match the number of entries in the dataframe."
        data_phase["image"] = pd.Series([x for x in pixel_values_phase])

    return data, data_phase, info


def read_and_process_dicoms(
    info: dict, list_dicoms: list, list_dicoms_phase: list, logger: logging, settings: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Read DICOM files into pandas
    Export the dataframe to a zip file
    Export diffusion info to css
    Export pixel values to HDF5
    Potentially archive DICOMs

    Parameters
    ----------
    info
    list_dicoms
    list_dicoms_phase
    logger
    settings

    Returns
    -------
    dataframe with diffusion info
    dataframe with phase diffusion info (if any)
    info dict

    """

    # create empty dataframe
    data_phase = pd.DataFrame()

    # read DICOM info
    data, info["manufacturer"] = get_data_from_dicoms(list_dicoms, settings, logger, image_type="mag")
    # check some global info
    info, data = check_global_info(data, info, logger)
    # adjust pixel values to the correct scale
    data = scale_dicom_pixel_values(data)
    # check rows and columns
    info = check_rows_and_columns(data, info, logger)
    # interpolate images if img_interp_factor > 1
    if "img_interp_factor" in settings and settings["img_interp_factor"] > 1:
        data, info = interpolate_dicom_pixel_values(
            data, info, logger, image_type="mag", factor=settings["img_interp_factor"]
        )

    # replace the nan directions with (0.0, 0.0, 0.0)
    data = tweak_directions_and_b_values(data)
    # add some columns if not in table
    data = add_missing_columns(data)

    if settings["complex_data"]:
        info_phase = {}
        data_phase, _ = get_data_from_dicoms(list_dicoms_phase, settings, logger, image_type="phase")
        # check some global info
        info_phase, data_phase = check_global_info(data_phase, info_phase, logger)
        # adjust pixel values to the correct scale
        data_phase = scale_dicom_pixel_values(data_phase)
        # check rows and columns
        info_phase = check_rows_and_columns(data_phase, info_phase, logger)
        # interpolate images if img_interp_factor > 1
        if "img_interp_factor" in settings and settings["img_interp_factor"] > 1:
            data_phase, info_phase = interpolate_dicom_pixel_values(
                data_phase, info_phase, logger, image_type="phase", factor=settings["img_interp_factor"]
            )
        data_phase = tweak_directions_and_b_values(data_phase)

        # check if the magnitude and phase tables match
        data_sort = sort_by_date_time(data)
        data_phase_sort = sort_by_date_time(data_phase)
        columns_to_keep = [
            "b_value",
            "diffusion_direction",
            "image_position",
            "acquisition_date_time",
            "image_orientation_patient",
            "Rows",
            "Columns",
        ]

        data_sort = data_sort[columns_to_keep]
        data_phase_sort = data_phase_sort[columns_to_keep]

        if not data_sort.equals(data_phase_sort):
            logger.error("Magnitude and phase DICOM tables do not match!")
            raise ValueError("Magnitude and phase DICOM tables do not match!")
        del data_phase_sort
        del data_sort

    # =========================================================
    # export the dataframe to a zip file
    # =========================================================
    # save the dataframe and the info dict
    data.attrs["info"] = info
    save_path = os.path.join(settings["dicom_folder"], "data.gz")
    data_without_imgs = data.drop(columns=["image"])
    data_without_imgs.to_pickle(save_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    if settings["complex_data"]:
        data_phase.attrs["info"] = info
        save_path = os.path.join(settings["dicom_folder_phase"], "data.gz")
        data_without_imgs_phase = data_phase.drop(columns=["image"])
        data_without_imgs_phase.to_pickle(save_path, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})

    # also save some diffusion info to a csv file
    save_path = os.path.join(settings["dicom_folder"], "diff_info_dataframe_h5.csv")
    columns_list = [
        "fiji_index",
        "file_name",
        "b_value",
        "diffusion_direction",
        "dir_in_image_plane",
        "image_position",
        "nominal_interval",
        "series_description",
        "series_number",
        "acquisition_date_time",
    ]

    # remove elements that are not in the columns list
    columns_list = [col for col in columns_list if col in data.columns]

    data.to_csv(
        save_path,
        columns=columns_list,
        index=False,
    )
    if settings["complex_data"]:
        save_path = os.path.join(settings["dicom_folder_phase"], "diff_info_dataframe_h5.csv")
        columns_list = [col for col in columns_list if col in data_phase.columns]
        data_phase.to_csv(
            save_path,
            columns=columns_list,
            index=False,
        )

    # finally save the pixel values to HDF5
    image_pixel_values = data["image"].to_numpy()
    image_pixel_values = np.stack(image_pixel_values)
    save_path = os.path.join(settings["dicom_folder"], "images.h5")
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("pixel_values", data=image_pixel_values, compression="gzip", compression_opts=1)
    if settings["complex_data"]:
        image_pixel_values_phase = data_phase["image"].to_numpy()
        image_pixel_values_phase = np.stack(image_pixel_values_phase)
        save_path = os.path.join(settings["dicom_folder_phase"], "images.h5")
        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("pixel_values", data=image_pixel_values_phase, compression="gzip", compression_opts=1)

    # if workflow is anon, we are going to archive the DICOMs
    # in a 7z file with the password set in the .env file.
    if settings["workflow_mode"] == "anon":
        # =========================================================
        # Archive DICOM files in a 7z file
        # =========================================================
        # create folder to store DICOMs
        dicom_archive_folder = os.path.join(settings["dicom_folder"], "dicom_archive")
        if not os.path.exists(dicom_archive_folder):
            os.makedirs(dicom_archive_folder)

        # move all DICOMs to the archive folder
        def move_file(file):
            file_path = os.path.join(settings["dicom_folder"], file)
            shutil.move(file_path, os.path.join(dicom_archive_folder, file))

        # get all dicom filenames
        dicom_list = data["file_name"].tolist()
        # remove any duplicates (happens with multiframe DICOMs)
        dicom_list = list(dict.fromkeys(dicom_list))
        [move_file(item) for item in dicom_list]

        # load password from .env file
        env_vars = dotenv_values(os.path.join(os.getcwd(), ".env"))

        # now encrypt folder with 7zip
        with py7zr.SevenZipFile(
            os.path.join(settings["dicom_folder"], "dicom_archive.7z"), "w", password=env_vars["ARCHIVE_PASS"]
        ) as archive:
            archive.writeall(dicom_archive_folder, "dicom_archive")

        # DELETE FOLDER WITH DICOMS!
        shutil.rmtree(dicom_archive_folder)

        if settings["complex_data"]:
            # create folder to store DICOMs
            dicom_archive_folder = os.path.join(settings["dicom_folder_phase"], "dicom_archive")
            if not os.path.exists(dicom_archive_folder):
                os.makedirs(dicom_archive_folder)

            # move all DICOMs to the archive folder
            def move_file(file):
                file_path = os.path.join(settings["dicom_folder_phase"], file)
                shutil.move(file_path, os.path.join(dicom_archive_folder, file))

            # get all dicom filenames
            dicom_list = data_phase["file_name"].tolist()
            # remove any duplicates (happens with multiframe DICOMs)
            dicom_list = list(dict.fromkeys(dicom_list))
            [move_file(item) for item in dicom_list]

            # now encrypt folder with 7zip
            with py7zr.SevenZipFile(
                os.path.join(settings["dicom_folder_phase"], "dicom_archive.7z"),
                "w",
                password=env_vars["ARCHIVE_PASS"],
            ) as archive:
                archive.writeall(dicom_archive_folder, "dicom_archive")

            # DELETE FOLDER WITH DICOMS!
            shutil.rmtree(dicom_archive_folder)

    return data, data_phase, info


def list_files(data_type: str, logger: logging, settings: dict) -> [str, list, list, list]:
    """
    List possible magnitude DICOMs, phase DICOMs, and NII files

    Parameters
    ----------
    data_type: type of data (dicoms, nii or none)
    logger
    settings

    Returns
    -------
    data_type: (dicoms, nii or none)
    list of dicoms
    list of phase dicoms
    list of nii files

    """

    list_dicoms = []
    list_dicoms_phase = []
    list_nii = []

    # first check if the subfolders and files inside diffusion_images make sense
    if not os.path.exists(settings["dicom_folder"]):
        logger.error("Data folder does not exist: " + settings["dicom_folder"])
        raise FileNotFoundError("Data folder does not exist: " + settings["dicom_folder"])

    # look for files either dcm, nii or data.gz
    included_extensions = ["dcm", "DCM", "IMA"]
    list_dicoms = [
        fn for fn in os.listdir(settings["dicom_folder"]) if any(fn.endswith(ext) for ext in included_extensions)
    ]
    included_extensions = ["nii", "nii.gz"]
    list_nii = [
        fn for fn in os.listdir(settings["dicom_folder"]) if any(fn.endswith(ext) for ext in included_extensions)
    ]
    list_data_gz = glob.glob(os.path.join(settings["dicom_folder"], "data.gz"))

    list_dicoms_mag = []
    list_dicoms_phase = []
    if len(list_data_gz) < 1 and len(list_dicoms) < 1 and len(list_nii) < 1:

        # we havent found any suitable files in the "diffusion_images" folder. So the next step is to check if there is data in subfolders called "mag" and "phase", i.e complex data.
        mag_folder = os.path.join(settings["dicom_folder"], "mag")
        phase_folder = os.path.join(settings["dicom_folder"], "phase")
        if not os.path.exists(mag_folder) or not os.path.exists(phase_folder):
            logger.error(
                "No data found in 'diffusion_images' or 'mag' and 'phase' folders. Please check the folder and try again."
            )
            raise FileNotFoundError(
                "No data found in 'diffusion_images' or 'mag' and 'phase' folders. Please check the folder and try again."
            )
        # look for DICOM files
        included_extensions = ["dcm", "DCM", "IMA"]
        list_dicoms_mag = [fn for fn in os.listdir(mag_folder) if any(fn.endswith(ext) for ext in included_extensions)]
        list_dicoms_phase = [
            fn for fn in os.listdir(phase_folder) if any(fn.endswith(ext) for ext in included_extensions)
        ]
        # look for nii files
        included_extensions = ["nii", "nii.gz"]
        list_nii_mag = [fn for fn in os.listdir(mag_folder) if any(fn.endswith(ext) for ext in included_extensions)]
        list_nii_phase = [
            fn for fn in os.listdir(phase_folder) if any(fn.endswith(ext) for ext in included_extensions)
        ]
        # look for data.gz files
        list_data_gz_mag = glob.glob(os.path.join(mag_folder, "data.gz"))
        list_data_gz_phase = glob.glob(os.path.join(phase_folder, "data.gz"))

        if len(list_data_gz_mag) < 1 and len(list_dicoms_mag) < 1 and len(list_nii_mag) < 1:
            logger.error(
                "No DICOM files, NIFTI files or pre-saved database found in the folder: "
                + settings["dicom_folder"]
                + ". Please check the folder and try again."
            )
            raise FileNotFoundError(
                "No DICOM files, NIFTI files or pre-saved database found in the folder: "
                + settings["dicom_folder"]
                + ". Please check the folder and try again."
            )
        elif (
            len(list_data_gz_mag) != len(list_data_gz_phase)
            or len(list_dicoms_mag) != len(list_dicoms_phase)
            or len(list_nii_mag) != len(list_nii_phase)
        ):
            logger.error(
                "Number of DICOM files, NIFTI files or pre-saved database in the mag and phase folders are different. "
                "Please check the folders and try again."
            )
            raise FileNotFoundError(
                "Number of DICOM files, NIFTI files or pre-saved database in the mag and phase folders are different. "
                "Please check the folders and try again."
            )

    # Check for DICOM files
    if len(list_dicoms) > 0:
        data_type = "dicom"
        logger.debug("DICOM files found.")
        logger.debug("Magnitude data only.")
        list_dicoms.sort()

    else:
        if len(list_dicoms_mag) > 0 and len(list_dicoms_phase) > 0:
            data_type = "dicom"
            settings["complex_data"] = True
            settings["dicom_folder"] = mag_folder
            settings["dicom_folder_phase"] = phase_folder
            logger.debug("Magnitude and phase DICOM files found.")
            logger.debug("Complex averaging on.")
            list_dicoms = list_dicoms_mag
            list_dicoms.sort()
            list_dicoms_phase.sort()
        else:
            logger.debug("No DICOM files found.")
            data_type = None
    if not data_type:
        # If no DICOMS, check for nii files
        if len(list_nii) > 0:
            data_type = "nii"
            logger.debug("NIFTI files found.")

    return data_type, list_dicoms, list_dicoms_phase, list_nii
