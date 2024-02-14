import logging
import math
import os
import pickle
import re
from datetime import datetime
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import yaml
from numpy.typing import NDArray


def data_summary_plots(data: pd.DataFrame, info: dict, settings: dict):
    """
    Summarise the data in histogram counts by b-value, direction, and slice
    Parameters
    ----------
    data: dataframe with all the dwi data
    info: dictionary with useful info
    settings: dictionary with useful info
    """

    # get directions order as a numeric vector
    data.direction = data.direction.apply(lambda x: tuple(x) if type(x) != str else tuple([x]))
    unique_dirs = data["direction"].unique().tolist()
    idxs = [item for item in range(0, len(unique_dirs) + 1)]
    dir_keys = {unique_dirs[i]: idxs[i] for i in range(len(unique_dirs))}

    direction_list = data["direction"].tolist()
    direction_idxs = [dir_keys[direction_list[i]] for i in range(len(direction_list))]

    # save bar plots in a montage
    plt.figure(figsize=(5, 5))

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
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def get_data(settings: dict, info: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Read all the DICOM files in data_folder_path and store important info
    in a dataframe and some header info in a dictionary

    Parameters
    ----------
    settings: dict
    info: dict

    Returns
    -------
    df: dataframe with the DICOM diffusion information
    info: dictionary with useful info
    """
    data_folder_path = settings["dicom_folder"]

    # list DICOM files
    included_extensions = ["dcm", "DCM", "IMA"]
    list_dicoms = [fn for fn in os.listdir(data_folder_path) if any(fn.endswith(ext) for ext in included_extensions)]
    list_dicoms.sort()

    # collect some header info in a dictionary from the first DICOM
    ds = pydicom.dcmread(open(os.path.join(data_folder_path, list_dicoms[0]), "rb"))
    # get DICOM header fields from yaml file
    yaml_file = os.path.join(settings["code_path"], "extensions", "dicom_header_collect.yaml")
    with open(yaml_file) as f:
        dicom_header_fields = yaml.load(f.read(), Loader=yaml.Loader)

    header_info = {}
    # image comments
    header_info["image_comments"] = (
        ds[dicom_header_fields["DICOM_header_classic"]["image_comments"]]._value
        if dicom_header_fields["DICOM_header_classic"]["image_comments"] in ds
        else None
    )

    # image orientation patient
    temp_val = ds[dicom_header_fields["DICOM_header_classic"]["image_orientation_patient"]].value
    header_info["image_orientation_patient"] = [float(i) for i in temp_val]

    # pixel spacing
    temp_val = ds[dicom_header_fields["DICOM_header_classic"]["pixel_spacing"]].value
    header_info["pixel_spacing"] = [float(i) for i in temp_val]

    # create a dataframe with all DICOM values
    df = []

    for idx, file_name in enumerate(list_dicoms):
        # read DICOM
        ds = pydicom.dcmread(open(os.path.join(data_folder_path, file_name), "rb"))
        # loop over the dictionary of header fields and collect them for this DICOM file
        c_dicom_header = {}
        for key, value in dicom_header_fields["DICOM_header_classic"].items():
            if dicom_header_fields["DICOM_header_classic"][key] in ds:
                c_dicom_header[key] = ds[dicom_header_fields["DICOM_header_classic"][key]]

        # append values (will be a row in the dataframe)
        df.append(
            (
                # file name
                file_name,
                # array of pixel values
                ds.pixel_array,
                # b-value or zero if not a field
                (
                    ds[dicom_header_fields["DICOM_header_classic"]["b_value"]]._value
                    if dicom_header_fields["DICOM_header_classic"]["b_value"] in ds
                    else 0
                ),
                # diffusion directions, or [1, 1, 1] normalised if not a field
                (
                    ds[dicom_header_fields["DICOM_header_classic"]["diffusiongradientdirection"]]._value
                    if dicom_header_fields["DICOM_header_classic"]["diffusiongradientdirection"] in ds
                    else [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)]
                ),
                # image position
                tuple(ds[dicom_header_fields["DICOM_header_classic"]["image_position_patient"]].value),
                # nominal interval
                float(ds["NominalInterval"]._value),
                # acquisition time
                ds[dicom_header_fields["DICOM_header_classic"]["acquisition_time"]].value,
                # acquisition date
                ds[dicom_header_fields["DICOM_header_classic"]["acquisition_date"]].value,
                # False if diffusion direction is a field
                False if dicom_header_fields["DICOM_header_classic"]["diffusiongradientdirection"] in ds else True,
                # dictionary with header fields
                c_dicom_header,
            )
        )
    df = pd.DataFrame(
        df,
        columns=[
            "file_name",
            "image",
            "b_value",
            "direction",
            "image_position",
            "nominal_interval",
            "acquisition_time",
            "acquisition_date",
            "dir_in_image_plane",
            "header",
        ],
    )
    df = df.sort_values("file_name")
    df = df.reset_index(drop=True)

    # merge dictionaries into info
    info = {**info, **header_info}

    return df, info


def estimate_rr_interval(data: pd.DataFrame) -> [pd.DataFrame, NDArray]:
    """
    This function will estimate the RR interval from the DICOM header
    and add it to the dataframe

    # if no nominal interval values are in the headers, then we will adjust
    # the b-values according to the RR interval by getting the time delta between images
    # convert time strings to microseconds

    Parameters
    ----------
    data: dataframe with diffusion database

    Returns:
    dataframe with added estimated RR interval column
    estimated_rr_intervals_original (before adjustment, only for debug)
    """

    time_stamps = data["acquisition_time"].values.tolist()
    time_stamps = [datetime.strptime(i, "%H%M%S.%f") for i in time_stamps]
    time_stamps = [i.second * 1e6 + i.minute * 60 * 1e6 + i.hour * 3600 * 1e6 + i.microsecond for i in time_stamps]
    # get half the time delta between images
    time_delta = np.diff(time_stamps) * 0.5 * 1e-6
    # prepend nan to the time delta
    time_delta = np.insert(time_delta, 0, np.nan)
    # get median time delta, and replace values above 4x the median with nan
    median_time = np.nanmedian(time_delta)
    time_delta[time_delta > 4 * median_time] = np.nan
    # convert to ms
    time_delta = time_delta * 1e3
    # add time delta to the dataframe
    data["estimated_rr_interval"] = time_delta
    # replace nans with the next non-nan value
    data["estimated_rr_interval"] = data["estimated_rr_interval"].bfill()

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
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(data["nominal_interval"], alpha=0.8)
    plt.plot(data["estimated_rr_interval"], alpha=0.8)
    plt.legend(["nominal", "adjusted RR"])
    plt.xlabel("image #")
    plt.ylabel("nominal intervals")
    plt.ylim([-0.25, 4])
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.title("nominal intervals", fontsize=7)
    plt.tight_layout(pad=1.0)
    plt.savefig(
        os.path.join(
            settings["debug_folder"],
            "nominal_and_rr_intervals.png",
        ),
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


def adjust_b_val_and_dir(
    data: pd.DataFrame,
    settings: dict,
    info: dict,
    logger: logging.Logger,
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

    Returns
    -------
    dataframe with adjusted b-values and diffusion directions
    """

    n_entries, _ = data.shape

    # copy the b-values to another column to save the original prescribed
    # b-value
    data["b_value_original"] = data["b_value"]
    data["direction_original"] = data["direction"]

    data = estimate_rr_interval(data)

    # get the b0 value and assumed RR interval from the image comment field
    if settings["sequence_type"] == "steam":
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

    else:
        assumed_rr_int = None
        calculated_real_b0 = None
        logger.debug("SE sequence, so not adjusting b-values or RR interval")

    # get the rotation matrix
    first_column = np.array(info["image_orientation_patient"][0:3])
    second_column = np.array(info["image_orientation_patient"][3:6])
    third_column = np.cross(first_column, second_column)
    rotation_matrix = np.stack((first_column, second_column, third_column), axis=-1)

    # loop through the entries and adjust b-values and directions
    for idx in range(n_entries):
        if settings["sequence_type"] == "steam":
            c_b_value = data.loc[idx, "b_value"]

            # replace b0 value
            if c_b_value == 0:
                c_b_value = calculated_real_b0

            # correct b_value relative to the assumed RR interval with the nominal interval if not 0.0.
            # Otherwise use the estimated RR interval.
            c_nominal_interval = data.loc[idx, "nominal_interval"]
            c_estimated_rr_interval = data.loc[idx, "estimated_rr_interval"]
            if c_nominal_interval != 0.0:
                c_b_value = c_b_value * (c_nominal_interval * 1e-3) / (assumed_rr_int * 1e-3)
            else:
                c_b_value = c_b_value * (c_estimated_rr_interval * 1e-3) / (assumed_rr_int * 1e-3)

            # add the adjusted b-value to the database
            data.at[idx, "b_value"] = c_b_value

        # Rotate the diffusion directions except the spoiler ones
        if not data.loc[idx, "dir_in_image_plane"]:
            c_diff_direction = data.loc[idx, "direction"]
            rot_direction = np.matmul(c_diff_direction, rotation_matrix)
            data.at[idx, "direction"] = list(rot_direction)

        # the DICOM standard for directions:
        # x positive is left to right
        # y positive top to bottom
        # so I need to invert the Y direction
        # in order to have the conventional cartesian orientations from the start
        # (x positive right to left, y positive bottom to top, z positive away from you)
        c_diff_direction = list(data.loc[idx, "direction"])
        c_diff_direction[1] = -c_diff_direction[1]
        data.at[idx, "direction"] = c_diff_direction

    if settings["debug"]:
        plot_b_values_adjustment(data, settings)

    return data


def create_2d_montage_from_database(
    data: pd.DataFrame,
    b_value_column_name: str,
    direction_column_name: str,
    info: dict,
    settings: dict,
    slices: list,
    filename: str,
    save_path: str,
    list_to_highlight: list = [],
):
    """
    Create a grid with all DWIs for each slice

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    b_value_column_name : str
        string of the column to use as the b-value
    info : dict
        useful info
    settings : dict
        settings info
    slices : list
        list of slices
    filename : str
        filename to save the montage
    list_to_highlight : list, optional
        list of indices to highlight, by default []
    save_path: str
        where to save the image

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
                c_img_stack[b_val, dir_idx] = np.stack(c_df_b_d.image.values, axis=0)

                # create a mask with 0s and 1s to highlight images in certain positions of the dataframe
                # these mask will be of the same shape as the images stack
                c_indices = c_df_b_d.index.to_numpy()
                mask = np.isin(c_indices, list_to_highlight)
                mask = mask[..., np.newaxis, np.newaxis]
                mask = np.repeat(mask, c_img_stack[b_val, dir_idx].shape[1], axis=1)
                mask = np.repeat(mask, c_img_stack[b_val, dir_idx].shape[2], axis=2)
                c_highlight_stack[b_val, dir_idx] = mask

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

            # repeat for mask
            cc_mask_stack = c_highlight_stack[key]
            cc_mask_stack = cc_mask_stack.transpose(1, 2, 0)
            cc_mask_stack = np.reshape(
                cc_mask_stack, (cc_mask_stack.shape[0], cc_mask_stack.shape[1] * cc_mask_stack.shape[2]), order="F"
            )
            montage_mask[idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_mask_stack.shape[1]] = (
                cc_mask_stack
            )

        # save montages in a figure
        fig = plt.figure(figsize=(len(c_img_stack), max_number_of_images))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(montage, cmap="Greys_r", vmin=np.min(montage), vmax=np.max(montage) * 0.35)
        plt.imshow(montage_mask, cmap="bwr", alpha=0.2 * montage_mask)
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
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def read_data(settings: dict, info: dict, logger: logging) -> [pd.DataFrame, dict, NDArray, dict]:
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
    dict
        info about the data
    pd.DataFrame
        dataframe with all the data
    dict
        settings from YAML file

    """

    # =========================================================
    # Here we have two routes: If settings["workflow_mode"] == "anon" then we will:
    # - read the dicoms
    # - remove any patient info
    # - save all the data to
    #   - a zipped dataframe
    #   - some diffusion info in a csv file
    #   - the info and slices to a pickled file
    #   - the original pixel values to an HDF5 file for easy viewing
    # - DELETE the DICOM FILES (WARNING, MAKE SURE YOU HAVE A BACKUP!)

    # If settings["workflow_mode"] == "main" then we will:
    # - load the dataframe data and continue with the processing
    # =========================================================

    if settings["workflow_mode"] == "anon":
        logger.debug("WORKFLOW MODE: ANON. Reading DICOM files.")

        data, info = get_data(settings, info)

        # number of dicom files
        info["n_files"] = data.shape[0]
        # image size
        info["img_size"] = list(data.loc[0, "image"].shape)

        # how many slices do we have and encode them with an integer
        slices = data.image_position.unique()
        n_slices = len(slices)
        info["n_slices"] = n_slices
        # create dictionaries to go from image position to integer and vice versa
        info["integer_to_image_positions"] = {}
        for idx, slice in enumerate(slices):
            info["integer_to_image_positions"][idx] = slice
        info["image_positions_to_integer"] = dict((v, k) for k, v in info["integer_to_image_positions"].items())
        # create a new column in the data table with the slice integer for that slice
        list_of_tuples = data.image_position.values
        slice_integer = [info["image_positions_to_integer"][i] for i in list_of_tuples]
        data["slice_integer"] = pd.Series(slice_integer)

        # slices is going to be a list of all the integers
        slices = data.slice_integer.unique()

        logger.debug("Number of dicom files: " + str(info["n_files"]))
        logger.debug("Number of slices: " + str(n_slices))
        logger.debug("Image size: " + str(info["img_size"]))

        # =========================================================
        # display all DWIs in a montage
        # =========================================================
        if settings["debug"]:
            create_2d_montage_from_database(
                data,
                "b_value",
                "direction",
                info,
                settings,
                slices,
                "dwis_original_dicoms",
                settings["debug_folder"],
                [],
            )

        # =========================================================
        # adjust b-values and diffusion directions to image
        # =========================================================
        data = adjust_b_val_and_dir(data, settings, info, logger)
        if settings["sequence_type"] == "steam":
            logger.info("Sequence: STEAM: b-values and directions adjusted")
        if settings["sequence_type"] == "se":
            logger.info("Sequence: SE: Directions adjusted")

        data_summary_plots(data, info, settings)

        # =========================================================
        # export all the data to files
        # =========================================================
        # save the dataframe
        save_path = os.path.join(settings["dicom_folder"], "data.zip")
        data.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})
        # save the info and slices to a pickled file
        save_path = os.path.join(settings["dicom_folder"], "info_and_slices.zip")
        with open(save_path, "wb") as f:  # Python 3: open(..., 'wb')
            pickle.dump([info, slices], f)

        # also save some diffusion info to a csv file
        save_path = os.path.join(settings["dicom_folder"], "diff_info.csv")
        data.to_csv(
            save_path,
            columns=[
                "file_name",
                "b_value",
                "b_value_original",
                "direction",
                "direction_original",
                "image_position",
                "slice_integer",
                "nominal_interval",
                "acquisition_time",
                "acquisition_date",
                "estimated_rr_interval",
            ],
            index=False,
        )

        # finally save the pixel values to HDF5
        image_pixel_values = data["image"].to_numpy()
        image_pixel_values = np.stack(image_pixel_values)
        save_path = os.path.join(settings["dicom_folder"], "images.h5")
        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("pixel_values", data=image_pixel_values, compression="gzip", compression_opts=9)

        # =========================================================
        # DELETE DICOM FILES
        # =========================================================
        def delete_file(row):
            file_path = os.path.join(settings["dicom_folder"], row["file_name"])
            os.remove(file_path)

        data.apply(delete_file, axis=1)
        logger.info("ALL DICOMS DELETED!")

    else:
        logger.debug("WORKFLOW MODE: MAIN. Reading dataframe data.")

        # read the dataframe
        save_path = os.path.join(settings["dicom_folder"], "data.zip")
        data = pd.read_pickle(save_path)

        # read the info and slices variables
        save_path = os.path.join(settings["dicom_folder"], "info_and_slices.zip")
        with open(save_path, "rb") as f:
            [info, slices] = pickle.load(f)

        n_slices = len(slices)
        logger.debug("Number of dicom files: " + str(info["n_files"]))
        logger.debug("Number of slices: " + str(n_slices))
        logger.debug("Image size: " + str(info["img_size"]))

        data_summary_plots(data, info, settings)

        # =========================================================
        # display all DWIs in a montage
        # =========================================================
        if settings["debug"]:
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "direction_original",
                info,
                settings,
                slices,
                "dwis_original_dicoms",
                settings["debug_folder"],
                [],
            )

        # plot the b-values (before and after adjustment)
        if settings["debug"]:
            plot_b_values_adjustment(data, settings)

    return data, info, slices
