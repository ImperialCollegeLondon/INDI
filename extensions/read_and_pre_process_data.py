import ast
import logging
import math
import os
import re
import sys
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import yaml
from numpy.typing import NDArray

from extensions.dwis_classifier import dwis_classifier
from extensions.extensions import crop_pad_rotate_array


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
    plt.plot(data.b_value.values, ".")
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
                ds[dicom_header_fields["DICOM_header_classic"]["b_value"]]._value
                if dicom_header_fields["DICOM_header_classic"]["b_value"] in ds
                else 0,
                # diffusion directions, or [1, 1, 1] normalised if not a field
                ds[dicom_header_fields["DICOM_header_classic"]["diffusiongradientdirection"]]._value
                if dicom_header_fields["DICOM_header_classic"]["diffusiongradientdirection"] in ds
                else [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)],
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
    estimated_rr_intervals_original = time_delta.copy()
    # get median time delta, and replace values above 4x the median with nan
    median_time = np.nanmedian(time_delta)
    time_delta[time_delta > 4 * median_time] = np.nan
    # add time delta to the dataframe
    data["estimated_rr_interval"] = time_delta
    # replace nans with the next non-nan value
    data["estimated_rr_interval"] = data["estimated_rr_interval"].bfill()

    return data, estimated_rr_intervals_original


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

    data, estimated_rr_intervals_original = estimate_rr_interval(data)

    # get the b0 value and assumed RR interval from the image comment field
    if settings["sequence_type"] == "steam":
        if info["image_comments"]:
            logger.debug("Dicom header comment found: " + info["image_comments"])
            # get all numbers from comment field
            m = re.findall(r"[-+]?(?:\d*\.*\d+)", info["image_comments"])
            m = [float(m) for m in m]
            if len(m) > 2:
                sys.exit("Seems we have > 2 numbers in the comment field!")
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
                c_b_value = c_b_value * (c_nominal_interval / 1000) / (assumed_rr_int / 1000)
            else:
                c_b_value = c_b_value * (c_estimated_rr_interval) / (assumed_rr_int / 1000)

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

    # plot the b-values (before and after adjustment)
    # plot the nominal intervals
    if settings["debug"]:
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
        plt.plot(estimated_rr_intervals_original, alpha=0.8)
        plt.legend(["nominal", "adjusted RR", "original RR"])
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
            montage[
                idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_img_stack.shape[1]
            ] = cc_img_stack

            # repeat for mask
            cc_mask_stack = c_highlight_stack[key]
            cc_mask_stack = cc_mask_stack.transpose(1, 2, 0)
            cc_mask_stack = np.reshape(
                cc_mask_stack, (cc_mask_stack.shape[0], cc_mask_stack.shape[1] * cc_mask_stack.shape[2]), order="F"
            )
            montage_mask[
                idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_mask_stack.shape[1]
            ] = cc_mask_stack

        # save montages in a figure
        fig = plt.figure(figsize=(len(c_img_stack), max_number_of_images))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(montage, cmap="Greys_r", vmin=np.min(montage), vmax=np.max(montage))
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


def remove_outliers_ai(
    data: pd.DataFrame,
    info: dict,
    settings: dict,
    slices: NDArray,
    logger: logging.Logger,
    threshold: float = 0.3,
) -> [pd.DataFrame, NDArray]:
    """Remove the bad frames from the dataframe using the AI classifier

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    info : dict
        useful info
    settings : dict
        settings
    slices : NDArray
        array with slice positions as strings
    logger : logging.Logger
        logger
    threshold : float, optional
        threshold value to consider bad in [0, 1], by default 0.3

    Returns
    -------
    Tuple[pd.DataFrame, NDArray]
        [dataframe without bad frames, array with bad frames positions]
    """

    # gather images from dataframe
    dwis = np.empty([info["n_files"], info["img_size"][0], info["img_size"][1]])
    for i in range(info["n_files"]):
        # moving image
        dwis[i] = data.loc[i, "image"]

    # make sure image stack has the correct dimensions
    dwis = crop_pad_rotate_array(dwis, (info["n_files"], 256, 96), True)

    # use the AI classifier to determine which ones are bad
    frame_labels = dwis_classifier(dwis, threshold)

    # drop frames frames labeled as bad (0)
    rows_to_drop = np.where(frame_labels < 1)[0]
    data_new = data.drop(index=list(rows_to_drop))

    logger.debug("Number of images removed by AI: " + str(len(rows_to_drop)))

    return data_new, rows_to_drop


def manual_image_removal(
    data: pd.DataFrame, info: dict, settings: dict, slices: NDArray, logger: logging.Logger
) -> [pd.DataFrame, pd.DataFrame, dict, NDArray]:
    """
    Manual removal of images. A matplotlib window will open, and we can select images to be removed.

    Parameters
    ----------
    data: dataframe with all the dwi data
    info: dict
    settings: dict
    slices: array with slice positions
    logger: logger for console and file

    Returns
    -------
    dataframe with all the data
    dataframe with data without the rejected images
    info: dict
    array with indices of rejected images in the original dataframe

    """
    # this is an empty dataframe to store all rejected data
    rejected_images_data = pd.DataFrame(columns=list(data.columns))
    # store also original dataframe before any image removal
    data_original = data.copy()

    # loop over the slices
    for slice_idx in slices:
        # dataframe with current slice
        c_df = data.loc[data["slice_integer"] == slice_idx].copy()

        # initiate maximum number of images found for each b-val and dir combination
        max_number_of_images = 0

        # convert list of directions to a tuple
        c_df["direction_original"] = c_df["direction_original"].apply(tuple)

        # get unique b-values
        b_vals = c_df.b_value_original.unique()
        b_vals.sort()

        # initiate the stacks for the images and the highlight masks
        c_img_stack = {}
        c_img_indices = {}

        # loop over sorted b-values
        for b_val in b_vals:
            # dataframe with current slice and current b-value
            c_df_b = c_df.loc[c_df["b_value_original"] == b_val].copy()

            # unique directions
            dirs = c_df_b["direction_original"].unique()
            dirs.sort()

            # loop over directions
            for dir_idx, dir in enumerate(dirs):
                # dataframe with current slice, current b-value, and current direction images
                c_df_b_d = c_df_b.loc[c_df_b["direction_original"] == dir].copy()

                # for each b_val and each dir collect all images
                c_img_stack[b_val, dir] = np.stack(c_df_b_d.image.values, axis=0)
                c_img_indices[b_val, dir] = c_df_b_d.index.values

                # record n_images if bigger than the values stored
                n_images = c_df_b_d.shape[0]
                if n_images > max_number_of_images:
                    max_number_of_images = n_images

        # plot all stored images,
        # y-axis b-value and direction combos
        # x-axis repetitions
        store_selected_images = []
        # retina screen resolution
        my_dpi = 192
        rows = len(c_img_stack)
        cols = max_number_of_images
        fig, axs = plt.subplots(rows, cols, figsize=(1500 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        for idx, key in enumerate(c_img_stack):
            cc_img_stack = c_img_stack[key]
            for idx2, img in enumerate(cc_img_stack):
                axs[idx, idx2].imshow(img, cmap="gray")
                axs[idx, idx2].text(
                    5,
                    5,
                    str(key[0]),
                    fontsize=3,
                    color="tab:orange",
                    horizontalalignment="left",
                    verticalalignment="top",
                )

                # axs[idx, idx2].set_title(
                #     str(key[0]) + "_" + str(idx),
                #     fontsize=2,
                # )

                axs[idx, idx2].set_xticks([])
                axs[idx, idx2].set_yticks([])

                axs[idx, idx2].name = str(key + (idx2,))
        # Setting the values for all axes.
        plt.setp(axs, xticks=[], yticks=[])
        plt.tight_layout(pad=0.1)
        # remove axes with no image
        [p.set_axis_off() for p in [i for i in axs.flatten() if len(i.images) < 1]]

        def onclick_select(event):
            """function to record the axes of the selected images in subplots"""
            if event.inaxes is not None:
                print(event.inaxes.name)
                if ast.literal_eval(event.inaxes.name) in store_selected_images:
                    store_selected_images.remove(ast.literal_eval(event.inaxes.name))
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(1)
                else:
                    event.inaxes.set_alpha(0.1)
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(1)
                    store_selected_images.append(ast.literal_eval(event.inaxes.name))
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick_select)
        plt.show()

        # now we have to loop over all selected images
        # remove them from the dataframe data
        for item in store_selected_images:
            c_bval = item[0]
            c_dir = item[1]
            c_idx = item[2]

            # locate item in dataframe containing all images for this slice
            c_table = c_df[(c_df["direction_original"] == c_dir)]
            c_table = c_table[(c_table["b_value_original"] == c_bval)]
            c_filename = c_table.iloc[c_idx]["file_name"]

            # # not a good idea to do this
            # # move file to rejected folder
            # origin = os.path.join(settings["dicom_folder"], c_filename)
            # destination = os.path.join(settings["dicom_folder"], "rejected_images", c_filename)
            # shutil.move(origin, destination)

            # remove row from the dataframe
            data.drop(data[data.file_name == c_filename].index, inplace=True)
            rejected_images_data = pd.concat([rejected_images_data, c_table.iloc[c_idx].to_frame().T])

    # store original dataframe with all images and also as an attribute
    # the indices of the rejected frames
    # this dataframe will be used to plot all dwis with red labels on the rejected ones
    rejected_indices = rejected_images_data.index.values
    data_original.attrs["rejected_images"] = rejected_indices
    save_path = os.path.join(settings["session"], "image_manual_removal.zip")
    data_original.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})

    # display some info
    info["n_files"] = len(data)
    data.reset_index(drop=True, inplace=True)
    info["n_images_rejected"] = len(rejected_indices)
    logger.info("Number of images rejected: " + str(info["n_images_rejected"]))

    return data_original, data, info, rejected_indices


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

    data_summary_plots(data, info, settings)

    # =========================================================
    # display all DWIs in a montage
    # =========================================================
    if settings["debug"]:
        create_2d_montage_from_database(
            data, "b_value", "direction", info, settings, slices, "dwis_original_dicoms", settings["debug_folder"], []
        )

    return data, info, slices


def pre_process_data(
    data: pd.DataFrame, slices: NDArray, settings: dict, info: dict, logger: logging
) -> [pd.DataFrame, dict, NDArray]:
    """
    Pre-process data: adjust diffusion info, remove outliers, and display all DWIs in a montage

    Parameters
    ----------
    data: dataframe with images and diffusion info
    slices: array with slice integers
    settings
    options
    info
    logger

    Returns
    -------
    data, info, slices

    """

    # =========================================================
    # adjust b-values and diffusion directions to image
    # =========================================================
    data = adjust_b_val_and_dir(data, settings, info, logger)
    if settings["sequence_type"] == "steam":
        logger.info("b-values and directions adjusted")
    if settings["sequence_type"] == "se":
        logger.info("Directions adjusted")
    # ========================================================={
    # manual removal of images
    # =========================================================
    if settings["remove_outliers_manually"]:
        # check if this manual removal has been previously done
        if os.path.exists(os.path.join(settings["session"], "image_manual_removal.zip")):
            logger.info("Manual image removal already done, loading information.")
            # load initial database with rejected images included
            data_original = pd.read_pickle(os.path.join(settings["session"], "image_manual_removal.zip"))
            rejected_indices = data_original.attrs["rejected_images"]
            info["n_images_rejected"] = len(rejected_indices)
            logger.debug("Number of images rejected: " + str(info["n_images_rejected"]))
        else:
            # Manual image removal
            logger.info("Starting manual image removal...")
            # if not os.path.exists(os.path.join(settings["dicom_folder"], "rejected_images")):
            #     os.makedirs(os.path.join(settings["dicom_folder"], "rejected_images"))
            data_original, data, info, rejected_indices = manual_image_removal(data, info, settings, slices, logger)
            logger.info("Manual image removal done.")
    else:
        # no image removal to be done
        logger.info("No image removal to be done.")
        data_original = data.copy()
        rejected_indices = []

    # =========================================================
    # remove outliers with AI
    # =========================================================
    if settings["remove_outliers_with_ai"]:
        data_new, rows_to_drop = remove_outliers_ai(data, info, settings, slices, logger, threshold=0.25)
        logger.info("Removed outliers with AI")
        if settings["debug"]:
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "direction_original",
                info,
                settings,
                slices,
                "dwis_outliers_with_ai",
                settings["debug_folder"],
                rows_to_drop,
            )
            logger.info("2d montage of DWIs after outlier removal with AI")

        # copy dataframe with removed images to the data variable
        data = data_new.reset_index(drop=True)
        del data_new
        # update number of dicom files
        info["n_files"] = data.shape[0]
        logger.debug("DWIs after outlier removal with AI: " + str(info["n_files"]))

    # =========================================================
    # display all DWIs in a montage
    # =========================================================
    create_2d_montage_from_database(
        data_original,
        "b_value_original",
        "direction_original",
        info,
        settings,
        slices,
        "dwis_after_image_rejection",
        os.path.join(settings["results"], "results_b"),
        rejected_indices,
    )

    return data, info, slices
