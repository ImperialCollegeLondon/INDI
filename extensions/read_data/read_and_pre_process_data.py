import json
import logging
import math
import os
import re
import shutil
import sys
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import py7zr
import pydicom
import scipy.ndimage
from dotenv import dotenv_values
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
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()


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


def collect_global_header_info(dicom_header_fields: dict, dicom_type: int) -> dict:
    """
    Collect global header information from the fist dicom

    Parameters
    ----------
    dicom_header_fields
    dicom_type

    Returns
    -------

    header_info dict

    """

    header_info = {}

    # image comments
    if dicom_type == 2:
        header_info["image_comments"] = (
            dicom_header_fields["ImageComments"] if "ImageComments" in dicom_header_fields.keys() else None
        )
    elif dicom_type == 1:
        header_info["image_comments"] = (
            dicom_header_fields["ImageComments"] if "ImageComments" in dicom_header_fields.keys() else None
        )

    # image orientation patient
    if dicom_type == 2:
        temp_val = dicom_header_fields["PerFrameFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0][
            "ImageOrientationPatient"
        ]
        header_info["image_orientation_patient"] = [float(i) for i in temp_val]
    elif dicom_type == 1:
        temp_val = dicom_header_fields["ImageOrientationPatient"]
        header_info["image_orientation_patient"] = [float(i) for i in temp_val]

    # pixel spacing
    if dicom_type == 2:
        temp_val = dicom_header_fields["PerFrameFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0][
            "PixelSpacing"
        ]
        header_info["pixel_spacing"] = [float(i) for i in temp_val]
    elif dicom_type == 1:
        temp_val = dicom_header_fields["PixelSpacing"]
        header_info["pixel_spacing"] = [float(i) for i in temp_val]

    # slice thickness
    if dicom_type == 2:
        temp_val = dicom_header_fields["PerFrameFunctionalGroupsSequence"][0]["PixelMeasuresSequence"][0][
            "SliceThickness"
        ]
        header_info["slice_thickness"] = float(temp_val)
    elif dicom_type == 1:
        temp_val = dicom_header_fields["SliceThickness"]
        header_info["slice_thickness"] = float(temp_val)

    return header_info


def get_pixel_array(ds: pydicom.dataset.Dataset, dicom_type: str, frame_idx: int, image_type: str) -> NDArray:
    """
    Get the pixel array from the DICOM header.
    Pixel values = data_array * slope + intercept

    Parameters
    ----------
    ds
    dicom_type
    frame_idx
    image type: mag or phase

    Returns
    -------
    pixel array

    """
    pixel_array = ds.pixel_array
    # check if largest dimension is lower than 192
    # if so, then interpolate array by a factor of two
    larger_dim = max(pixel_array.shape)
    interp_img = True if larger_dim <= 192 else False

    def interpolate_img(img, image_type):
        if image_type == "mag":
            img = scipy.ndimage.zoom(img, 2, order=3)
            # zero any negative pixels after interpolation
            img[img < 0] = 0
        elif image_type == "phase":
            # convert phase to real and imaginary before interpolating
            img = np.pi * img / 4096
            img_real = np.cos(img)
            img_real = scipy.ndimage.zoom(img_real, 2, order=0)
            img_imag = np.sin(img)
            img_imag = scipy.ndimage.zoom(img_imag, 2, order=0)
            # revert back to the original phase values
            img = np.arctan2(img_imag, img_real)
            img = 4096 * img / np.pi
        return img

    if dicom_type == 2:
        slope = float(
            ds["PerFrameFunctionalGroupsSequence"][frame_idx]["PixelValueTransformationSequence"][0].RescaleSlope
        )
        intercept = float(
            ds["PerFrameFunctionalGroupsSequence"][frame_idx]["PixelValueTransformationSequence"][0].RescaleIntercept
        )
        if pixel_array.ndim == 3:
            img = pixel_array[frame_idx] * slope + intercept
            if interp_img:
                img = interpolate_img(img, image_type)
        elif pixel_array.ndim == 2:
            img = pixel_array * slope + intercept
            if interp_img:
                img = interpolate_img(img, image_type)
    elif dicom_type == 1:
        if "RescaleSlope" in ds:
            img = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        else:
            img = pixel_array
        if interp_img:
            img = interpolate_img(img, image_type)

    return img


def get_b_value(c_dicom_header: dict, dicom_type: str, dicom_manufacturer: str, frame_idx: int) -> float:
    """
    Get b-value from a dict with the DICOM header.
    If no b-value fond, then return 0.0

    Parameters
    ----------
    c_dicom_header
    dicom_type
    dicom_manufacturer
    frame_idx

    Returns
    -------
    b_value

    """
    if dicom_type == 2:
        if (
            "DiffusionBValue"
            in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0].keys()
        ):
            return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0][
                "DiffusionBValue"
            ]
        else:
            return 0.0

    elif dicom_type == 1:
        if dicom_manufacturer == "siemens":
            if "DiffusionBValue" in c_dicom_header.keys():
                return c_dicom_header["DiffusionBValue"]
            else:
                return 0.0
        elif dicom_manufacturer == "philips":
            return c_dicom_header["DiffusionBValue"]


def get_diffusion_directions(
    c_dicom_header: dict, dicom_type: str, dicom_manufacturer: str, frame_idx: int, settings: dict
) -> Tuple:
    """
    Get diffusion direction 3D vector.
    If no direction found, and sequence STEAM
    then return a normalised vector [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)].
    This makes sense for the STEAM because of the spoilers. For the SE if no direction, then b-value
    will be 0, and this gradient is not significant.

    Parameters
    ----------
    c_dicom_header
    dicom_type
    dicom_manufacturer
    frame_idx

    Returns
    -------
    Diffusion direction

    """
    if dicom_type == 2:
        if (
            "DiffusionGradientDirectionSequence"
            in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0].keys()
            and c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0][
                "DiffusionDirectionality"
            ]
            != "NONE"
        ):
            val = tuple(
                [
                    float(i)
                    for i in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0][
                        "DiffusionGradientDirectionSequence"
                    ][0]["DiffusionGradientOrientation"]
                ]
            )
            return val
        else:
            if settings["sequence_type"] == "steam":
                return (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))
            else:
                return (0.0, 0.0, 0.0)

    elif dicom_type == 1:
        if dicom_manufacturer == "siemens":
            if "DiffusionGradientDirection" in c_dicom_header:
                return tuple([float(i) for i in c_dicom_header["DiffusionGradientDirection"]])
            else:
                if settings["sequence_type"] == "steam":
                    return (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))
                else:
                    return (0.0, 0.0, 0.0)
        elif dicom_manufacturer == "philips":
            return tuple(c_dicom_header["DiffusionGradientOrientation"])


def get_image_position(c_dicom_header: dict, dicom_type: str, frame_idx: int) -> Tuple:
    """
    Get the image position patient info from the DICOM header

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    image position patient

    """
    if dicom_type == 2:
        val = tuple(
            [
                float(i)
                for i in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["PlanePositionSequence"][0][
                    "ImagePositionPatient"
                ]
            ]
        )

        return val

    elif dicom_type == 1:
        val = tuple([float(i) for i in c_dicom_header["ImagePositionPatient"]])

        return val


def get_nominal_interval(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> float:
    """
    Get the nominal interval from the DICOM header

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Nominal interval

    """
    if dicom_type == 2:
        if "CardiacSynchronizationSequence" in c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]:
            val = float(
                c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["CardiacSynchronizationSequence"][0][
                    "RRIntervalTimeNominal"
                ]
            )
        else:
            val = "None"
        return val

    elif dicom_type == 1:
        if "NominalInterval" in c_dicom_header:
            val = float(c_dicom_header["NominalInterval"])
        else:
            val = "None"
        return val


def get_acquisition_time(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get acquisition time string

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Acquisition time

    """
    if dicom_type == 2:
        return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["FrameContentSequence"][0][
            "FrameAcquisitionDateTime"
        ][8:]

    elif dicom_type == 1:
        return c_dicom_header["AcquisitionTime"]


def get_acquisition_date(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get acquisition date string.

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    Acquisition date

    """
    if dicom_type == 2:
        return c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["FrameContentSequence"][0][
            "FrameAcquisitionDateTime"
        ][:8]

    elif dicom_type == 1:
        return c_dicom_header["AcquisitionDate"]


def get_diffusion_direction_in_plane_bool(
    c_dicom_header: dict, dicom_type: int, dicom_manufacturer: str, frame_idx: int
) -> bool:
    """
    Get boolean if the direction given is in the image plane or not.
    For the STEAM sequence the spoiler gradients of the b0 are in the image plane,
    but the standard diffusion directions are not for the SE and STEAM.

    Parameters
    ----------
    c_dicom_header
    dicom_type
    dicom_manufacturer
    frame_idx

    Returns
    -------
    boolean

    """
    if dicom_type == 2:
        if (
            c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["MRDiffusionSequence"][0][
                "DiffusionDirectionality"
            ]
            == "BMATRIX"
        ):
            return False
        else:
            return True

    elif dicom_type == 1:
        if dicom_manufacturer == "siemens":
            if "DiffusionGradientDirection" in c_dicom_header:
                return False
            else:
                return True
        elif dicom_manufacturer == "philips":
            if "DiffusionGradientDirection" in c_dicom_header:
                return False
            else:
                True


def get_series_description(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> str:
    """
    Get series description

    Parameters
    ----------
    c_dicom_header
    dicom_type
    frame_idx

    Returns
    -------
    series description
    """

    if dicom_type == 2:
        return c_dicom_header["SeriesDescription"]

    elif dicom_type == 1:
        if "SeriesDescription" in c_dicom_header.keys():
            return c_dicom_header["SeriesDescription"]
        else:
            return "None"


# get DICOM header fields
def dictify(ds: pydicom.dataset.Dataset) -> dict:
    """
    Turn a pydicom Dataset into a dict with keys derived from the Element tags.
    Private info is not collected, because we cannot access it with the keyword.
    So we need to manually fish the diffusion information in the old DICOMs.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The Dataset to dictify

    Returns
    -------
    DICOM header as a dict
    """

    output = dict()
    # iterate over all non private fields
    for elem in ds:
        if elem.VR != "SQ":
            output[elem.keyword] = elem.value
        else:
            output[elem.keyword] = [dictify(item) for item in elem]

    # add manually private diffusion fields if they exist
    if [0x0019, 0x100C] in ds:
        output["DiffusionBValue"] = ds[0x0019, 0x100C].value
    if [0x0019, 0x100E] in ds:
        output["DiffusionGradientDirection"] = ds[0x0019, 0x100E].value
    return output


def get_data_old_or_modern_dicoms(
    list_dicoms: list, settings: dict, info: dict, logger: logging.Logger, image_type: str = "mag"
) -> Tuple[pd.DataFrame, dict]:
    """
    Read all the DICOM files in data_folder_path and store important info
    in a dataframe and some header info in a dictionary

    Parameters
    ----------
    list_dicoms: list with DICOM files
    settings: dict
    info: dict
    logger
    image_type: str

    Returns
    -------
    df: dataframe with the DICOM diffusion information
    info: dictionary with useful info
    """
    # path to DICOM files
    if image_type == "mag":
        data_folder_path = settings["dicom_folder"]
        logger.debug("Magnitude DICOMs")
    elif image_type == "phase":
        data_folder_path = settings["dicom_folder_phase"]
        logger.debug("Phase DICOMs")
    else:
        sys.exit("Image type not supported.")

    # collect some header info in a dictionary from the first DICOM
    ds = pydicom.dcmread(os.path.join(data_folder_path, list_dicoms[0]))

    # check version of dicom
    dicom_type = 0
    if "PerFrameFunctionalGroupsSequence" in ds:
        dicom_type = 2
        logger.debug("DICOM type: Modern")
        # How many images in one file?
        n_images_per_file = len(ds.PerFrameFunctionalGroupsSequence)
        logger.debug("Number of images per DICOM: " + str(n_images_per_file))
    else:
        dicom_type = 1
        logger.debug("DICOM type: Legacy")
        n_images_per_file = 1

    # check manufacturer
    if "Manufacturer" in ds:
        if ds.Manufacturer == "Siemens Healthineers" or ds.Manufacturer == "Siemens" or ds.Manufacturer == "SIEMENS":
            logger.debug("Manufacturer: SIEMENS")
            dicom_manufacturer = "siemens"
        elif ds.Manufacturer == "Philips Medical Systems" or ds.Manufacturer == "Philips":
            logger.debug("Manufacturer: Philips")
            dicom_manufacturer = "philips"
        elif ds.Manufacturer == "GE MEDICAL SYSTEMS":
            logger.debug("Manufacturer: GE")
            sys.exit("GE DICOMs not supported yet.")
        else:
            logger.debug("Manufacturer: " + ds.Manufacturer)
            sys.exit("Manufacturer not supported.")
    else:
        logger.debug("Manufacturer: None")
        sys.exit("Manufacturer not supported.")

    # get DICOM header in a dict
    dicom_header_fields = dictify(ds)

    # collect some global header info in a dictionary
    header_info = collect_global_header_info(dicom_header_fields, dicom_type)

    # load sensitive fields from csv into a dataframe
    sensitive_fields = pd.read_csv(os.path.join(settings["code_path"], "extensions", "anon_fields.csv"))

    # create a dataframe with all DICOM values
    df = []
    for idx, file_name in enumerate(list_dicoms):
        # read current DICOM
        ds = pydicom.dcmread(os.path.join(data_folder_path, file_name))
        # loop over the dictionary of header fields and collect them for this DICOM file
        c_dicom_header = dictify(ds)
        # remove sensitive data
        field_list = sensitive_fields["sensitive_fields"].tolist()
        for field in field_list:
            if field in c_dicom_header:
                c_dicom_header.pop(field)

        # loop over each frame within each file
        for frame_idx in range(n_images_per_file):
            # append values (will be a row in the dataframe)
            df.append(
                (
                    # file name
                    file_name,
                    # array of pixel values
                    get_pixel_array(ds, dicom_type, frame_idx, image_type),
                    # b-value or zero if not a field
                    get_b_value(c_dicom_header, dicom_type, dicom_manufacturer, frame_idx),
                    # diffusion directions
                    get_diffusion_directions(c_dicom_header, dicom_type, dicom_manufacturer, frame_idx, settings),
                    # image position
                    get_image_position(c_dicom_header, dicom_type, frame_idx),
                    # nominal interval
                    get_nominal_interval(c_dicom_header, dicom_type, frame_idx),
                    # acquisition time
                    get_acquisition_time(c_dicom_header, dicom_type, frame_idx),
                    # acquisition date
                    get_acquisition_date(c_dicom_header, dicom_type, frame_idx),
                    # False if diffusion direction is a field
                    get_diffusion_direction_in_plane_bool(c_dicom_header, dicom_type, dicom_manufacturer, frame_idx),
                    # series description
                    get_series_description(c_dicom_header, dicom_type, frame_idx),
                    # get_series_number
                    c_dicom_header["SeriesNumber"] if "SeriesNumber" in c_dicom_header.keys() else None,
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
            "series_description",
            "series_number",
            "header",
        ],
    )

    # merge header info into info
    info = {**info, **header_info}

    return df, info


def get_nii_pixel_array(nii_px_array, c_slice_idx, c_frame_idx):
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


def get_nii_timings(
    rr_interval_table: pd.DataFrame,
    nii_file: str,
    frame_idx: int,
    slice_idx: int,
    n_images_per_file: int,
    n_slices_per_file: int,
    col_string: str,
) -> int | str:
    """
    Get the nominal interval or the acquisition time or date of the current image

    Parameters
    ----------
    rr_interval_table
    nii_file
    frame_idx
    slice_idx
    n_images_per_file
    n_slices_per_file
    col_string

    Returns
    -------
    nominal interval or string with acquisition date or time.

    """

    if not rr_interval_table.empty:
        nii_file_string = nii_file.replace(".nii", "")
        c_table = rr_interval_table[rr_interval_table["nii_file_suffix"].str.endswith(nii_file_string)]
        while len(c_table) == 0:
            nii_file_string = nii_file_string.split("_", 1)[1]
            c_table = rr_interval_table[rr_interval_table["nii_file_suffix"].str.endswith(nii_file_string)]

        # here I am assuming that the order of the timings in the csv file is ordered like this:
        # first goes through all slices, then moves to the next bval/bvec.
        # Not sure if this will be the case everytime with enhanced DICOMs.
        row_pos = slice_idx + frame_idx * n_slices_per_file

        return c_table.iloc[row_pos][col_string]
    else:
        return None


def get_current_rr_table(rr_interval_table: pd.DataFrame, nii_file: str) -> pd.DataFrame:
    """
    Get the current table of values for this series

    Parameters
    ----------
    rr_interval_table
    nii_file

    Returns
    -------
    dataframe

    """
    # If table is not empty, then get the smaller table for this series
    # otherwise return an empty table
    if not rr_interval_table.empty:
        nii_file_string = nii_file.replace(".nii", "")
        c_table = rr_interval_table[rr_interval_table["nii_file_suffix"].str.endswith(nii_file_string)]
        while len(c_table) == 0:
            nii_file_string = nii_file_string.split("_", 1)[1]
            c_table = rr_interval_table[rr_interval_table["nii_file_suffix"].str.endswith(nii_file_string)]
    else:
        c_table = pd.DataFrame()

    return c_table


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
            "direction",
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


def estimate_rr_interval(data: pd.DataFrame, settings) -> tuple[pd.DataFrame, NDArray]:
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
    dataframe with added estimated RR interval column
    estimated_rr_intervals_original (before adjustment, only for debug)
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
        dpi=200,
        pad_inches=0,
        transparent=False,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
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
    data["direction_original"] = data["direction"]

    data = estimate_rr_interval(data, settings)

    # adjust b-values if STEAM sequence and DICOM data
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

        # loop through the entries and adjust b-values and directions
        logger.debug("STEAM sequence and DICOM data: adjusting b-values")
        for idx in range(n_entries):
            c_b_value = data.loc[idx, "b_value"]

            # replace b0 value
            if c_b_value == 0:
                c_b_value = calculated_real_b0

            # This is bypassed for ex-vivo data, as the prescribed b-value is the correct one
            # (with the exception of the b0 which was corrected above)
            # correct b_value relative to the assumed RR interval with the nominal interval if not 0.0.
            # Otherwise use the estimated RR interval.
            if settings["ex_vivo"]:
                data.at[idx, "b_value"] = c_b_value
            else:
                c_nominal_interval = data.loc[idx, "nominal_interval"]
                c_estimated_rr_interval = data.loc[idx, "estimated_rr_interval"]
                if c_nominal_interval != 0.0 and c_nominal_interval != "None":
                    c_b_value = c_b_value * (c_nominal_interval * 1e-3) / (assumed_rr_int * 1e-3)
                else:
                    c_b_value = c_b_value * (c_estimated_rr_interval * 1e-3) / (assumed_rr_int * 1e-3)
                # add the adjusted b-value to the database
                data.at[idx, "b_value"] = c_b_value

    else:
        assumed_rr_int = None
        calculated_real_b0 = None
        logger.debug("SE sequence or NIFTI data: not adjusting b-values")

    # adjust the diffusion directions to the image plane
    if data_type == "dicom" or data_type == "pandas":
        logger.debug("DICOM data: rotating directions to the image plane.")

        # get the rotation matrix
        first_column = np.array(info["image_orientation_patient"][0:3])
        second_column = np.array(info["image_orientation_patient"][3:6])
        third_column = np.cross(first_column, second_column)
        rotation_matrix = np.stack((first_column, second_column, third_column), axis=-1)

        # loop through the entries and adjust b-values and directions
        for idx in range(n_entries):
            # Rotate the diffusion directions except if dir_in_image_plane is True
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
    if data_type == "dicom" or data_type == "pandas":
        for idx in range(n_entries):
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
                c_img_stack[b_val, dir_idx] = np.stack(c_df_b_d[image_label].values, axis=0)
                c_img_stack_series_description[b_val, dir_idx] = c_df_b_d.series_description.values

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

            if list_to_highlight:
                # repeat for mask
                cc_mask_stack = c_highlight_stack[key]
                cc_mask_stack = cc_mask_stack.transpose(1, 2, 0)
                cc_mask_stack = np.reshape(
                    cc_mask_stack, (cc_mask_stack.shape[0], cc_mask_stack.shape[1] * cc_mask_stack.shape[2]), order="F"
                )
                montage_mask[
                    idx * info["img_size"][0] : (idx + 1) * info["img_size"][0], : cc_mask_stack.shape[1]
                ] = cc_mask_stack

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
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def reorder_by_slice(data, settings, info, logger):
    """
    Reorder data by slice and remove slices if needed

    Parameters
    ----------
    data
    """
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

    return data, info, slices, n_slices


def read_data(settings: dict, info: dict, logger: logging) -> tuple[pd.DataFrame, dict, NDArray, dict]:
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

    # initiate variables
    data_type = None
    settings["complex_data"] = False
    list_dicoms = []
    list_dicoms_phase = []
    list_nii = []

    # gather possible dicoms, or nii file paths into a list
    data_type, list_dicoms, list_dicoms_phase, list_nii = list_files(data_type, logger, settings)

    # If DICOMs are present, then we will read them and build our diffusion database.
    # If DICOMs are not present but NIFTI files are, then we will read them and build our diffusion database.
    # If neither DICOMs nor NIFTI files are present, then we will read the pre-saved database
    if data_type == "dicom":
        data, data_phase, info = read_and_process_dicoms(info, list_dicoms, list_dicoms_phase, logger, settings)

    elif data_type == "nii":
        # read nii files
        logger.debug("Nii files found.")
        data, info = read_and_process_niis(list_nii, settings, info, logger)

    else:
        # read pandas
        data_type = "pandas"
        data, data_phase, info = read_and_process_pandas(info, logger, settings)

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
    # re-order data again, this time by slice first, then by date-time
    # also potentially remove slices
    # =========================================================
    data, info, slices, n_slices = reorder_by_slice(data, settings, info, logger)

    # number of dicom files
    info["n_images"] = data.shape[0]
    # image size
    info["img_size"] = list(data.loc[0, "image"].shape)

    data_summary_plots(data, info, settings)

    logger.debug("Number of images: " + str(info["n_images"]))
    logger.debug("Number of slices: " + str(len(slices)))
    logger.debug("Image size: " + str(info["img_size"]))

    # =========================================================
    # display all DWIs in a montage
    # =========================================================
    if settings["debug"]:
        create_2d_montage_from_database(
            data,
            "b_value_original",
            "direction_original",
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
                "direction_original",
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
    data.to_csv(
        save_path,
        columns=[
            "file_name",
            "b_value",
            "b_value_original",
            "direction",
            "direction_original",
            "dir_in_image_plane",
            "image_position",
            "image_position_label",
            "slice_integer",
            "nominal_interval",
            "estimated_rr_interval",
            "acquisition_date_time",
            "series_description",
            "series_number",
        ],
        index=False,
    )

    # plot the b-values (before and after adjustment)
    if settings["debug"] and settings["sequence_type"] == "steam":
        plot_b_values_adjustment(data, settings)

    return data, info, slices


def read_and_process_pandas(info: dict, logger: logging, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Read pandas dataframe

    Parameters
    ----------
    info
    logger
    settings

    Returns
    -------

    """
    logger.debug("No DICOM or Nii files found. Reading diffusion database files previously created.")
    # check if subfolders "mag" and "phase" exist
    mag_folder = os.path.join(settings["dicom_folder"], "mag")
    phase_folder = os.path.join(settings["dicom_folder"], "phase")
    if os.path.exists(mag_folder) and os.path.exists(phase_folder):
        settings["complex_data"] = True
        settings["dicom_folder"] = mag_folder
        settings["dicom_folder_phase"] = phase_folder

        logger.debug("Magnitude and phase folders found.")
        logger.debug("Complex averaging on.")
    # =========================================================
    # read the dataframe
    # =========================================================
    save_path = os.path.join(settings["dicom_folder"], "data.gz")
    data = pd.read_pickle(save_path)
    info = data.attrs["info"]
    if settings["complex_data"]:
        save_path = os.path.join(settings["dicom_folder_phase"], "data.gz")
        data_phase = pd.read_pickle(save_path)
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

    """

    # create empty dataframe
    data_phase = pd.DataFrame()

    # read DICOM info
    data, info = get_data_old_or_modern_dicoms(list_dicoms, settings, info, logger, image_type="mag")
    if settings["complex_data"]:
        data_phase, info = get_data_old_or_modern_dicoms(list_dicoms_phase, settings, info, logger, image_type="phase")

        # check if the magnitude and phase tables match
        data_sort = sort_by_date_time(data)
        data_phase_sort = sort_by_date_time(data_phase)
        if not data_sort.drop(columns=["file_name", "image", "series_number", "header"]).equals(
            data_phase_sort.drop(columns=["file_name", "image", "series_number", "header"])
        ):
            logger.error("Magnitude and phase DICOM tables do not match!")
            sys.exit()
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
    data.to_csv(
        save_path,
        columns=[
            "file_name",
            "b_value",
            "direction",
            "dir_in_image_plane",
            "image_position",
            "nominal_interval",
            "series_description",
            "series_number",
        ],
        index=False,
    )
    if settings["complex_data"]:
        save_path = os.path.join(settings["dicom_folder_phase"], "diff_info_dataframe_h5.csv")
        data_phase.to_csv(
            save_path,
            columns=[
                "file_name",
                "b_value",
                "direction",
                "dir_in_image_plane",
                "image_position",
                "nominal_interval",
                "series_description",
                "series_number",
            ],
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
        env_vars = dotenv_values(os.path.join(settings["code_path"], ".env"))

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


def list_files(data_type, logger, settings):
    list_dicoms = []
    list_dicoms_phase = []
    list_nii = []

    # Check for DICOM files
    included_extensions = ["dcm", "DCM", "IMA"]
    list_dicoms = [
        fn for fn in os.listdir(settings["dicom_folder"]) if any(fn.endswith(ext) for ext in included_extensions)
    ]
    if len(list_dicoms) > 0:
        data_type = "dicom"
        logger.debug("DICOM files found.")
        list_dicoms.sort()

    else:
        # check if subfolders "mag" and "phase" exist
        # if so read all dicom files in those folders
        mag_folder = os.path.join(settings["dicom_folder"], "mag")
        phase_folder = os.path.join(settings["dicom_folder"], "phase")
        if os.path.exists(mag_folder) and os.path.exists(phase_folder):
            list_dicoms = [fn for fn in os.listdir(mag_folder) if any(fn.endswith(ext) for ext in included_extensions)]
            list_dicoms_phase = [
                fn for fn in os.listdir(phase_folder) if any(fn.endswith(ext) for ext in included_extensions)
            ]
            if len(list_dicoms) > 0 and len(list_dicoms_phase) > 0:
                data_type = "dicom"
                settings["complex_data"] = True
                settings["dicom_folder"] = mag_folder
                settings["dicom_folder_phase"] = phase_folder
                # check if both folders have the same number of files
                if len(list_dicoms) != len(list_dicoms_phase):
                    logger.error("Number of DICOM files in mag and phase folders are different.")
                    sys.exit(1)
                logger.debug("Magnitude and phase DICOM files found.")
                logger.debug("Complex averaging on.")
                list_dicoms.sort()
                list_dicoms_phase.sort()
    if not data_type:
        # If no DICOMS, check for nii files
        included_extensions = ["nii", "nii.gz"]
        list_nii = [
            fn for fn in os.listdir(settings["dicom_folder"]) if any(fn.endswith(ext) for ext in included_extensions)
        ]
        if len(list_nii) > 0:
            data_type = "nii"
    return data_type, list_dicoms, list_dicoms_phase, list_nii
