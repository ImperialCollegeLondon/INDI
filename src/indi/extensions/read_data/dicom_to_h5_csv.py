"""
Python script to convert DICOM files to HDF5 (pixel array), and CSV files with metadata information:
- global_table.csv: contains global header information
- frame_table.csv: contains header information for each frame
- pixel_array.h5: contains the pixel arrays data
"""

import copy
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import pydicom
import scipy
import yaml
from tqdm import tqdm

from indi.extensions.extensions import mag_to_rad, rad_to_mag


# get DICOM header fields
def dictify(ds: pydicom.dataset.Dataset, manufacturer: str, dicom_type: str) -> dict:
    """Turn a pydicom Dataset into a dict with keys derived from the Element tags.
    Private info is not collected, because we cannot access it with the keyword.
    So we need to manually fish the diffusion information in the old DICOMs.

    Args:
        ds: The Dataset to dictify
        manufacturer: Manufacturer of the DICOM files (siemens, philips, ge, uih)
        dicom_type: DICOM type (legacy or enhanced)

    Returns:
        output: A dictionary with the DICOM header information

    """

    output = dict()
    # iterate over all non private fields
    for elem in ds:
        if elem.VR != "SQ":
            output[elem.keyword] = elem.value
        else:
            output[elem.keyword] = [dictify(item, manufacturer, dicom_type) for item in elem]

    # add manually private diffusion fields if they exist for legacy DICOMs
    if dicom_type == "legacy":
        if manufacturer == "siemens":
            if [0x0019, 0x100C] in ds:
                output["b_value"] = ds[0x0019, 0x100C].value
            if [0x0019, 0x100E] in ds:
                output["diffusion_direction"] = ds[0x0019, 0x100E].value

        if manufacturer == "philips":
            if [0x0018, 0x9087] in ds:
                output["b_value"] = ds[0x0018, 0x9087].value
            if [0x0018, 0x9089] in ds:
                output["diffusion_direction"] = ds[0x0018, 0x9089].value

        if manufacturer == "ge":
            if [0x0018, 0x9087] in ds:
                output["b_value"] = ds[0x0018, 0x9087].value
            if [0x0019, 0x10BB] in ds and [0x0019, 0x10BC] in ds and [0x0019, 0x10BD] in ds:
                output["diffusion_direction"] = [
                    ds[0x0019, 0x10BB].value,
                    ds[0x0019, 0x10BC].value,
                    ds[0x0019, 0x10BD].value,
                ]
                # convert list of strings to list of floats
                output["diffusion_direction"] = [float(i) for i in output["diffusion_direction"]]

        if manufacturer == "uih":
            # I was told by UIH team that the real DiffusionBValue is in the following tag [0x0065, 0x1009].
            # There is also the tag DiffusionBValue [0x0018, 0x9087], but this one seems to have approximate
            # b-values. So I am using the first one:
            if [0x0065, 0x1009] in ds:
                output["b_value"] = ds[0x0065, 0x1009].value
            if [0x0018, 0x9089] in ds:
                output["diffusion_direction"] = ds[0x0018, 0x9089].value

            # I was also told by UIH team that the DiffusionGradientDirection is in the following
            # tag [0x0065, 0x1037] and the directions are in the image coordinate system.
            # But the header already contains another field called DiffusionGradientOrientation,
            # so I am using that one instead, which seems to be in the magnetic coordinate system.
            # if [0x0065, 0x1037] in ds:
            #     output["DiffusionGradientDirection"] = ds[0x0065, 0x1037].value

    return output


def flatten_dict(input_dict: dict, separator: str = "_", prefix: str = ""):
    """Flatten a multilevel dictionary.

    Args:
      input_dict: multilevel dictionary
      separator: separator string to use
      prefix: prefix to use

    Returns:
        output_dict: flattened dictionary

    """
    output_dict = {}
    for key, value in input_dict.items():
        if key == "diffusion_direction":
            output_dict[key] = value
        elif isinstance(value, dict) and value:
            deeper = flatten_dict(value, separator, prefix + key + separator)
            output_dict.update({key2: val2 for key2, val2 in deeper.items()})
        elif isinstance(value, list) and value:
            for index, sublist in enumerate(value, start=1):
                if isinstance(sublist, dict) and sublist:
                    deeper = flatten_dict(
                        sublist,
                        separator,
                        prefix + key + separator + str(index) + separator,
                    )
                    output_dict.update({key2: val2 for key2, val2 in deeper.items()})
                else:
                    output_dict[prefix + key + separator + str(index)] = sublist
        else:
            output_dict[prefix + key] = value
    return output_dict


def simplify_global_dict(c_dicom_header: dict, dicom_type: str) -> dict:
    """Simplify the dictionary keys by removing some common strings

    Args:
      c_dicom_header: input DICOM header dictionary
      dicom_type: DICOM type (legacy or enhanced)

    Returns:
        c_dicom_header: simplified DICOM header dictionary

    """
    if dicom_type == "legacy":
        pass
    elif dicom_type == "enhanced":
        c_dicom_header = {
            k.replace(
                ("SharedFunctionalGroupsSequence_1_"),
                "",
            ): v
            for k, v in c_dicom_header.items()
        }
        c_dicom_header = {
            k.replace(
                ("_1_"),
                "_",
            ): v
            for k, v in c_dicom_header.items()
        }

    return c_dicom_header


def get_data_from_dicoms(
    dicom_files: list, settings: dict, logger: logging.Logger, image_type: str = "mag"
) -> pd.DataFrame:
    """From a list of DICOM files get:

    - header information in a dataframe
    - pixel arrays from DICOM files.

    Args:
      dicom_files: List of DICOM files
      settings: Settings dictionary
      logger: Logger
      image_type: Image type, either "mag" or "phase"

    Returns:
        header_table: DataFrame with header information
        manufacturer: Manufacturer of the DICOM files (siemens, philips, ge)

    """

    # get full paths of the DICOM files
    if image_type == "mag":
        data_folder_path = settings["dicom_folder"]
        logger.debug("Magnitude DICOMs")
    elif image_type == "phase":
        data_folder_path = settings["dicom_folder_phase"]
        logger.debug("Phase DICOMs")
    else:
        sys.exit("Image type not supported.")
    dicom_files = [os.path.join(data_folder_path, f) for f in dicom_files]

    # ===================================================================
    # Check DICOMs
    # ===================================================================
    # collect some header info in a dictionary from the first DICOM
    dicom_header = pydicom.dcmread(open(dicom_files[0], "rb"))

    # check version of dicom
    dicom_type, n_images_per_file = get_dicom_version(dicom_header, logger)

    # get manufacturer
    manufacturer = get_manufacturer(dicom_header, logger)

    # read yaml file with fields to keep
    with open(os.path.join(os.path.dirname(__file__), "fields_to_keep.yaml"), "r") as stream:
        to_keep = yaml.safe_load(stream)

    # keep only the fields we defined in the yaml file above
    if dicom_type == "legacy":
        header_field_list = to_keep["fields_to_keep_legacy"]
    else:
        header_field_list = to_keep["fields_to_keep_enhanced"]

    # ===================================================================
    # FRAME HEADER INFO
    # ===================================================================
    header_table = read_all_dicom_files(dicom_files, dicom_type, n_images_per_file, header_field_list, manufacturer)

    # sort the columns alphabetically
    header_table = header_table.reindex(sorted(header_table.columns), axis=1)

    # # sort the rows by acquisition date and time
    # if dicom_type == "enhanced":
    #     header_table.sort_values(by=["FrameContentSequence_FrameAcquisitionDateTime"], inplace=True)
    # elif dicom_type == "legacy":
    #     header_table.sort_values(by=["AcquisitionDateTime"], inplace=True)

    # reset index
    header_table.reset_index(drop=True, inplace=True)

    # add an index starting at 1, in order to match the h5 image index in fiji
    header_table["fiji_index"] = header_table.index
    header_table["fiji_index"] = header_table["fiji_index"] + 1

    # rename some columns
    header_table = rename_columns(dicom_type, header_table)

    # build the gradient directions and b-matrix columns with all the values
    if dicom_type == "enhanced":
        # check that diffusion gradient direction is present in the header
        if (
            "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_1"
            in header_table.columns
        ):
            header_table = build_gradient_directions(header_table, logger)

        # check the bmatrix is present in the header
        if "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueXX" in header_table.columns:
            header_table = build_bmatrix(header_table, logger)

    # move some columns to the start of the table for easier access to the most important columns
    header_table = reorder_columns(header_table)

    return header_table, manufacturer


def build_bmatrix(data: pd.DataFrame, logger: logging):
    """
    Build the bmatrix from the DICOM header.

    Parameters
    ----------
    data
    logger

    Returns
    -------
    data
    """

    bmatrix_columns = [
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueXX",
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueXY",
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueXZ",
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueYY",
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueYZ",
        "MRDiffusionSequence_DiffusionBMatrixSequence_DiffusionBValueZZ",
    ]
    bmatrix_vals = data[bmatrix_columns].values
    bmatrix = np.zeros((len(bmatrix_vals), 3, 3))
    bmatrix[:, 0, 0] = bmatrix_vals[:, 0]
    bmatrix[:, 0, 1] = bmatrix_vals[:, 1]
    bmatrix[:, 0, 2] = bmatrix_vals[:, 2]
    bmatrix[:, 1, 0] = bmatrix_vals[:, 1]
    bmatrix[:, 1, 1] = bmatrix_vals[:, 3]
    bmatrix[:, 1, 2] = bmatrix_vals[:, 4]
    bmatrix[:, 2, 0] = bmatrix_vals[:, 2]
    bmatrix[:, 2, 1] = bmatrix_vals[:, 4]
    bmatrix[:, 2, 2] = bmatrix_vals[:, 5]

    data["bmatrix"] = bmatrix.tolist()
    data["bmatrix"] = data["bmatrix"].apply(lambda x: np.asarray(x))
    return data


def build_gradient_directions(data: pd.DataFrame, logger: logging):
    """
    Build the gradient directions from the DICOM header.

    Args:
        data: DataFrame with DICOM header information
        logger: Logger for error messages

    Returns:
        data: DataFrame with updated gradient directions
    """

    direction_columns = [
        "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_1",
        "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_2",
        "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_3",
    ]
    direction_vals = data[direction_columns].values
    directions = np.zeros((len(direction_vals), 3))
    directions[:, 0] = direction_vals[:, 0]
    directions[:, 1] = direction_vals[:, 1]
    directions[:, 2] = direction_vals[:, 2]

    data["diffusion_direction"] = directions.tolist()
    return data


def check_global_info(data: pd.DataFrame, info: dict, logger: logging) -> tuple[dict, pd.DataFrame]:
    """Check that some columns are unique in the table and merge them into the info dictionary.

    Args:
      data: Image data
      info: Info dictionary
      logger: Logger

    Returns:
      info: Info dictionary with merged header information
      data: DataFrame with header information

    """

    def is_unique(s):
        a = s.to_numpy()
        return (a[0] == a).all()

    header_info = {}

    field_list = ["image_comments", "image_orientation_patient", "pixel_spacing", "slice_thickness"]
    # remove fields that are not present
    field_list = [field for field in field_list if field in data.columns]

    for field in field_list:
        data["temp"] = data[field].astype(str)
        if is_unique(data["temp"]):
            header_info[field] = data[field].values[0]
        else:
            if field == "image_orientation_patient":
                # check if different values are different just in rounding errors
                decimal_places = 3
                unique_vals = data["temp"].unique()

                rows = []
                for val in unique_vals:
                    temp = json.loads(val)
                    temp = [f"{i:.{decimal_places}f}" for i in temp]  # noqa
                    temp = ["0" if float(x) == 0 else x for x in temp]
                    rows.append(temp)

                def equalLists(lists):
                    return not lists or all(lists[0] == b for b in lists[1:])

                if equalLists(rows):
                    header_info[field] = data[field].values[0]
                else:
                    logger.error("Field " + field + " is not unique in table.")
                    raise ValueError("Error: Field " + field + " is not unique in table.")
            elif field == "image_comments":
                strings = data[field].values
                rr_int_values = []
                real_b0_values = []
                for text in strings:
                    m = re.findall(r"[-+]?(?:\d*\.*\d+)", text)
                    m = [float(m) for m in m]
                    if len(m) > 2:
                        rr_int_values.append(np.nan)
                        real_b0_values.append(np.nan)
                    if len(m) == 2:
                        rr_int_values.append(m[1])
                        real_b0_values.append(m[0])
                    elif len(m) == 1:
                        rr_int_values.append(np.nan)
                        real_b0_values.append(m[0])
                    else:
                        rr_int_values.append(np.nan)
                        real_b0_values.append(np.nan)

                # round the numbers to integer and check if unique
                rr_int_values = [int(a) for a in rr_int_values if not np.isnan(a)]
                real_b0_values = [int(a) for a in real_b0_values if not np.isnan(a)]

                def equalLists(lists):
                    return not lists or all(lists[0] == b for b in lists[1:])

                if equalLists(rr_int_values) and equalLists(real_b0_values):
                    header_info[field] = data[field].values[0]
                else:
                    logger.error("Field " + field + " is not unique in table.")
                    raise ValueError("Error: Field " + field + " is not unique in table.")

            else:
                logger.error("Field " + field + " is not unique in table.")
                raise ValueError("Error: Field " + field + " is not unique in table.")

    # merge header info into info
    info = {**info, **header_info}

    # remove temp column
    data = data.drop("temp", axis=1)

    return info, data


def scale_dicom_pixel_values(data: pd.DataFrame) -> pd.DataFrame:
    """Scale pixel values using RescaleSlope and RescaleIntercept columns if they exist in the header.

    Args:
        data: DataFrame with image data

    Returns:
        data: DataFrame with scaled pixel values

    """
    # check that RescaleSlope and RescaleIntercept columns exist
    if "RescaleSlope" in data.columns and "RescaleIntercept" in data.columns:
        # scale pixel values
        data["image"] = data["image"].apply(
            lambda x: x * data["RescaleSlope"].values[0] + data["RescaleIntercept"].values[0]
        )

    return data


def check_rows_and_columns(data: pd.DataFrame, info: dict, logger: logging) -> dict:
    """Check that Rows and Columns fields are unique in the table and merge them into the info dictionary.
    Args:
      data: DataFrame with image data
      info: Info dictionary
      logger: Logger
    Returns:
      info: Info dictionary with merged Rows and Columns values
    """

    def is_unique(s):
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    field_list = ["Rows", "Columns"]

    for field in field_list:
        if is_unique(data[field]):
            info[field] = data[field].values[0]
        else:
            logger.error("Field " + field + " is not unique in table.")
            raise ValueError("Error: Field " + field + " is not unique in table.")

    return info


def interpolate_dicom_pixel_values(
    data: pd.DataFrame, info: dict, logger: logging, image_type: str = "mag", factor: int = 1
) -> [pd.DataFrame, dict]:
    """Interpolate pixel matrix

    Args:
      data: DataFrame with image data
      info: Info dictionary
      logger: Logger
      image_type: Image type, either "mag" or "phase"
      factor: Interpolation factor, default is 1 (no interpolation)

    Returns:
      data: DataFrame with interpolated pixel values
      info: Info dictionary with updated Rows and Columns values

    """

    def interpolate_img(img, image_type):
        if image_type == "mag":
            img = scipy.ndimage.zoom(img, factor, order=3)
            # zero any negative pixels after interpolation
            img[img < 0] = 0
        elif image_type == "phase":
            # convert phase to real and imaginary before interpolating
            img = mag_to_rad(img)
            img_real = np.cos(img)
            img_real = scipy.ndimage.zoom(img_real, factor, order=0)
            img_imag = np.sin(img)
            img_imag = scipy.ndimage.zoom(img_imag, factor, order=0)
            # revert back to the original phase values
            img = np.arctan2(img_imag, img_real)
            img = rad_to_mag(img)
        return img

    data["image"] = data["image"].apply(lambda x: interpolate_img(x, image_type))
    old_rows_column = [info["Rows"], info["Columns"]]
    info["Rows"] *= factor
    info["Columns"] *= factor

    logger.debug(
        f"Image matrix interpolated from {old_rows_column[0]} x {old_rows_column[1]} to {info["Rows"]} x {info["Columns"]}."
    )

    return data, info


def tweak_directions_and_b_values(data: pd.DataFrame) -> pd.DataFrame:
    """Tweak the directions in the table. If a direction is a list of NaNs, then
    change to a null vector (0,0,0).
    If the b-value is NaN, set it to 0.

    Args:
      data: DataFrame with image data

    Returns:
        data: DataFrame with tweaked directions and b-values

    """
    # add new column to table to indicate if the directions are in the image plane
    data["dir_in_image_plane"] = False

    # replace [nan, nan, nan] directions with (0.0,0.0,0.0)
    data["diffusion_direction"] = data["diffusion_direction"].apply(
        lambda x: (0.0, 0.0, 0.0) if np.isnan(x).any() else tuple(x)
    )

    # repeat for the b-matrix
    if "bmatrix" in data.columns:
        data["bmatrix"] = data["bmatrix"].apply(lambda x: np.zeros((3, 3)) if np.isnan(x).any() else x)

    # if missing b-value, set it to 0
    if "b_value" in data.columns:
        data["b_value"] = data["b_value"].apply(lambda x: 0.0 if np.isnan(x) else x)

    return data


def add_missing_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add missing required columns to the table.
    More columns can be added here if needed.

    Args:
      data: DataFrame with image data

    Returns:
        data: DataFrame with added columns

    """
    list_of_fields = ["series_description"]

    for field in list_of_fields:
        if field not in data.columns:
            data[field] = None

    return data


def get_dicom_version(global_dicom_header: pydicom.dataset.Dataset, logger: logging) -> [str, int]:
    """Get the DICOM version:
    - legacy
    - enhanced

    Args:
      global_dicom_header: pydicom.dataset.Dataset:
      logger: logger

    Returns:
      dicom_type: either "legacy" or "enhanced"
      n_images_per_file: number of images per DICOM file

    """
    dicom_type = None
    if "PerFrameFunctionalGroupsSequence" in global_dicom_header:
        dicom_type = "enhanced"
        logger.debug("DICOM type: Enhanced")
        # How many images in one file?
        n_images_per_file = len(global_dicom_header.PerFrameFunctionalGroupsSequence)
        logger.debug("Number of images per DICOM: " + str(n_images_per_file))
    else:
        dicom_type = "legacy"
        logger.debug("DICOM type: Legacy")
        n_images_per_file = 1
        logger.debug("Number of images per DICOM: " + str(n_images_per_file))

    return dicom_type, n_images_per_file


def get_manufacturer(header: pydicom.dataset.Dataset, logger: logging):
    """Get manufacturer from the DICOM header.
    This function will set the manufacturer variable to one of the following:
    - "siemens"
    - "philips"
    - "ge"

    Args:
      header: Dataset header
      logger: logger

    Returns:
        manufacturer: string with the manufacturer name

    """
    if "Manufacturer" in header:
        val = header["Manufacturer"].value
        if val == "Siemens Healthineers" or val == "Siemens" or val == "SIEMENS":
            manufacturer = "siemens"
            logger.debug("Manufacturer: Siemens")
        elif val == "Philips Medical Systems" or val == "Philips":
            manufacturer = "philips"
            logger.debug("Manufacturer: Philips")
        elif val == "GE MEDICAL SYSTEMS" or val == "GE":
            manufacturer = "ge"
            logger.debug("Manufacturer: GE")
        elif val == "UIH" or val == "United Imaging Healthcare":
            manufacturer = "uih"
            logger.debug("Manufacturer: United Imaging Healthcare")
        else:
            raise ValueError("Manufacturer not supported.")
    else:
        raise ValueError("Manufacturer field not found in header.")

    return manufacturer


def rename_columns(dicom_type: str, table_frame: pd.DataFrame) -> pd.DataFrame:
    """Rename important columns in the table.
    This will also have the effect of naming some columns the same string for both
    legacy and enhanced DICOMs.

    Args:
      dicom_type: either "legacy" or "enhanced"
      table_frame: dataframe with image data

    Returns:
        table_frame: dataframe with renamed columns

    """
    if dicom_type == "enhanced":
        table_frame = table_frame.rename(
            columns={
                "FileName": "file_name",
                "MRDiffusionSequence_DiffusionBValue": "b_value",
                "PlanePositionSequence_ImagePositionPatient": "image_position",
                "PlaneOrientationSequence_ImageOrientationPatient": "image_orientation_patient",
                "CardiacSynchronizationSequence_RRIntervalTimeNominal": "nominal_interval",
                "FrameContentSequence_FrameAcquisitionDateTime": "acquisition_date_time",
                "SeriesDescription": "series_description",
                "SeriesNumber": "series_number",
                "ImageComments": "image_comments",
                "PixelMeasuresSequence_PixelSpacing": "pixel_spacing",
                "PixelMeasuresSequence_SliceThickness": "slice_thickness",
                "PixelValueTransformationSequence_RescaleSlope": "RescaleSlope",
                "PixelValueTransformationSequence_RescaleIntercept": "RescaleIntercept",
                "PixelValueTransformationSequence_RescaleType": "RescaleType",
            }
        )

    elif dicom_type == "legacy":
        table_frame = table_frame.rename(
            columns={
                "FileName": "file_name",
                "ImagePositionPatient": "image_position",
                "ImageOrientationPatient": "image_orientation_patient",
                "NominalInterval": "nominal_interval",
                "AcquisitionDateTime": "acquisition_date_time",
                "SeriesDescription": "series_description",
                "SeriesNumber": "series_number",
                "ImageComments": "image_comments",
                "PixelSpacing": "pixel_spacing",
                "SliceThickness": "slice_thickness",
            }
        )

    return table_frame


def reorder_columns(table_frame: pd.DataFrame) -> pd.DataFrame:
    """Move some columns to the start of the table for easier access to the most important columns.

    Args:
      table_frame: dataframe with image data

    Returns:
        table_frame: dataframe with reordered columns

    """

    cols_to_move = [
        "fiji_index",
        "file_name",
        "series_number",
        "series_description",
        "image_position",
        "acquisition_date_time",
        "b_value",
        "diffusion_direction",
        "nominal_interval",
        "image_comments",
        "image_orientation_patient",
    ]

    # make sure the elements of the list above exist, otherwise remove them from the list
    cols_to_move = [col for col in cols_to_move if col in table_frame.columns]
    table_frame = table_frame[cols_to_move + [col for col in table_frame.columns if col not in cols_to_move]]

    return table_frame


def read_all_dicom_files(
    dicom_files: list,
    dicom_type: str,
    n_images_per_file: int,
    header_field_list: list,
    manufacturer: str,
) -> pd.DataFrame:
    """Read all DICOM files and extract header information to a dataframe

    Args:
        dicom_files: list of DICOM files
        dicom_type: DICOM type (legacy or enhanced)
        n_images_per_file: number of images per DICOM file
        header_field_list: list of fields to keep in the header
        manufacturer: manufacturer of the DICOM files (siemens, philips, ge)

    Returns:
        header_table: DataFrame with header information

    """
    # loop through all DICOM files
    list_of_dictionaries = []
    for idx, file_name in enumerate(tqdm(dicom_files, desc="Reading DICOMs")):
        # read current DICOM
        c_dicom_header = pydicom.dcmread(open(file_name, "rb"))

        for frame_idx in range(n_images_per_file):
            # collect pixel values
            c_pixel_array = c_dicom_header.pixel_array
            if c_pixel_array.ndim == 3:
                c_pixel_array = c_pixel_array[frame_idx]

            # convert header to dictionary
            c_dicom_header_dict = dictify(c_dicom_header, manufacturer, dicom_type)
            # remove pixel data
            c_dicom_header_dict.pop("PixelData")
            # flatten dictionary
            c_dicom_header_dict = flatten_dict(c_dicom_header_dict)

            # fields to keep are defined in the yaml file
            c_dict_general = {key: c_dicom_header_dict[key] for key in header_field_list if key in c_dicom_header_dict}

            # simplify some keys that are very long
            c_dict_general = simplify_global_dict(c_dict_general, dicom_type)

            # ====================================
            # legacy dicom format
            # ====================================
            if dicom_type == "legacy":
                # add filename to the current dictionary
                c_dict_general["FileName"] = os.path.basename(file_name)

                # add pixel array to the current dictionary
                c_dict_general["image"] = c_pixel_array

                # if dictionary does not have AcquisitionDateTime key, add it manually
                if "AcquisitionDateTime" not in c_dict_general:
                    c_dict_general["AcquisitionDateTime"] = (
                        c_dict_general["AcquisitionDate"] + c_dicom_header_dict["AcquisitionTime"]
                    )

                c_dict = copy.deepcopy(c_dict_general)

                list_of_dictionaries.append(c_dict)

            # ====================================
            # enhanced dicom format
            # ====================================
            if dicom_type == "enhanced":
                # keep only info from the PerFrameFunctionalGroupsSequence
                for k in list(c_dicom_header_dict.keys()):
                    if not k.startswith("PerFrameFunctionalGroupsSequence"):
                        del c_dicom_header_dict[k]

                # copy the header above with PerFrameFunctionalGroupsSequence
                c_dict = copy.deepcopy(c_dicom_header_dict)

                # keep only the part corresponding to the current image
                for k in list(c_dict.keys()):
                    if not k.startswith("PerFrameFunctionalGroupsSequence_" + str(frame_idx + 1) + "_"):
                        del c_dict[k]

                # simplify the dictionary keys
                c_dict = simplify_per_frame_dictionary(c_dict, frame_idx)

                # add filename to the current dictionary
                c_dict["FileName"] = os.path.basename(file_name)

                # add pixel array to the current dictionary
                c_dict["image"] = c_pixel_array

                # combine the two dictionaries
                # (PerFrameFunctionalGroupsSequence and some
                # fields from the general one)
                c_dict = {**c_dict_general, **c_dict}

                list_of_dictionaries.append(c_dict)

    # create dataframe from list_of_dictionaries
    header_table = pd.DataFrame(list_of_dictionaries)

    return header_table


def simplify_per_frame_dictionary(c_dict: dict, frame_idx: int) -> dict:
    """Simplify the dictionary keys by removing some recurrent strings

    Args:
      c_dict: input dictionary
      frame_idx: frame index

    Returns:
        c_dict: simplified dictionary

    """
    c_dict = {
        k.replace(
            ("PerFrameFunctionalGroupsSequence_" + str(frame_idx + 1) + "_"),
            "",
        ): v
        for k, v in c_dict.items()
    }
    # _1_ -> _
    c_dict = {
        k.replace(
            "_1_",
            "_",
        ): v
        for k, v in c_dict.items()
    }

    # remove the key "_"
    c_dict.pop("_", None)

    # remove the key "PerFrameFunctionalGroupsSequence_2__1_"
    c_dict.pop("PerFrameFunctionalGroupsSequence_2__1_", None)

    return c_dict
