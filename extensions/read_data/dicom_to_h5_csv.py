"""
Python script to convert DICOM files to HDF5 (pixel array), and CSV files with metadata information:
- global_table.csv: contains global header information
- frame_table.csv: contains header information for each frame
- pixel_array.h5: contains the pixel arrays data
"""

import copy
import glob
import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pydicom
import scipy
import yaml
from tqdm import tqdm


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


def flatten_dict(input_dict: dict, separator: str = "_", prefix: str = ""):
    """
    Flatten a multilevel dictionary.

    Parameters
    ----------
    input_dict : multilevel dictionary
    separator : separator string to use
    prefix : prefix to use

    Returns
    -------
    flattened dictionary
    """
    output_dict = {}
    for key, value in input_dict.items():
        if key == "DiffusionGradientDirection":
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
    """
    Simplify the dictionary keys by removing some common strings

    Parameters
    ----------
    c_dicom_header
    dicom_type

    Returns
    -------
    c_dicom_header

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
) -> [pd.DataFrame, dict]:
    """
    From a list of DICOM files get:
    - header information in a dataframe
    - pixel arrays from DICOM files.

    Parameters
    ----------
    dicom_files : list
        List of DICOM files
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
    get_manufacturer(dicom_header, logger)

    # read yaml file with fields to keep
    with open(os.path.join(settings["code_path"], "extensions", "read_data", "fields_to_keep.yaml"), "r") as stream:
        to_keep = yaml.safe_load(stream)

    # keep only the fields we defined in the yaml file above
    if dicom_type == "legacy":
        header_field_list = to_keep["fields_to_keep_legacy"]
    else:
        header_field_list = to_keep["fields_to_keep_enhanced"]

    # ===================================================================
    # FRAME HEADER INFO
    # ===================================================================
    # create empty dataframe to store the frame info.
    header_table = pd.DataFrame()

    header_table = read_all_dicom_files(dicom_files, dicom_type, n_images_per_file, header_field_list, header_table)

    # sort the columns alphabetically
    header_table = header_table.reindex(sorted(header_table.columns), axis=1)

    # sort the rows by acquisition date and time
    if dicom_type == 2:
        header_table.sort_values(by=["FrameContentSequence_FrameAcquisitionDateTime"], inplace=True)
    elif dicom_type == 1:
        header_table.sort_values(by=["AcquisitionDateTime"], inplace=True)

    # reset index
    header_table.reset_index(drop=True, inplace=True)

    # add an index starting at 1, in order to match the h5 image index in fiji
    header_table["fiji_index"] = header_table.index
    header_table["fiji_index"] = header_table["fiji_index"] + 1

    # rename some columns
    header_table = rename_columns(dicom_type, header_table)

    # move some columns to the start of the table for easier access to the most important columns
    header_table = reorder_columns(header_table)

    return header_table


def check_global_info(data: pd.DataFrame, info: dict, logger: logging) -> dict:
    def is_unique(s):
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    header_info = {}

    field_list = ["image_comments", "image_orientation_patient", "pixel_spacing", "slice_thickness"]

    for field in field_list:
        data["temp"] = data[field].astype(str)
        if is_unique(data["temp"]):
            header_info[field] = data[field].values[0]
        else:
            logger.error("Field " + field + " is not unique in table.")
            sys.exit()

    # merge header info into info
    info = {**info, **header_info}

    # remove temp column
    data = data.drop("temp", axis=1)

    return info, data


def scale_dicom_pixel_values(data: pd.DataFrame) -> pd.DataFrame:
    # check that RescaleSlope and RescaleIntercept columns exist
    if "RescaleSlope" in data.columns and "RescaleIntercept" in data.columns:
        # scale pixel values
        data["image"] = data["image"].apply(
            lambda x: x * data["RescaleSlope"].values[0] + data["RescaleIntercept"].values[0]
        )

    return data


def interpolate_dicom_pixel_values(
    data: pd.DataFrame, info: dict, logger: logging, image_type: str = "mag"
) -> pd.DataFrame:
    def is_unique(s):
        a = s.to_numpy()  # s.values (pandas<0.24)
        return (a[0] == a).all()

    field_list = ["Rows", "Columns"]

    for field in field_list:
        if is_unique(data[field]):
            info[field] = data[field].values[0]
        else:
            logger.error("Field " + field + " is not unique in table.")
            sys.exit()

    def interpolate_img(img, image_type):
        if image_type == "mag":
            img = scipy.ndimage.zoom(img, 2, order=3)
            # zero any negative pixels after interpolation
            img[img < 0] = 0
        elif image_type == "phase":
            # convert phase to real and imaginary before interpolating
            # TODO this needs to be a function
            img = np.pi * img / 4096
            img_real = np.cos(img)
            img_real = scipy.ndimage.zoom(img_real, 2, order=0)
            img_imag = np.sin(img)
            img_imag = scipy.ndimage.zoom(img_imag, 2, order=0)
            # revert back to the original phase values
            img = np.arctan2(img_imag, img_real)
            # TODO this needs to be a function
            img = 4096 * img / np.pi
        return img

    # if largest dimension < 192, then interpolate by a factor of 2
    if max([info["Rows"], info["Columns"]]) < 192:
        data["image"] = data["image"].apply(lambda x: interpolate_img(x, image_type))
        info["Rows"] *= 2
        info["Columns"] *= 2

    return data, info


def tweak_directions(data):
    # add new column to table to indicate if the directions are in the image plane
    data["dir_in_image_plane"] = False

    # replace [nan, nan, nan] directions with (0.0,0.0,0.0)
    data["diffusion_direction"] = data["diffusion_direction"].apply(
        lambda x: (0.0, 0.0, 0.0) if np.isnan(x).any() else tuple(x)
    )

    return data


def get_dicom_version(global_dicom_header: pydicom.dataset.Dataset, logger: logging) -> [str, int]:
    """
    Get the DICOM version:
    - legacy
    - enhanced

    Parameters
    ----------
    global_dicom_header

    Returns
    dicom_type and n_images_per_file
    -------

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


def get_manufacturer(header: pydicom.dataset.Dataset, logger: logging) -> str:
    """
    Get manufacturer from the DICOM header.

    Parameters
    ----------
    header

    Returns
    -------
    manufacturer

    """
    if "Manufacturer" in header:
        val = header["Manufacturer"].value
        if val == "Siemens Healthineers" or val == "Siemens" or val == "SIEMENS":
            # manufacturer = "siemens"
            logger.debug("Manufacturer: Siemens")
        elif val == "Philips Medical Systems" or val == "Philips":
            # manufacturer = "philips"
            logger.debug("Manufacturer: Philips")
        else:
            sys.exit("Manufacturer not supported.")
    else:
        sys.exit("Manufacturer not supported.")

    # return manufacturer


def rename_columns(dicom_type, table_frame):
    if dicom_type == "enhanced":
        table_frame = table_frame.rename(
            columns={
                "FileName": "file_name",
                "MRDiffusionSequence_DiffusionBValue": "b_value",
                "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_1": "b_vec_x",
                "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_2": "b_vec_y",
                "MRDiffusionSequence_DiffusionGradientDirectionSequence_DiffusionGradientOrientation_3": "b_vec_z",
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

        # combine the xyz of the diffusion directions in one list
        table_frame["diffusion_direction"] = table_frame[["b_vec_x", "b_vec_y", "b_vec_z"]].values.tolist()
        table_frame = table_frame.drop(columns=["b_vec_x", "b_vec_y", "b_vec_z"])

    elif dicom_type == "legacy":
        table_frame = table_frame.rename(
            columns={
                "FileName": "file_name",
                "DiffusionBValue": "b_value",
                "DiffusionGradientDirection": "diffusion_direction",
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


def reorder_columns(table_frame):
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

    table_frame = table_frame[cols_to_move + [col for col in table_frame.columns if col not in cols_to_move]]
    return table_frame


def read_all_dicom_files(
    dicom_files: list,
    dicom_type: str,
    n_images_per_file: int,
    header_field_list: list,
    header_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Read all DICOM files and extract header information to a dataframe

    Parameters
    ----------
    dicom_files
    dicom_type
    n_images_per_file
    header_field_list
    header_table

    Returns
    -------

    """
    # loop through all DICOM files
    for idx, file_name in enumerate(tqdm(dicom_files, desc="Reading DICOMs")):
        # read current DICOM
        c_dicom_header = pydicom.dcmread(open(file_name, "rb"))

        for frame_idx in range(n_images_per_file):
            # collect pixel values
            c_pixel_array = c_dicom_header.pixel_array
            if c_pixel_array.ndim == 3:
                c_pixel_array = c_pixel_array[frame_idx]

            # convert header to dictionary
            c_dicom_header_dict = dictify(c_dicom_header)
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

                # convert dict to dataframe
                new_table = pd.DataFrame.from_dict(c_dict, orient="index")

                # concatenate the new dataframe to the existing one
                header_table = pd.concat([header_table, new_table.T], axis=0, ignore_index=True)

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

                # convert dict to dataframe
                new_table = pd.DataFrame.from_dict(c_dict, orient="index")

                # concatenate the new dataframe to the existing one
                header_table = pd.concat([header_table, new_table.T], axis=0, ignore_index=True)

    return header_table


def simplify_per_frame_dictionary(c_dict: dict, frame_idx: int) -> dict:
    """
    Simplify the dictionary keys by removing some common strings

    Parameters
    ----------
    c_dict
    frame_idx

    Returns
    -------
    c_dict

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


if __name__ == "__main__":
    paths = {}

    # python script path
    abspath = os.path.abspath(sys.argv[0])
    paths["code"] = os.path.dirname(abspath)

    # data input and output paths
    paths["data_folder_path"] = os.path.abspath(sys.argv[1])
    paths["export_data_path"] = os.path.abspath(sys.argv[2])

    # list all dicom files
    dicom_files = glob.glob(os.path.join(paths["data_folder_path"], "*.dcm"))
    dicom_files.sort()

    info = {}

    # get header_table and pixel_arrays from DICOMs
    header_table, pixel_arrays = get_data_from_dicoms(dicom_files)

    # create folder if it does not exist
    if not os.path.exists(paths["export_data_path"]):
        os.makedirs(paths["export_data_path"])

    # export to csv header table
    header_table.to_csv(os.path.join(paths["export_data_path"], "header_table.csv"))

    # export pixel array to HDF5
    with h5py.File(os.path.join(paths["export_data_path"], "pixel_array.h5"), "w") as hf:
        hf.create_dataset("pixel_array", data=pixel_arrays)

    print("Done.")