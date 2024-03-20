"""
Python script to convert DICOM files to NIfTI + bvals + bvecs + json, HDF5, and CSV files.
"""

import glob
import math
import os
from typing import Tuple

# import pandas as pd
import pydicom
from numpy.typing import NDArray


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


def get_pixel_array(ds: pydicom.dataset.Dataset, dicom_type: int, frame_idx: int) -> NDArray:
    """
    Get the pixel array from the DICOM header

    Parameters
    ----------
    ds
    dicom_type
    frame_idx

    Returns
    -------
    pixel array

    """
    pixel_array = ds.pixel_array
    if dicom_type == 2:
        if pixel_array.ndim == 3:
            return pixel_array[frame_idx]
        elif pixel_array.ndim == 2:
            return pixel_array[:, :]
    elif dicom_type == 1:
        return pixel_array


def get_b_value(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> float:
    """
    Get b-value from a dict with the DICOM header.
    If no b-value fond, then return 0.0

    Parameters
    ----------
    c_dicom_header
    dicom_type
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
        if "DiffusionBValue" in c_dicom_header.keys():
            return c_dicom_header["DiffusionBValue"]
        else:
            return 0.0


def get_diffusion_directions(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> Tuple:
    """
    Get diffusion direction 3D vector.
    If no direction found, then return a normalised vector [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)].
    This makes sense for the STEAM because of the spoilers. For the SE if no direction, then b-value
    will be 0, and this gradient is not significant.

    Parameters
    ----------
    c_dicom_header
    dicom_type
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
            return (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))

    elif dicom_type == 1:
        if "DiffusionGradientDirection" in c_dicom_header:
            return tuple([float(i) for i in c_dicom_header["DiffusionGradientDirection"]])
        else:
            return (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))


def get_image_position(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> Tuple:
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
        val = float(
            c_dicom_header["PerFrameFunctionalGroupsSequence"][frame_idx]["CardiacSynchronizationSequence"][0][
                "RRIntervalTimeNominal"
            ]
        )
        return val

    elif dicom_type == 1:
        val = float(c_dicom_header["NominalInterval"])
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


def get_diffusion_direction_in_plane_bool(c_dicom_header: dict, dicom_type: int, frame_idx: int) -> bool:
    """
    Get boolean if the direction given is in the image plane or not.
    For the STEAM sequence the spoiler gradients of the b0 are in the image plane,
    but the standard diffusion directions are not for the SE and STEAM.

    Parameters
    ----------
    c_dicom_header
    dicom_type
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
        if "DiffusionGradientDirection" in c_dicom_header:
            return False
        else:
            return True


def get_nii_files(data_folder_path: str):
    # use the DICOM to NIFTI converter
    run_command = "dcm2niix -z y 9 -m y -b y -f 'cdti_%i_%s_%t' " + data_folder_path
    os.system(run_command)

    # move created files to subfolder
    os.makedirs(os.path.join(data_folder_path, "nii_files"), exist_ok=True)

    nii_files = glob.glob(os.path.join(data_folder_path, "*.nii.gz"))
    nii_files.sort()
    for nii_file in nii_files:
        os.rename(nii_file, os.path.join(data_folder_path, "nii_files", os.path.basename(nii_file)))

    bval_files = glob.glob(os.path.join(data_folder_path, "*.bval"))
    bval_files.sort()
    for bval_file in bval_files:
        os.rename(bval_file, os.path.join(data_folder_path, "nii_files", os.path.basename(bval_file)))

    bvec_files = glob.glob(os.path.join(data_folder_path, "*.bvec"))
    bvec_files.sort()
    for bvec_file in bvec_files:
        os.rename(bvec_file, os.path.join(data_folder_path, "nii_files", os.path.basename(bvec_file)))

    json_files = glob.glob(os.path.join(data_folder_path, "*.json"))
    json_files.sort()
    for json_file in json_files:
        os.rename(json_file, os.path.join(data_folder_path, "nii_files", os.path.basename(json_file)))


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


def get_data_from_dicoms_and_export(
    data_folder_path: str,
):
    dicom_files = glob.glob(os.path.join(data_folder_path, "*.dcm"))
    dicom_files.sort()

    get_nii_files(data_folder_path)

    # collect some header info in a dictionary from the first DICOM
    # ds = pydicom.dcmread(open(os.path.join(data_folder_path, dicom_files[0]), "rb"))

    # # check version of dicom
    # dicom_type = 0
    # if "PerFrameFunctionalGroupsSequence" in ds:
    #     dicom_type = 2
    #     # How many images in one file?
    #     n_images_per_file = len(ds.PerFrameFunctionalGroupsSequence)
    # else:
    #     dicom_type = 1
    #     n_images_per_file = 1
    #
    # # get DICOM header in a dict
    # dicom_header_fields = dictify(ds)
    #
    # # TODO need to expand this to collect many other fields
    # # collect some global header info in a dictionary
    # header_info = collect_global_header_info(dicom_header_fields, dicom_type)

    # # load sensitive fields from csv into a dataframe
    # sensitive_fields = pd.read_csv(os.path.join(settings["code_path"], "extensions", "anon_fields.csv"))
    #
    # # create a dataframe with all DICOM values
    # df = []
    # for idx, file_name in enumerate(list_dicoms):
    #     # read current DICOM
    #     ds = pydicom.dcmread(open(os.path.join(data_folder_path, file_name), "rb"))
    #     # loop over the dictionary of header fields and collect them for this DICOM file
    #     c_dicom_header = dictify(ds)
    #     # remove sensitive data
    #     field_list = sensitive_fields["sensitive_fields"].tolist()
    #     for field in field_list:
    #         if field in c_dicom_header:
    #             c_dicom_header.pop(field)
    #
    #     # loop over each frame within each file
    #     for frame_idx in range(n_images_per_file):
    #         # append values (will be a row in the dataframe)
    #         df.append(
    #             (
    #                 # file name
    #                 file_name,
    #                 # array of pixel values
    #                 get_pixel_array(ds, dicom_type, frame_idx),
    #                 # b-value or zero if not a field
    #                 get_b_value(c_dicom_header, dicom_type, frame_idx),
    #                 # diffusion directions, or [1, 1, 1] normalised if not a field
    #                 get_diffusion_directions(c_dicom_header, dicom_type, frame_idx),
    #                 # image position
    #                 get_image_position(c_dicom_header, dicom_type, frame_idx),
    #                 # nominal interval
    #                 get_nominal_interval(c_dicom_header, dicom_type, frame_idx),
    #                 # acquisition time
    #                 get_acquisition_time(c_dicom_header, dicom_type, frame_idx),
    #                 # acquisition date
    #                 get_acquisition_date(c_dicom_header, dicom_type, frame_idx),
    #                 # False if diffusion direction is a field
    #                 get_diffusion_direction_in_plane_bool(c_dicom_header, dicom_type, frame_idx),
    #                 # dictionary with header fields
    #                 c_dicom_header,
    #             )
    #         )
    # df = pd.DataFrame(
    #     df,
    #     columns=[
    #         "file_name",
    #         "image",
    #         "b_value",
    #         "direction",
    #         "image_position",
    #         "nominal_interval",
    #         "acquisition_time",
    #         "acquisition_date",
    #         "dir_in_image_plane",
    #         "header",
    #     ],
    # )
    #
    # # merge header info into info
    # info = {**info, **header_info}
    #
    # return df, info


if __name__ == "__main__":
    # Example usage
    data_folder_path = "/Users/pf/WORK/WORK_2/CIMA_scans/initial_test_scans/STEAM/diffusion_images/dicom_archive"
    get_data_from_dicoms_and_export(data_folder_path)
