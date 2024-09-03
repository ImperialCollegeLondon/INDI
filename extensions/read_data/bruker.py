import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from brukerapi.dataset import Dataset
from tqdm import tqdm

BRUKER_NAMES_CORRESPONDENCE_DICT = {
    "FG_DIFFUSION": "diffusion_direction",
    "FG_SLICE": "slice",
    "FG_CYCLE": "repetition",
}


def construct_pixel_spacing(file_contents: str) -> list:
    """Computes the pixel spacing for a Bruker dataset

    Args:
        file_contents: content of visu_pars file as string

    Returns:
        pixel_spacing value
    """
    visu_core_extent_string = get_relevant_substring(file_contents, "pixel_spacing_core_extent")
    visu_core_extent_elements = get_relevant_elements_from_substring(
        visu_core_extent_string, "pixel_spacing_core_extent"
    )

    visu_core_size_string = get_relevant_substring(file_contents, "pixel_spacing_core_size")
    visu_core_size_elements = get_relevant_elements_from_substring(visu_core_size_string, "pixel_spacing_core_extent")

    pixel_spacing = [float(i) / float(j) for i, j in zip(visu_core_extent_elements, visu_core_size_elements)]
    return pixel_spacing


def remove_endline_characters(file_contents: str) -> str:
    """Removes endline characters from a string (\n and \r)

    Args:
        file_contents: content of a file as string

    Returns:
        the file content as string with no endline characters
    """
    return file_contents.replace("\r", " ").replace("\n", " ")


def remove_multiple_white_spaces(file_contents: str) -> str:
    """Removes multiple white spaces from a string

    Args:
        file_contents: content of a file as string

    Returns:
        the file content as string with single whitespaces

    """
    return re.sub(r"\s\s+", " ", file_contents)


def get_relevant_substring(file_contents: str, retrieval_parameter: str) -> str:
    """Returns raw substring of interest from file_contents

    Args:
        file_contents: the content of the file as a string
        retrieval_parameter: string denoting the retrieval parameter, it
            must be one the keys of options dictionary
    Returns:
        the relevant portion of the string
    """

    options = {
        "order": "##$VisuFGOrderDesc=(",
        "fgelem": "##$VisuFGElemComment=(",
        "slice_distance": "##$VisuCoreSlicePacksSliceDist=(",
        "slice_number": "##$VisuCoreSlicePacksSlices=(",
        "pixel_spacing_core_extent": "##$VisuCoreExtent=",
        "pixel_spacing_core_size": "##$VisuCoreSize=",
        "diff_dir": "##$PVM_DwGradVec=(",
        "diff_grad_orient": "##$PVM_SPackArrGradOrient=(",
        "transposition": "##$RECO_transposition=(",
    }

    if retrieval_parameter in options:
        start_pos = file_contents.find(options[retrieval_parameter])

    else:
        raise ValueError("Invalid retrieval parameter")

    if start_pos == -1:
        return ""

    end_pos = file_contents.find("##$", start_pos + 1)
    retrieval_parameter_string = file_contents[start_pos:end_pos]
    return retrieval_parameter_string


def get_relevant_elements_from_substring(substring: str, retrieval_parameter: str) -> list:
    """Returns the elements of interest from the already modified substring

    Args:
        substring: the return value of get_relevant_substring method
        retrieval_parameter: string denoting the retrieval parameter

    Returns:
        the relevant elements for the corresponding input configuration
    """
    if retrieval_parameter == "order":
        elems = re.split(r"\(*\)", substring)[1:-1]
        elems = [elem.strip()[1:] for elem in elems]

    elif retrieval_parameter == "fgelem":
        first_paranthesis_pos = substring.find("<")
        elems = re.split("<*>", substring[first_paranthesis_pos:])[:-1]
        elems = [elem.strip()[1:] for elem in elems]

    elif retrieval_parameter == "slice_distance":
        elems = float(substring.split(")")[1].strip())

    elif retrieval_parameter == "slice_number":
        elems = int(substring.split(", ")[1].strip(") "))

    elif retrieval_parameter == "image_orientation_patient":
        substring = substring.split(")")[1].strip()
        temp_elems = np.reshape(list(map(float, substring.split(" ")))[:9], (3, 3))
        elems = temp_elems.tolist()

    elif retrieval_parameter in [
        "pixel_spacing_core_size",
        "pixel_spacing_core_extent",
    ]:
        elems = substring.split(")")[1].strip().split(" ")

    # method file
    elif retrieval_parameter == "diff_dir":
        elems = substring.split(")")[1].strip().split(" ")

    elif retrieval_parameter == "diff_grad_orient":
        elems = substring.split(")")[1].strip().split(" ")

    # reco file
    elif retrieval_parameter == "transposition":
        elems = substring.split(")")[1].strip().split(" ")
        elems = [int(item) for item in elems]

        if sum(elems) == 0:
            elems = 0
        elif sum(elems) == len(elems):
            elems = 1
        else:
            raise ValueError("Transposition elemens from reco seem to differ from all 0s or all 1s!")

    else:
        raise ValueError("Invalid retrieval parameter")

    return elems


def construct_order_dictionary(visu_order_desc_string_elems: list, correspondence_dict: dict) -> dict:
    """Constructs the order dictionary.
    The order dictionary contains as keys elements from the list: slice, diffusion,
    repetition and as values the corresponding number of elements.

    The order in which the keys appear in the dictionary dictates the order in which the
    construction of the final lists of diffusion parameters for the entire set of images
    is made.

    e.g.
    order = {slice:70, diffusion:150, repetition:6} -> 70*15*06 = 63000 total images
    This means the the scan has been made slice-first, then diffusion, then repetition.
    In this case, the final diffusion parameters list will be:
    [b1,b2,...b150] -> [b1,b1,b1..b1 (70 times) b2 x 70 ..... b150 x 70,b1 x 70,...]
    (each bn x 70 will be repeated 6 times)
    [d1,d2,...d70] -> [d1 x 900, d2 x 900, ..., d70 x 900 ]

    Args:
        visu_order_desc_string_elems:
            elements from the visu_order_desc_string tag in visu_pars file
        correspondence_dict: the correspondence dictionary

    Returns:
        The order dictionary

    """
    order = {}
    for elem in visu_order_desc_string_elems:
        elem = elem.split(",")
        for name in correspondence_dict:
            if name in elem[1]:
                order[correspondence_dict[name]] = int(elem[0])
    return order


def construct_b_values_list(no_diff_dirs: int, visu_fg_elem_comment_elems: list) -> Tuple[list, int]:
    """Creates the b_values list from the relevant Bruker file

    Args:
        no_diff_dirs: number of diffusion directions
        visu_fg_elem_comment_elems:
            elements from the visu_fg_elem_comment tag in visu_pars file

    Returns:
        list of b_values
        number of A0 images
    """
    b_vals = []
    number_of_a0_images = 0
    for i in range(no_diff_dirs):
        elem = visu_fg_elem_comment_elems[i]
        elem = elem.split(" ")
        if "A0" == elem[0]:
            number_of_a0_images += 1
        b_vals.append(int(elem[3]))
    return b_vals, number_of_a0_images


def construct_diffusion_directions_from_method_file(
    no_diff_dirs: int, method_diffusion_gradient_comment_elems: list
) -> list:
    """Constructs the diffusion directions list from the method file

    Args:
        no_diff_dirs: total number of diffusion directions
        visu_acq_diffusion_gradient_comment_elems: the elements from corresponding to
            the diffusion gradient tag in the method file

    Returns:
        the list of diffusion directions
    """
    diff_dirs = []
    for i in range(0, no_diff_dirs * 3, 3):
        current_diff_config = (
            float(method_diffusion_gradient_comment_elems[i]),
            float(method_diffusion_gradient_comment_elems[i + 1]),
            float(method_diffusion_gradient_comment_elems[i + 2]),
        )

        # # normalise directions
        # diff_dirs_ = np.array(current_diff_config)
        # diff_dirs_ /= np.maximum(1e-20, np.linalg.norm(diff_dirs_))
        # current_diff_config = tuple(diff_dirs_.tolist())
        diff_dirs.append(current_diff_config)

    return diff_dirs


def construct_diffusion_directions(
    method_diff_grad_comment_string: str,
    folder_path: Path,
    number_of_a0_images: int,
    order: dict,
) -> list:
    """Aggregates the two construct diffusion direction methods

    Args:
        visu_acq_diff_grad_comment_string: the relevant substring for diffusion
            gradient, obtained using get_relevant_substring method
        folder_path: folder path where method file lays
        number_of_a0_images: number of diffusion free images
        order: order dictionary

    Returns:
        the diffusion directions list
    """

    method_diff_grad_comment_elems = get_relevant_elements_from_substring(method_diff_grad_comment_string, "diff_dir")

    diff_dirs = construct_diffusion_directions_from_method_file(
        order["diffusion_direction"], method_diff_grad_comment_elems
    )

    return diff_dirs


def get_reconstructed_images(dataset: Dataset) -> list:
    """Retrieves the images from the Bruker dataset

    Args:
        dataset: the Dataset constructed from the 2dseq file

    Returns:
        the list of reconstructed images

    """
    reconstructed_images_pixels_arrays = []
    data = dataset.data.reshape(dataset.data.shape[0], dataset.data.shape[1], -1)  # keep only the first 2 dimensions
    reconstructed_images_pixels_arrays.extend(
        [np.rot90(np.flipud(data[:, :, index]), -1) for index in range(data.shape[2])]
    )
    return reconstructed_images_pixels_arrays


def convert_rps_to_img(
    diff_dirs: list,
    image_orientation_patient: list,
    transposition: int,
) -> list:
    """In our Bruker scans, the diffusion directions from the method file are
    given in the (read, phase, slice) coordinates. We need to convert these
    directions to the image coordinates as in the DICOM standard.

    Args:
        diff_dirs: list of diffusion directions (RPS coordinates)
        image_orientation_patient: acquisition gradient matrix
        transposition: flag that defines which transposition matrix

    Returns:
        diff_dirs: list of diffusion directions (image coordinates)
    """

    # Acquisition gradient matrix, transforms ParaVision patient
    # coordinate system into r/p/s
    A = np.array(image_orientation_patient)

    # Transposition Matrix (Read-Phase swap)
    if transposition == 0:
        T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif transposition == 1:
        T = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    # DICOM transfomation matrix,
    # transforms ParaVision patient system into DICOM system
    D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    # Reverse matrix (this matrix fixes the slice direction)
    r = round(np.linalg.det(A @ D @ T))
    if r == 1:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif r == -1:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        raise ValueError("Determinant of R does not result in +1 or -1")

    matrix_rps_to_img = R.T @ D.T @ T.T
    diff_dirs = np.array(diff_dirs)
    diff_dirs_img_coord = matrix_rps_to_img @ diff_dirs.T
    diff_dirs_img_coord = diff_dirs_img_coord.T.tolist()
    diff_dirs_img_coord = [tuple(x) for x in diff_dirs_img_coord]

    return diff_dirs_img_coord


def read_bruker_file(folder_path: Path, data_type: str) -> Tuple[dict, list, list, list, list, list, list]:
    """Reads the image data from a 2dseq file, as well as the corresponding tags
    from visu_pars and method file, where necessary

    Args:
        folder_path: the path where the relevant file/files to be loaded are.
        data_type: "magnitude" or "phase"

    Returns:
        order: dictionary containing the order of acquisition components
            keys - one of diffusion_direction, slice, repetition
            value - number of items the the specific category
        reconstructed_images_pixels_arrays: list of images
        b_vals: list of b-values
        diff_dirs: list of diffusion directions
        slice_locations: list of slice locations
        pixel_spacing: list pixel spacing in the 2 planar directions
        image_orientation_patient: 6 values representing the first 2 column
            of the VisuCoreOrientation parameter from visu_pars file
    """
    # images
    pdata_filepath = folder_path / "pdata"
    twodseq_filepath = pdata_filepath / data_type / "2dseq"
    if not twodseq_filepath.exists():
        if data_type == "magnitude":
            twodseq_filepath = pdata_filepath / "1" / "2dseq"
        elif data_type == "phase":
            twodseq_filepath = pdata_filepath / "2" / "2dseq"
            if not twodseq_filepath.exists():
                twodseq_filepath = pdata_filepath / "3" / "2dseq"
    if pdata_filepath.exists() and twodseq_filepath.exists():
        dataset = Dataset(twodseq_filepath)
    else:
        return {}, [], [], [], [], [], []
    reconstructed_images_pixels_arrays = get_reconstructed_images(dataset)

    # read tags
    diff_dirs = []
    visu_pars_file_path = folder_path / "visu_pars"

    # read visu_pars file
    with open(visu_pars_file_path) as file:
        file_contents = file.read()
        file_contents = remove_endline_characters(file_contents)
        file_contents = remove_multiple_white_spaces(file_contents)

        # order
        visu_order_desc_string = get_relevant_substring(file_contents, "order")
        visu_order_desc_string_elems = get_relevant_elements_from_substring(visu_order_desc_string, "order")
        order = construct_order_dictionary(visu_order_desc_string_elems, BRUKER_NAMES_CORRESPONDENCE_DICT)

        # assert check to capture potential datasets mismatches
        if np.prod(list(order.values())) != len(reconstructed_images_pixels_arrays):
            raise ValueError("Invalid number of images")

        # fgelem
        visu_fg_elem_comment_string = get_relevant_substring(file_contents, "fgelem")
        visu_fg_elem_comment_elems = get_relevant_elements_from_substring(visu_fg_elem_comment_string, "fgelem")

        b_vals, number_of_a0_images = construct_b_values_list(order["diffusion_direction"], visu_fg_elem_comment_elems)

        # slice distance
        visu_core_slice_distance = get_relevant_substring(file_contents, "slice_distance")
        slice_distance = get_relevant_elements_from_substring(visu_core_slice_distance, "slice_distance")

        # slice number
        visu_core_slice_number = get_relevant_substring(file_contents, "slice_number")
        slice_number = get_relevant_elements_from_substring(visu_core_slice_number, "slice_number")

        # slice locations
        slice_locations = list(np.arange(0, slice_distance * slice_number, slice_distance, dtype=float))

        # pixel spacing
        pixel_spacing = construct_pixel_spacing(file_contents)

    # now read the method file
    method_file_path = folder_path / "method"
    with open(method_file_path) as file:
        file_contents = file.read()
        file_contents = remove_endline_characters(file_contents)
        file_contents = remove_multiple_white_spaces(file_contents)

        # diffusion directions
        method_diff_grad_comment_string = get_relevant_substring(file_contents, "diff_dir")
        diff_dirs = construct_diffusion_directions(
            method_diff_grad_comment_string, folder_path, number_of_a0_images, order
        )

        # Gradient orientation
        method_diff_grad_orient_comment_string = get_relevant_substring(file_contents, "diff_grad_orient")
        image_orientation_patient = get_relevant_elements_from_substring(
            method_diff_grad_orient_comment_string, "image_orientation_patient"
        )

    # now finally read the reco file
    reco_file_path = pdata_filepath / data_type / "reco"
    if not reco_file_path.exists():
        if data_type == "magnitude":
            reco_file_path = pdata_filepath / "1" / "reco"
        elif data_type == "phase":
            reco_file_path = pdata_filepath / "2" / "reco"
            if not reco_file_path.exists():
                reco_file_path = pdata_filepath / "3" / "reco"

    with open(reco_file_path) as file:
        file_contents = file.read()
        file_contents = remove_endline_characters(file_contents)
        file_contents = remove_multiple_white_spaces(file_contents)

        # transposition value
        reco_transposition_string = get_relevant_substring(file_contents, "transposition")
        transposition = get_relevant_elements_from_substring(reco_transposition_string, "transposition")

    # convert diff directions from RPS to image coordinates
    diff_dirs = convert_rps_to_img(
        diff_dirs,
        image_orientation_patient,
        transposition,
    )

    return (
        order,
        reconstructed_images_pixels_arrays,
        b_vals,
        diff_dirs,
        slice_locations,
        pixel_spacing,
        image_orientation_patient,
    )


def get_full_diffusion_parameters(
    order: dict, slice_locations: list, b_vals: list, diff_dirs: list
) -> Tuple[list, list, list]:
    """Construct full diffusion parameter arrays based on order dictionary.

    This operation is necessary as the data read from the relevant Bruker
    files is not given for every image in the dataset. Each parameter has
    a list of unique values (sometimes called raw in the code), which need
    to be duplicated according to the other diffusion parameters.
    E.g. PV6 dataset, 150 b-values, 70 slice locations, 4 repetitions.
    As we have 150*70*4 = 4200 images, we need to extend the 150 b-values
    to a list of 4200 b-values based on the order dictionary.

    Args:
        order: order dictionary
        slice_locations: initial list of slice locations
        b_vals: initial list of b-values
        diff_dirs: initial list of diffusion directions

    Returns:
        extended slice_locations, b_vals, diff_dirs lists

    """
    order_keys = list(order.keys())
    if order_keys[:2] == ["slice", "diffusion_direction"]:
        no_repetitions = order.get("repetition", 1)
        increment_factor = order["diffusion_direction"] * no_repetitions
        slice_locations = list(np.repeat(slice_locations, increment_factor))
        b_vals = list(np.repeat(b_vals, repeats=no_repetitions)) * order["slice"]
        diff_dirs = [item for item in diff_dirs for i in range(no_repetitions)] * order["slice"]
        # TODO: changed these lines, but this needs further testing
        # b_vals = b_vals * order["slice"] * no_repetitions
        # diff_dirs = diff_dirs * order["slice"] * no_repetitions

    elif len(order_keys) == 1 and order_keys[0] == "diffusion_direction":
        slice_locations *= len(diff_dirs)
    else:
        raise ValueError("Unknown format of Bruker data")

    return slice_locations, b_vals, diff_dirs


def load_bruker(paths: List[Path], phase_data_present: bool) -> Tuple[pd.DataFrame, dict]:
    """Constructs the dataframe from the data

    Args:
        paths: list of paths where the relevant file/files
            to be loaded are.
        phase_data_present: boolean variable denoting whether phase data is present

    Returns:
        df: a pandas DataFrame representing the input_data
        attributes: a dictionary representing the attributes

    """
    df = []
    # attributes
    attrs = {"image_comments": None}
    repetition_factors: Dict[Tuple, int] = {}
    images_phase = []

    for idx, path in enumerate(tqdm(paths)):
        print(path)
        (
            order,
            images,
            b_vals,
            diff_dirs,
            slice_locations,
            pixel_spacing,
            image_orientation_patient,
        ) = read_bruker_file(path, data_type="magnitude")

        if phase_data_present:
            (
                order_phase,
                current_images_phase,
                b_vals_phase,
                diff_dirs_phase,
                slice_locations_phase,
                pixel_spacing_phase,
                image_orientation_patient_phase,
            ) = read_bruker_file(path, data_type="phase")
            images_phase.extend(current_images_phase)
            if order_phase != order:
                raise ValueError("Incompatible phase data!")

        # order dictionary must be present in all files, otherwise data is invalid
        if order == {}:
            return pd.DataFrame(), {}

        # initialize attributes with the ones from the first path
        # these should be consistent across scans

        # sanity check for the case of different attributes
        if idx != 0 and (
            image_orientation_patient != attrs["image_orientation_patient"] or pixel_spacing != attrs["pixel_spacing"]
        ):
            raise ValueError("Different attributes between bruker files!")
        if idx == 0:
            attrs["image_orientation_patient"] = image_orientation_patient
            attrs["pixel_spacing"] = pixel_spacing
        (
            extended_slice_locations,
            extended_b_vals,
            extended_diff_dirs,
        ) = get_full_diffusion_parameters(order, slice_locations, b_vals, diff_dirs)

        for i, (image, diffusion_direction, slice_location, b_value) in enumerate(
            zip(
                images,
                extended_diff_dirs,
                extended_slice_locations,
                extended_b_vals,
            )
        ):
            config = (
                b_value,
                diffusion_direction,
                slice_location,
            )
            last_repetition = repetition_factors[config] = repetition_factors.get(config, -1) + 1
            df.append(
                (
                    None,
                    image,
                    b_value,
                    diffusion_direction,
                    slice_location,
                    last_repetition,
                )
            )

    df = pd.DataFrame(
        df,
        columns=[
            "file_name",
            "image",
            "b_value",
            "diffusion_direction",
            "slice_position",
            "repetition",
        ],
    )

    # make this table more complete to match siemens data
    # rename slice_position to image_position
    df.rename(columns={"slice_position": "image_position"}, inplace=True)
    # make the image_position a list with x y z
    df["image_position"] = df["image_position"].apply(lambda x: [0, 0, x])
    # add missing columns
    df["series_number"] = 0
    df["series_description"] = "None"
    df["acquisition_date_time"] = "None"
    df["nominal_interval"] = "None"
    df["image_comments"] = "None"
    iop_list = [
        attrs["image_orientation_patient"][0][0],
        attrs["image_orientation_patient"][0][1],
        attrs["image_orientation_patient"][0][2],
        attrs["image_orientation_patient"][1][0],
        attrs["image_orientation_patient"][1][1],
        attrs["image_orientation_patient"][1][2],
    ]
    df["image_orientation_patient"] = [iop_list for i in df.index]

    if phase_data_present:
        df["phase_image"] = images_phase

    return df, attrs
