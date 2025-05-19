import argparse
import logging
import pathlib
import shutil

import pyautogui
from scipy.io import loadmat
from tqdm import tqdm

from indi.extensions.read_data.read_and_pre_process_data import read_data


def open_matlab_outliers(matlab_path):

    mat_file = matlab_path / "average_matrix.mat"
    if not mat_file.exists():
        mat_file = matlab_path / "frame_selection_matrix.mat"
        mat_data = loadmat(mat_file)
        average_matrix = mat_data["frame_selection_matrix"][0, 0]
    else:
        mat_data = loadmat(mat_file)
        average_matrix = mat_data["new_average_matrix"]

    return average_matrix


def get_logger():
    logger = logging.getLogger("remove_repetitions")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_arguments():
    parser = argparse.ArgumentParser(description="Remove repetitions from data")

    parser.add_argument(
        "--path",
        type=pathlib.Path,
        required=True,
        help="Path to data folders",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        required=True,
        help="Path to save the output",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions to keep",
    )

    return parser.parse_args()


def copy_repetitions(data, slices, current_folder, output_folder, number_of_repetitions):

    slices = data["slice_integer"].unique()

    for slice_idx in slices:
        current_entries = data.loc[data["slice_integer"] == slice_idx].copy()
        # how many diffusion configs do we have for this slice?
        current_entries["diffusion_direction"] = [tuple(lst_in) for lst_in in current_entries["diffusion_direction"]]
        diffusion_configs_table = (
            current_entries.groupby(["b_value_original", "diffusion_direction"]).size().reset_index(name="Freq")
        )

        for i, row in diffusion_configs_table.iterrows():

            temp = current_entries[
                (current_entries["b_value_original"] == row["b_value_original"])
                & (current_entries["diffusion_direction"] == row["diffusion_direction"])
            ]
            temp = temp.reset_index(drop=True)

            # print(temp)

            data_to_keep = temp.iloc[:number_of_repetitions, :]
            data_to_keep = data_to_keep.reset_index(drop=True)

            for current_file in data_to_keep["file_name"]:
                file = pathlib.Path(current_file)

                shutil.copy(
                    current_folder / file,
                    output_folder / file.name,
                )


def main():

    logger = get_logger()
    args = get_arguments()

    folders = list(args.path.glob("*/**/diffusion_images/"))

    tempdir = args.out_path / "temp"
    tempdir.mkdir(parents=True, exist_ok=True)

    settings = {
        "debug": False,
        "workflow_mode": "main",
        "sequence_type": "se",
        "remove_b_values": [],
        "remove_slices": [],
        "results": tempdir.as_posix(),
        "screen_size": pyautogui.size(),
        "ex_vivo": False,
        "print_series_description": False,
    }

    for folder in tqdm(folders):

        folder_name = folder.relative_to(args.path)
        output_folder_all = args.out_path / "all" / folder_name.as_posix()
        output_folder_less_repetitions = args.out_path / f"{args.repetitions}_repetitions" / folder_name.as_posix()
        output_folder_all.mkdir(parents=True, exist_ok=True)
        output_folder_less_repetitions.mkdir(parents=True, exist_ok=True)

        info = {}
        settings["dicom_folder"] = folder.as_posix()

        data, info, slices = read_data(settings, info, logger)

        # indicies_to_remove = manual_image_removal(
        #     data,
        #     slices,
        #     {},
        #     np.ones(info["img_size"], dtype=bool),
        #     settings,
        #     "pre",
        #     info,
        # )

        matlab_path = folder / "../matlab_data/"
        average_matrix = open_matlab_outliers(matlab_path)

        directions = data["diffusion_direction"].unique()
        n_directions = len(directions)
        averages = []

        for i in range(n_directions):
            current_data = data[data["diffusion_direction"] == directions[i]]
            averages.append(len(current_data))

        indicies_to_remove = []
        for i in range(n_directions):
            for j in range(averages[i]):
                current_data = data[data["diffusion_direction"] == directions[i]]
                current_data.reset_index(drop=True, inplace=True)
                # print(current_data)
                # montage_image[counter, :, :] = current_data.iloc[j].image
                # montage_image[counter, :, :] = data.iloc[counter].image
                if average_matrix[i, j] == 0:
                    index = data[current_data.iloc[j].file_name == data["file_name"]].index[0]
                    indicies_to_remove.append(index)

        all_indicies = data.index.tolist()
        indicies_to_keep = list(set(all_indicies) - set(indicies_to_remove))

        for index in indicies_to_keep:
            # copy the file to the output folder
            source_file = folder / data.loc[index]["file_name"]
            destination_file = output_folder_all / source_file.name
            shutil.copy(source_file, destination_file)

        # remove the indicies from the data
        data = data.loc[indicies_to_keep].copy()
        data.reset_index(drop=True, inplace=True)

        # copy the repetitions
        copy_repetitions(data, slices, folder, output_folder_less_repetitions, args.repetitions)
