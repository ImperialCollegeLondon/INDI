import argparse
import logging
import pathlib
import shutil

from indi.extensions.read_data.read_and_pre_process_data import read_data


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
        "--number-of-repetitions",
        type=int,
        default=1,
        help="Number of repetitions to keep",
    )

    return parser.parse_args()


def main():

    logger = get_logger()
    args = get_arguments()

    folders = args.path.glob("*/**/diffusion_images/")

    for folder in folders:

        folder_name = folder.relative_to(args.path)
        output_folder = args.out_path / folder_name.as_posix()
        output_folder.mkdir(parents=True, exist_ok=True)

        settings = {
            "debug": False,
            "dicom_folder": folder.as_posix(),
            "workflow_mode": "main",
            "sequence_type": "se",
            "remove_b_values": [],
            "remove_slices": [],
            "results": output_folder.as_posix(),
        }

        info = {}

        data, info, slices = read_data(settings, info, logger)

        for slice_idx in slices:
            current_entries = data.loc[data["slice_integer"] == slice_idx].copy()
            # how many diffusion configs do we have for this slice?
            current_entries["diffusion_direction"] = [
                tuple(lst_in) for lst_in in current_entries["diffusion_direction"]
            ]
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

                data_to_keep = temp.iloc[: args.number_of_repetitions, :]
                data_to_keep = data_to_keep.reset_index(drop=True)

                for current_file in data_to_keep["file_name"]:
                    file = pathlib.Path(current_file)

                    shutil.copy(
                        folder / file,
                        output_folder / file.name,
                    )
