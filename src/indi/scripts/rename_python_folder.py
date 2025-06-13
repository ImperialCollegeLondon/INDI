import argparse
import datetime
import pathlib
import shutil

import yaml
from pyautogui import Size


def get_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("tag:yaml.org,2002:python/object/new:pyautogui.Size", Size)
    return loader


def get_arguments():
    parser = argparse.ArgumentParser(description="Copy Python folder with denoiser name ")

    parser.add_argument(
        "--path",
        type=pathlib.Path,
        required=True,
        help="Path to data folders",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    loader = get_loader()

    folders = args.path.glob("*/**/Python_post_processing/")

    date = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    for folder in folders:

        with open(folder / "settings.yml") as f:
            settings = yaml.load(f, Loader=loader)

        name = date + "_"

        if settings["image_denoising"]:
            name += settings["denoise_method"]
        elif settings["tensor_denoising"]:
            name += "tensor_denoising"
        elif settings["uformer_denoise"]:
            name += "uformer_denoise"
        elif settings["tensor_fit_method"] == "DIP":
            name += "tensor_DIP_fit"
        else:
            name += "no_denoising"

        shutil.copytree(
            folder,
            folder / ".." / name,
            dirs_exist_ok=True,
        )
