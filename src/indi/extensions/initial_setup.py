"""initial setup for the pipeline"""

import argparse
import glob
import logging
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for processing diffusion images")

    parser.add_argument("settings", type=str, help="path to the settings YAML file")

    parser.add_argument(
        "--start_folder",
        type=str,
        default=None,
        help="path to the folder where the diffusion images are stored",
    )

    return parser.parse_args()


def solve_conflicts(settings: dict, args: argparse.Namespace, logger: logging.Logger) -> dict:
    """solve conflicts in YAML file

    Parameters
    ----------
    settings : dict
        settings from YAML file

    Returns
    -------
    dict
        settings with conflicts resolved
    """

    if settings["remove_outliers_manually"]:
        logger.info("Removing outliers with AI is disabled because manual removal of outliers is enabled!")
        settings["remove_outliers_with_ai"] = False
    if settings["sequence_type"] == "se":
        logger.info("U-Net segmentation is disabled because sequence type is SE!")
        settings["u_net_segmentation"] = False
    if settings["remove_outliers_manually_pre"] == True and settings["remove_outliers_manually"] == False:
        logger.info("Enabling manual removal post segmentation as remove_outliers_manually_pre is enabled!")
        settings["remove_outliers_manually"] = True
    if settings["uformer_denoise"]:
        settings["tensor_fit_method"] = "LS"
        logger.info("Enabling LS tensor fit because uformer_denoise is enabled!")

    # ex-vivo settings
    if settings["ex_vivo"]:
        logger.info("Ex-vivo settings enabled!")
        settings["registration_extra_debug"] = False
        logger.info("registration_extra_debug set to False!")
        settings["remove_outliers_manually"] = False
        logger.info("remove_outliers_manually set to False!")
        settings["remove_outliers_manually_pre"] = False
        logger.info("remove_outliers_manually_pre set to False!")
        settings["remove_outliers_with_ai"] = False
        logger.info("remove_outliers_with_ai set to False!")
        settings["print_series_description"] = False
        logger.info("print_series_description set to False!")
        settings["u_net_segmentation"] = False
        logger.info("u_net_segmentation set to False!")
        settings["uformer_denoise"] = False
        logger.info("uformer_denoise set to False!")

    # check we have a path defined in either the YAML file or as a command argument
    if not settings["start_folder"] and not args.start_folder:
        logger.error("No path defined in YAML file or command argument!")
        raise ValueError("No path defined in YAML file or command argument!")
    # if path exists in the command argument then overwrite any path given in YAML file
    if args.start_folder:
        settings["start_folder"] = args.start_folder
        logger.info("Path defined in command argument!")
    # finally check if path exists
    if not os.path.exists(settings["start_folder"]):
        logger.error("Start path does not exist!")
        raise ValueError("Start path does not exist!")

    return settings


def initial_setup(script_path: str) -> tuple[dict, dict, dict, logging.Logger, logging.Formatter, list]:
    """
    initial setup for the pipeline

    Parameters
    ----------
    script_path : str
        path to the script folder

    Returns
    -------
    dti : dict
        DTI dictionary to hold tensor and parameters
    settings : dict
        settings from YAML file
    logger : logging
        logger for console
    log_format : logging
        logger format
    all_to_be_analysed_folders : list
        list of folders to be analysed

    """
    # logger setup
    log_format = logging.Formatter("%(levelname)s : %(asctime)s :: %(message)s")
    logger = logging.getLogger(__name__)
    # logger for console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # dictionary to hold dti data
    dti = {}

    args = parse_args()

    # read settings from YAML file
    settings = {}
    with open(os.path.join(args.settings), "r") as handle:
        settings = yaml.safe_load(handle)

    # add root path of the code
    settings["code_path"] = script_path

    # console logger level
    if settings["debug"]:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # solve conflicts from different settings:
    settings = solve_conflicts(settings, args, logger)

    # move to the start folder
    # os.chdir(settings["start_folder"])
    # or you change directory or you add it to the start of the glob in the next line

    # find all subfolders called dicoms recursively
    all_to_be_analysed_folders = glob.glob(settings["start_folder"] + "/**/diffusion_images", recursive=True)
    all_to_be_analysed_folders.sort()
    if len(all_to_be_analysed_folders) == 0:
        logger.error("No subfolder named 'diffusion_images' found!")
        raise FileNotFoundError("No subfolder named 'diffusion_images' found!")

    return dti, settings, logger, log_format, all_to_be_analysed_folders
