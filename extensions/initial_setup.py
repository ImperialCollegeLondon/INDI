"""initial setup for the pipeline"""

import glob
import logging
import os

import yaml


def solve_conflicts(settings: dict, logger: logging.Logger) -> dict:
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
    if not settings["manual_segmentation"]:
        logger.info("U-Net segmentation is enabled because manual segmentation is disabled!")
        settings["u_net_segmentation"] = True
    if settings["sequence_type"] == "se":
        logger.info("U-Net segmentation is disabled because sequence type is SE!")
        settings["u_net_segmentation"] = False

    return settings


def initial_setup(script_path: str) -> [dict, dict, dict, logging, logging, list]:
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

    # read settings from YAML file
    settings = {}
    yaml_file = open(os.path.join(script_path, "settings.yaml"), "r")
    settings = yaml.safe_load(yaml_file)

    # solve conflicts from different settings:
    settings = solve_conflicts(settings, logger)

    # add root path of the code
    settings["code_path"] = script_path

    # console logger level
    if settings["debug"]:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # move to the start folder
    os.chdir(settings["start_folder"])

    # find all subfolders called dicoms recursively
    all_to_be_analysed_folders = glob.glob(settings["start_folder"] + "/**/diffusion_images", recursive=True)
    all_to_be_analysed_folders.sort()

    return dti, settings, logger, log_format, all_to_be_analysed_folders
