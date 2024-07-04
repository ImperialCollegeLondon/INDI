"""initial setup for the pipeline"""

import glob
import logging
import os
import sys

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
    if settings["sequence_type"] == "se":
        logger.info("U-Net segmentation is disabled because sequence type is SE!")
        settings["u_net_segmentation"] = False
    if settings["remove_outliers_manually_pre"] == True and settings["remove_outliers_manually"] == False:
        logger.info("Enabling manual removal post segmentation as remove_outliers_manually_pre is enabled!")
        settings["remove_outliers_manually"] = True

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
    if not settings["start_folder"] and len(sys.argv) == 1:
        logger.error("No path defined in YAML file or command argument!")
        sys.exit(1)
    # if path exists in the command argument then overwrite any path given in YAML file
    if len(sys.argv) > 1:
        settings["start_folder"] = sys.argv[1]
        logger.info("Path defined in command argument!")
    # finally check if path exists
    if not os.path.exists(settings["start_folder"]):
        logger.error("Start path does not exist!")
        sys.exit(1)

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
    with open(os.path.join(script_path, "settings.yaml"), "r") as handle:
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
    settings = solve_conflicts(settings, logger)

    # move to the start folder
    os.chdir(settings["start_folder"])

    # find all subfolders called dicoms recursively
    all_to_be_analysed_folders = glob.glob(settings["start_folder"] + "/**/diffusion_images", recursive=True)
    all_to_be_analysed_folders.sort()

    return dti, settings, logger, log_format, all_to_be_analysed_folders
