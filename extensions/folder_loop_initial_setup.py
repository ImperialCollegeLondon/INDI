import logging
import os

# import pprint
import shutil

import yaml


def folder_loop_initial_setup(
    current_folder: str, settings: dict, logger: logging.Logger, log_format: logging.Formatter
) -> tuple[dict, dict, logging.Logger]:
    """
    Initial setup for the folder loop.
    Parameters
    ----------
    current_folder : str with current folder with diffusion data
    settings : settings from YAML file
    logger : logger
    log_format : logging.Formatter
    Returns
    -------
    info : dict
    dti_data : dict
    settings : dict
    logger : logger


    """
    # initialise dictionaries
    info = {}

    # define working folders
    settings["dicom_folder"] = current_folder
    current_folder = os.path.dirname(current_folder)

    settings["work_folder"] = os.path.join(current_folder, "Python_post_processing")
    settings["debug_folder"] = os.path.join(settings["work_folder"], "debug")
    settings["results"] = os.path.join(settings["work_folder"], "results")
    settings["session"] = os.path.join(settings["work_folder"], "session")

    # create folders
    if settings["debug"]:
        if not os.path.exists(settings["debug_folder"]):
            os.makedirs(settings["debug_folder"])
        else:
            shutil.rmtree(settings["debug_folder"])
            os.makedirs(settings["debug_folder"])
    else:
        if os.path.exists(settings["debug_folder"]):
            shutil.rmtree(settings["debug_folder"])

    if not os.path.exists(settings["results"]):
        os.makedirs(settings["results"])
        os.makedirs(os.path.join(settings["results"], "data"))
        os.makedirs(os.path.join(settings["results"], "results_a"))
        os.makedirs(os.path.join(settings["results"], "results_b"))
    else:
        shutil.rmtree(settings["results"])
        os.makedirs(settings["results"])
        os.makedirs(os.path.join(settings["results"], "data"))
        os.makedirs(os.path.join(settings["results"], "results_a"))
        os.makedirs(os.path.join(settings["results"], "results_b"))

    # save settings to YAML file
    with open(os.path.join(settings["work_folder"], "settings.yml"), "w") as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)

    # file logger setup
    # delete previous log file
    log_filename = os.path.join(settings["work_folder"], "analysis.log")
    if os.path.exists(log_filename):
        os.remove(log_filename)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if settings["debug"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("============================================================")
    logger.info("============================================================")
    logger.info("Current Folder: " + current_folder)
    logger.info("============================================================")
    # logger.info(f"Settings: \n{pprint.pformat(settings)}")

    # create a debug folder
    if settings["debug"]:
        if not os.path.exists(settings["debug_folder"]):
            os.makedirs(settings["debug_folder"])
        else:
            shutil.rmtree(settings["debug_folder"])
            os.makedirs(settings["debug_folder"])
    else:
        if os.path.exists(settings["debug_folder"]):
            shutil.rmtree(settings["debug_folder"])

    if not os.path.exists(settings["results"]):
        os.makedirs(settings["results"])
    if not os.path.exists(settings["session"]):
        os.makedirs(settings["session"])

    return info, settings, logger
