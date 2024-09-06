import os

import numpy as np
import pandas as pd

from extensions.read_data.read_and_pre_process_data import read_data


def test_read_data():
    """test if this function creates the DTI table correctly"""
    table_true = pd.read_pickle(os.path.join("tests", "data", "cdti_table_from_dicoms.zip"))

    # mock dictionaries
    info = {}
    settings = {}
    settings["dicom_folder"] = os.path.join("tests", "data", "dicom_data")
    abspath = os.path.abspath(os.path.join("extensions"))
    dname = os.path.dirname(abspath)
    settings["code_path"] = dname
    table_calculated, info = read_data(settings, info)

    # we need to separate the dir list into 3 columns
    table_true[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(table_true.direction.tolist(), index=table_true.index)
    table_true = table_true.drop("diffusion_direction", axis=1)

    table_calculated[["dir_x", "dir_y", "dir_z"]] = pd.DataFrame(
        table_calculated.direction.tolist(), index=table_calculated.index
    )
    table_calculated = table_calculated.drop("diffusion_direction", axis=1)

    # we also need to convert the image column to a numpy array
    table_images_true = table_true["image"].to_numpy()
    table_images_true = np.stack(table_images_true)
    table_true = table_true.drop("image", axis=1)

    table_images_calculated = table_calculated["image"].to_numpy()
    table_images_calculated = np.stack(table_images_calculated)
    table_calculated = table_calculated.drop("image", axis=1)

    # I am going to skip the header column as it is a very large dictionary for each cell
    table_calculated = table_calculated.drop("header", axis=1)

    # compare both tables
    comparison_table = table_calculated.compare(table_true)
    assert comparison_table.empty
    # compare the image arrays
    assert np.allclose(table_images_calculated, table_images_true)
