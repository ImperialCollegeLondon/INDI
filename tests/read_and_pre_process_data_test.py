import logging
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from extensions.read_data.read_and_pre_process_data import read_data


@pytest.fixture
def dummy_settings():
    settings = {}

    path = pathlib.Path(os.path.join("tests", "data", "dicom_files"))
    settings["dicom_folder"] = (
        sorted(filter(os.path.isdir, path.glob("*")), key=os.path.getmtime)[0] / "diffusion_images"
    )
    abspath = os.path.abspath(os.path.join("extensions"))
    dname = os.path.dirname(abspath)

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)
    settings.update(
        {
            "code_path": os.path.abspath("./"),
            "session": tmp,
            "results": tmp,
            "debug_folder": tmp,
            "debug": True,
            "ex_vivo": True,
            "registration_mask_scale": 1.0,
        }
    )
    settings["code_path"] = dname
    settings["ex_vivo"] = True
    settings["workflow_mode"] = "main"
    settings["sequence_type"] = "se"
    settings["debug"] = False
    settings["remove_slices"] = []
    return settings


@pytest.fixture
def dummy_info():
    info = {}
    return info


@pytest.fixture
def table_true():
    table = pd.read_pickle(os.path.join("tests", "data", "cdti_table_from_dicoms.zip"))
    return table


def test_read_data(dummy_settings, dummy_info, table_true):
    """test if this function creates the DTI table correctly"""

    # Sort both tables by slice_integer and b_value (or name ...)
    table_calculated, _, _ = read_data(dummy_settings, dummy_info, logging.getLogger(__name__))

    table_true["diffusion_direction"] = table_true["direction"].apply(lambda x: tuple(x))

    dir_true = np.asarray(table_true["diffusion_direction"].tolist())
    dir_calculated = np.asarray(table_calculated["diffusion_direction"].tolist())

    print(dir_true)
    print(dir_calculated)

    print(np.isclose(dir_true, dir_calculated))
    print(np.isclose(np.abs(dir_true), np.abs(dir_calculated)))

    print(table_calculated["diffusion_direction"])

    print(table_true["diffusion_direction"])

    assert table_calculated["slice_integer"].equals(table_true["slice_integer"]), "slice_integer column is not equal"
    assert table_calculated["b_value"].equals(table_true["b_value"]), "b_value column is not equal"
    assert table_calculated["image"].equals(table_true["image"]), "image column is not equal"
    assert table_calculated["diffusion_direction"].equals(
        table_true["diffusion_direction"]
    ), "direction column is not equal"
