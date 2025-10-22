import logging
import os

import numpy as np
import pandas as pd

import pytest

from indi.extensions.tensor_fittings import dipy_tensor_fit


@pytest.fixture
def logger():
    return logging.getLogger("test_logger")


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("method", ["NLLS", "LS", "WLS", "RESTORE", "RNLLS", "RWLS"])
def test_dipy_tensor_fit(tmp_path, debug, method, logger):
    # Here I am loading data from DTI_numerical_phantom_3D repo
    # load table with pixel values, b_values and directions
    # For this test there is no noise, so SNR is infinite
    table = pd.read_pickle(os.path.join("tests", "data", "cdti_table.zip"))

    # Rename some columns to match what is expected in the function
    table = table.rename(columns={"direction": "diffusion_direction"})

    slice_position_column = np.repeat(["0.0"], len(table))
    table["slice_position"] = slice_position_column
    slice_integer_column = np.repeat([0], len(table))
    table["slice_integer"] = slice_integer_column
    table["to_be_removed"] = np.repeat([False], len(table))
    table["bmatrix"] = np.repeat([None], len(table))
    # load mask (0 background, 1 left ventricle, 2 right ventricle)
    mask_3c = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask_3c = mask_3c["mask_3c"]
    # load tensor that was used to create these pixel values
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = tensor_true["DT"]
    slices = [0]

    average_images = {0: table["image"].mean(axis=0)}

    # mock settings
    settings = {}
    settings["debug"] = debug
    (tmp_path / "results_b").mkdir(parents=True, exist_ok=True)
    (tmp_path / "debug").mkdir(parents=True, exist_ok=True)
    settings["debug_folder"] = str(tmp_path / "debug")
    settings["results"] = str(tmp_path)
    # mock info
    info = {
        "img_size": table["image"][0].shape,
        "n_slices": len(slices),
    }
    # mock dictionary with settings
    settings["tensor_fit_method"] = method

    # calculate tensors with DiPy functions
    tensor, s0, residuals_img, residuals_map, residuals_img_all, info = dipy_tensor_fit(
        slices=slices,
        data=table,
        info=info,
        settings=settings,
        mask_3c=mask_3c,
        average_images=average_images,
        logger=logger,
        method=settings["tensor_fit_method"],
        quick_mode=False,
    )

    # create a mask to ignore nan values
    mask = ~(np.isnan(tensor_true))

    # assert that the tensors are the same
    mask = ~(np.isnan(tensor) | np.isnan(tensor_true))
    assert np.allclose(tensor[mask], tensor_true[mask], atol=1e-2)

    # check if residuals are close to zero
    # create a mask to ignore nan values
    mask = ~(np.isnan(residuals_map[0]))
    # because there is no noise, I expect the residuals to be very small
    assert np.isclose(np.median(residuals_map[0][mask]), 0, atol=1e-1)
