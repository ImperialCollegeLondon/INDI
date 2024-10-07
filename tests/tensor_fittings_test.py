import logging
import os

import numpy as np
import pandas as pd

from extensions.tensor_fittings.tensor_fittings import quick_tensor_fit


def test_dipy_tensor_fit():
    # Here I am loading data from DTI_numerical_phantom_3D repo
    # load table with pixel values, b_values and directions
    # For this test there is no noise, so SNR is infinite
    table = pd.read_pickle(os.path.join("tests", "data", "cdti_table.zip"))
    slice_position_column = np.repeat(["0.0"], len(table))
    table["slice_position"] = slice_position_column
    # load mask (0 background, 1 left ventricle, 2 right ventricle)
    mask_3c = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask_3c = mask_3c["mask_3c"]
    # load tensor that was used to create these pixel values
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = tensor_true["DT"]
    # # mock noise as zero
    # noise = np.zeros(table["image"][0].shape)
    # mock slices
    slices = ["0.0"]

    # mock options
    class Options:
        def __init__(self, debug):
            self.debug = debug

    # mock settings
    settings = {}
    settings["debug"] = False

    # mock info
    info = {
        "img_size": table["image"][0].shape,
    }
    # mock dictionary with settings
    settings["tensor_fit_method"] = "LS"
    # mock logger
    logger = logging.getLogger(__name__)

    # calculate tensors with DiPy functions
    tensor_calculated, s0, residuals_img, residuals_map, _ = quick_tensor_fit(
        slices,
        table,
        info,
        settings,
        mask_3c,
        logger,
        method=settings["tensor_fit_method"],
        quick_mode=False,
    )

    # create a mask to ignore nan values
    mask = ~(np.isnan(tensor_true))

    # assert that the tensors are the same
    mask = ~(np.isnan(tensor_calculated) | np.isnan(tensor_true))
    assert np.allclose(tensor_calculated[mask], tensor_true[mask])

    # check if residuals are close to zero
    # create a mask to ignore nan values
    mask = ~(np.isnan(residuals_map["0.0"]))
    # because there is no noise, I expect the residuals to be very small
    assert np.allclose(residuals_map["0.0"][mask], 0, atol=1e-3)
    mask = ~(np.isnan(residuals_img["0.0"]))
    assert np.allclose(residuals_img["0.0"][mask], 0, atol=1)
