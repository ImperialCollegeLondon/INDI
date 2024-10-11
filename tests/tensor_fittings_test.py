import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from extensions.extensions import convert_array_to_dict_of_arrays
from extensions.tensor_fittings.tensor_fittings import (
    TensorFit,
    get_residual_z_scores,
    plot_residuals_map,
    plot_residuals_plot,
    quick_tensor_fit,
)

IM_SIZE = 10
NUM_SLICES = 3


@pytest.fixture
def dummy_residuals():
    return np.random.rand(NUM_SLICES, IM_SIZE, IM_SIZE)


@pytest.fixture
def dummy_average_images():
    return np.random.rand(NUM_SLICES, IM_SIZE, IM_SIZE)


@pytest.fixture
def dummy_mask():
    return 1.0 * np.random.randint(0, 3, (NUM_SLICES, IM_SIZE, IM_SIZE))


@pytest.fixture
def dummy_settings():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)
    settings = {
        "code_path": os.path.abspath("./"),
        "session": tmp,
        "results": tmp,
        "debug_folder": tmp,
        "debug": True,
        "ex_vivo": True,
        "registration_mask_scale": 1.0,
    }
    return settings


@pytest.fixture
def get_data():
    # Here I am loading data from DTI_numerical_phantom_3D repo
    # load table with pixel values, b_values and directions
    # For this test there is no noise, so SNR is infinite
    table = pd.read_pickle(os.path.join("tests", "data", "cdti_table.zip"))

    # load mask (0 background, 1 left ventricle, 2 right ventricle)
    mask_3c = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask_3c = mask_3c["mask_3c"]
    # load tensor that was used to create these pixel values
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = np.squeeze(tensor_true["DT"])

    table["to_be_removed"] = False
    table["slice_integer"] = 0
    table["diffusion_direction"] = table["direction"]

    # mock info
    info = {
        "img_size": table["image"][0].shape,
        "n_slices": 1,
    }

    return table, mask_3c, tensor_true, info


def test_plot_residuals_plot(dummy_residuals, dummy_settings):
    slice_idx = 1
    plot_residuals_plot(np.mean(dummy_residuals, axis=(-2, -1)), slice_idx, dummy_settings)

    file_name = os.path.join(
        dummy_settings["debug_folder"],
        "tensor_residuals_" + "_slice_" + str(slice_idx).zfill(2) + ".png",
    )
    assert os.path.exists(file_name), "Residuals plot was not created"


def test_plot_residuals_map(dummy_residuals, dummy_average_images, dummy_mask, dummy_settings):
    slice_idx = 0
    print(dummy_residuals[slice_idx].shape)
    print(dummy_residuals[slice_idx].dtype)
    print(dummy_residuals[slice_idx])
    plot_residuals_map(dummy_residuals[slice_idx], dummy_average_images, dummy_mask, slice_idx, dummy_settings)

    file_name = os.path.join(
        dummy_settings["debug_folder"],
        "tensor_residuals_map_" + "_slice_" + str(slice_idx).zfill(2) + ".png",
    )
    assert os.path.exists(file_name), "Residuals map was not created"


def test_quick_tensor_fit(get_data):
    table, _, tensor_true, info = get_data

    # calculate tensors with DiPy functions
    tensor_calculated = quick_tensor_fit(0, table, info)

    # assert that the tensors are the same
    mask = ~(np.isnan(tensor_calculated) | np.isnan(tensor_true))
    assert np.allclose(tensor_calculated[mask], tensor_true[mask])


def test_get_residual_z_scores(dummy_residuals):
    z_scores, outliers, outliers_pos = get_residual_z_scores(dummy_residuals)

    assert isinstance(z_scores, np.ndarray)
    assert isinstance(outliers, np.ndarray)
    assert isinstance(outliers_pos, np.ndarray)
    assert z_scores.shape == dummy_residuals.shape
    assert outliers.dtype == bool


# TODO Check if the combinations of methods and quick_mode are correct
@pytest.mark.parametrize(
    "method, quick_mode",
    [
        ("NLLS", False),
        ("RESTORE", False),
        ("LS", True),
        ("WLS", True),
        ("LS", False),
        ("WLS", False),
    ],
)
def test_tensor_fitting(method, quick_mode, get_data, dummy_settings):
    data, mask_3c, tensor_true, info = get_data
    context = {
        "data": data,
        "info": info,
        "mask_3c": convert_array_to_dict_of_arrays(mask_3c, np.arange(1)),
        "slices": np.arange(1),
        "average_images": np.mean(data["image"].values, axis=0, keepdims=True),
    }
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)

    settings = dummy_settings
    settings["tensor_fitting_method"] = method

    fit = TensorFit(context, settings, logging.getLogger(__name__), method, quick_mode)
    fit.run()
    dti = context["dti"]

    assert np.allclose(
        dti["tensor"][0][mask_3c[0] == 1], tensor_true[mask_3c[0] == 1], atol=1e-3
    ), f"Tensor fitting failed for {method} method"
    if not quick_mode and method != "RESTORE":
        assert np.allclose(
            dti["residuals_map"][0][mask_3c[0] == 1], 0, atol=1e-3
        ), f"Fitting residual failed for {method} method"
