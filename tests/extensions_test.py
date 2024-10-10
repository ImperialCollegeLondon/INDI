import logging
import os

import numpy as np
import pandas as pd

from extensions.extensions import (
    convert_array_to_dict_of_arrays,
    convert_dict_of_arrays_to_array,
    get_cylindrical_coordinates_short_axis,
    get_snr_maps,
)
from extensions.get_fa_md import get_fa_md


def test_convert_array_to_dict_of_arrays():
    """test if this function converts an array to a dictionary of arrays"""
    # create a 3D array
    array = np.random.rand(10, 10, 10)

    # convert array to dictionary of arrays
    dict_of_arrays = convert_array_to_dict_of_arrays(array, np.arange(10))

    # convert dictionary of arrays back to array
    array_back = convert_dict_of_arrays_to_array(dict_of_arrays)

    assert np.allclose(array, array_back)


def test_get_fa_md():
    """test if this function calculates the correct FA and MD values"""

    # load tensor from numerical phantom
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = tensor_true["DT"]
    tensor_true = np.nan_to_num(tensor_true)

    slices = np.arange(len(tensor_true))

    # load RV and LV mask
    mask = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask = convert_array_to_dict_of_arrays(mask["mask_3c"], slices)

    # mock info dictionary
    info = {}

    # mock logger
    logger = logging.getLogger(__name__)

    # calculate eigenvalues from tensor
    eigenvalues, _ = np.linalg.eigh(tensor_true)
    eigenvalues = convert_array_to_dict_of_arrays(eigenvalues, slices)
    # get FA and MD from eigenvalues
    md_calculated, fa_calculated, _ = get_fa_md(eigenvalues, info, mask, slices, logger)

    md_calculated = convert_dict_of_arrays_to_array(md_calculated)
    fa_calculated = convert_dict_of_arrays_to_array(fa_calculated)

    mean_md_calculated, std_md_calculated = [
        np.mean(md_calculated[md_calculated > 0]),
        np.std(md_calculated[md_calculated > 0]),
    ]
    mean_fa_calculated, std_fa_calculated = [
        np.mean(fa_calculated[fa_calculated > 0]),
        np.std(fa_calculated[fa_calculated > 0]),
    ]

    # correct MD and FA from phantom
    e1, e2, e3 = (2 * 1e-3, 1 * 1e-3, 0.5 * 1e-3)
    md_true = (e1 + e2 + e3) / 3
    fa_true = np.sqrt(1 / 2) * (
        np.sqrt((e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2) / np.sqrt(e1**2 + e2**2 + e3**2)
    )

    assert np.allclose(mean_md_calculated, md_true)
    assert np.allclose(std_md_calculated, 0.0)

    assert np.allclose(mean_fa_calculated, fa_true)
    assert np.allclose(std_fa_calculated, 0.0)


def test_get_snr_maps():
    # load RV and LV mask
    n_configs = 2
    n_average = 6
    n_slices = 3
    im_size = 10

    indices_one_slice = sum([[i] * n_average for i in range(n_configs)], start=[])
    indices = indices_one_slice * n_slices
    slices = sum([[i] * n_configs * n_average for i in range(n_slices)], start=[])

    b_values_one_slice = sum([[i] * n_average for i in range(n_configs)], start=[])
    diffusion_direction_one_slice = sum([[np.random.rand(3)] * n_average for i in range(n_configs)], start=[])

    n_images = len(indices)
    assert n_images == n_configs * n_average * n_slices
    std = 1
    X, Y = np.meshgrid(np.arange(im_size), np.arange(im_size))
    circle = (X - im_size / 2) ** 2 + (Y - im_size / 2) ** 2 < (im_size / 4) ** 2
    images = [circle + std * np.random.randn(im_size, im_size) for _ in range(n_images)]

    # Check that the dataframe is correctly created
    assert len(indices) == len(slices) == len(images)

    data = pd.DataFrame(
        {
            "image": images,
            "diff_config": indices,
            "slice_integer": slices,
            "b_value": b_values_one_slice * n_slices,
            "b_value_original": b_values_one_slice * n_slices,
            "diffusion_direction": diffusion_direction_one_slice * n_slices,
            "diffusion_direction_original": diffusion_direction_one_slice * n_slices,
        }
    )

    mask = np.stack([circle for _ in range(n_slices)], axis=0)
    snr_true = 1 / (std)

    # mock settings dictionary
    settings = {}

    settings["debug"] = False
    settings["ex_vivo"] = False
    # mock logger
    logger = logging.getLogger(__name__)
    slices = np.arange(n_slices)
    # mock info dictionary
    info = {}
    snr_estimated, noise_estimated, _, _ = get_snr_maps(data, mask, None, slices, settings, logger, info)
    print(snr_true, std)
    for snr_slice in snr_estimated:
        for snr in snr_estimated[snr_slice]:
            assert np.allclose(
                np.mean(snr_estimated[snr_slice][snr][circle == 1]), snr_true, 1
            ), f"SNR estimate is not correct ({snr_slice} {snr})"
    for noise_slice in noise_estimated:
        for noise in noise_estimated[noise_slice]:
            assert np.allclose(
                np.mean(noise_estimated[noise_slice][noise][circle == 1]), std, 0.5
            ), f"Noise estimate is not correct ({noise_slice} {noise})"


def test_get_cylindrical_coordinates_short_axis():
    # load RV and LV mask
    mask = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask = mask["mask_3c"]
    # repeat mask 3 times for XYZ vector components
    mask_xyz = np.expand_dims(mask, axis=3)
    mask_xyz = np.repeat(mask_xyz, 3, axis=3)
    mask_xyz[mask_xyz == 2] = 0

    # load phantom cardiac coordinates
    cardiac_coordinates = np.load(os.path.join("tests", "data", "cardiac_coordinates.npz"))
    long_true = mask_xyz * cardiac_coordinates["longitudinal_component"]
    circ_true = mask_xyz * cardiac_coordinates["circumferential_component"]
    radial_true = mask_xyz * cardiac_coordinates["radial_component"]

    # mock settings
    settings = {}

    # mock slices
    slices = ["0.0"]

    settings["debug"] = False
    local_cardiac_coordinates = get_cylindrical_coordinates_short_axis(mask, mask, slices, settings)
    long_calculated = mask_xyz * local_cardiac_coordinates["long"]
    circ_calculated = mask_xyz * local_cardiac_coordinates["circ"]
    radial_calculated = mask_xyz * local_cardiac_coordinates["radi"]

    assert np.allclose(long_true, long_calculated)
    assert np.allclose(circ_true, circ_calculated)
    assert np.allclose(radial_true, radial_calculated)
