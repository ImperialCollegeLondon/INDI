import logging
import os

import numpy as np
import pandas as pd

from indi.extensions.extensions import get_cylindrical_coordinates_short_axis, get_snr_maps
from indi.extensions.get_fa_md import get_fa_md


def test_get_fa_md():
    """test if this function calculates the correct FA and MD values"""

    # load tensor from numerical phantom
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = tensor_true["DT"]
    tensor_true = np.nan_to_num(tensor_true)

    # load RV and LV mask
    mask = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask = mask["mask_3c"]

    # mock slices
    slices = ["0.0"]

    # mock info dictionary
    info = {}

    # mock logger
    logger = logging.getLogger(__name__)

    # calculate eigenvalues from tensor
    eigenvalues, _ = np.linalg.eigh(tensor_true)
    # get FA and MD from eigenvalues
    md_calculated, fa_calculated, _ = get_fa_md(eigenvalues, info, mask, slices, logger)
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
    mask = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask = mask["mask_3c"]

    """test if this function calculates the correct SNR and noise maps"""
    # Here I am loading data created by create_dwi_dicoms.py in DTI_numerical_phantom_3D repo
    # load table with pixel values, b_values and directions
    # I am going to load data simulated with an SNR of 20 in the myocardial wall
    data = pd.read_pickle(os.path.join("tests", "data", "cdti_table_snr_20.zip"))
    slice_position_column = np.repeat(["0.0"], len(data))
    data["slice_position"] = slice_position_column
    data["b_value_original"] = data["b_value"]

    simulated_gain = 1000
    snr_true = 20
    lv_myocardial_noise_true = simulated_gain / snr_true

    # mock settings dictionary
    settings = {}

    settings["debug"] = False
    # mock logger
    logger = logging.getLogger(__name__)

    # mock info dictionary
    info = {}

    [_, noise_calculated, snr_b0_lv_calculated, _] = get_snr_maps(data, mask, settings, logger, info)

    # check if SNR in the LV myo matches the simulated SNR
    assert np.allclose(snr_b0_lv_calculated["0.0"]["mean"], snr_true, atol=10)

    # check also if the noise in the LV myocardium matches the simulated value
    noise_mat = noise_calculated["0.0"]["0_0.58_0.58_0.58"]
    noise_lv_calculated = np.mean(noise_mat[mask[0] == 1])
    assert np.allclose(noise_lv_calculated, lv_myocardial_noise_true, atol=10)


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
