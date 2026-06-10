import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import moment

from indi.extensions.extensions import get_cylindrical_coordinates_short_axis, get_snr_maps
from indi.extensions.get_fa_md import get_fa_md


def test_get_fa_md():
    """Verify FA and MD calculations against phantom ground truth."""

    # load tensor from numerical phantom
    tensor_true = np.load(os.path.join("tests", "data", "DT.npz"))
    tensor_true = tensor_true["DT"]
    tensor_true = np.nan_to_num(tensor_true)

    # load RV and LV mask
    mask = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    mask = mask["mask_3c"]

    # mock slices
    slices = [0]

    # mock info dictionary
    info = {}

    # mock logger
    logger = logging.getLogger(__name__)

    # calculate eigenvalues from tensor
    eigenvalues, _ = np.linalg.eigh(tensor_true)
    # get FA and MD from eigenvalues
    md_calculated, fa_calculated, mode_calculated, frob_norm_calculated, mag_anisotropy_calculated, _ = get_fa_md(
        eigenvalues, info, mask, slices, logger
    )

    mean_md_calculated, std_md_calculated = [
        np.mean(md_calculated[mask == 1]),
        np.std(md_calculated[mask == 1]),
    ]
    mean_fa_calculated, std_fa_calculated = [
        np.mean(fa_calculated[mask == 1]),
        np.std(fa_calculated[mask == 1]),
    ]
    mean_mode_calculated, std_mode_calculated = [
        np.mean(mode_calculated[mask == 1]),
        np.std(mode_calculated[mask == 1]),
    ]
    mean_frob_norm_calculated, std_frob_norm_calculated = [
        np.mean(frob_norm_calculated[mask == 1]),
        np.std(frob_norm_calculated[mask == 1]),
    ]
    mean_mag_anisotropy_calculated, std_mag_anisotropy_calculated = [
        np.mean(mag_anisotropy_calculated[mask == 1]),
        np.std(mag_anisotropy_calculated[mask == 1]),
    ]

    # correct metrics from simulated phantom data
    e1, e2, e3 = (2 * 1e-3, 1 * 1e-3, 0.5 * 1e-3)

    eigv = np.array([[[e1, e2, e3]]])  # shape (1, 1, 3)

    second_moment = moment(eigv, moment=2, axis=-1)
    third_moment = moment(eigv, moment=3, axis=-1)

    mode_true = np.sqrt(2) * third_moment * second_moment ** (-3 / 2)

    # get Frobenius norm of the tensor
    frob_norm_true = np.linalg.norm(eigv, axis=-1)

    # get the magnitude of anisotropy
    mag_anisotropy_true = np.sqrt(3 * second_moment)

    md_true = (e1 + e2 + e3) / 3

    fa_true = np.sqrt(1 / 2) * (
        np.sqrt((e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2) / np.sqrt(e1**2 + e2**2 + e3**2)
    )

    assert np.allclose(mean_md_calculated, md_true)
    assert np.allclose(std_md_calculated, 0.0)

    assert np.allclose(mean_fa_calculated, fa_true)
    assert np.allclose(std_fa_calculated, 0.0)

    assert np.allclose(mean_mode_calculated, mode_true)
    assert np.allclose(std_mode_calculated, 0.0)

    assert np.allclose(mean_frob_norm_calculated, frob_norm_true)
    assert np.allclose(std_frob_norm_calculated, 0.0)

    assert np.allclose(mean_mag_anisotropy_calculated, mag_anisotropy_true)
    assert np.allclose(std_mag_anisotropy_calculated, 0.0)


def test_get_snr_maps():
    """Check SNR maps and noise estimates computed from simulated data."""
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
    data["slice_integer"] = 0
    data["diffusion_direction"] = data["direction"]

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

    # slice index for testing
    slices = [0]

    # mock average images dictionary
    img_shape = data["image"][0].shape
    average_images = np.zeros((1, img_shape[0], img_shape[1]))  # shape (num_slices, height, width)

    [_, noise_calculated, snr_b0_lv_calculated, _] = get_snr_maps(
        data, mask, average_images, slices, settings, logger, info
    )

    # check if SNR in the LV myo matches the simulated SNR
    assert np.allclose(snr_b0_lv_calculated[0]["median"], snr_true, atol=0.5 * snr_true)

    # check also if the noise in the LV myocardium matches the simulated value
    noise_mat = noise_calculated[0]["0_0.58_0.58_0.58"]
    noise_lv_calculated = np.mean(noise_mat[mask[0] == 1])
    assert np.allclose(noise_lv_calculated, lv_myocardial_noise_true, atol=lv_myocardial_noise_true * 0.5)


def test_get_cylindrical_coordinates_short_axis():
    """Confirm cardiac coordinate vectors match phantom references."""
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

    local_cardiac_coordinates = get_cylindrical_coordinates_short_axis(mask)
    long_calculated = mask_xyz * local_cardiac_coordinates["long"]
    circ_calculated = mask_xyz * local_cardiac_coordinates["circ"]
    radial_calculated = mask_xyz * local_cardiac_coordinates["radi"]

    assert np.allclose(long_true, long_calculated)
    assert np.allclose(circ_true, circ_calculated)
    assert np.allclose(radial_true, radial_calculated)
