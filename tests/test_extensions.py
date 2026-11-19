from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd

from indi.extensions.extensions import get_cylindrical_coordinates_short_axis, get_snr_maps


def test_get_snr_maps(mask_3c: np.ndarray, data_dir: Path, logger: Logger):
    """Check SNR maps and noise estimates computed from simulated data."""
    # Here I am loading data created by create_dwi_dicoms.py in DTI_numerical_phantom_3D repo
    # Load table with pixel values, b_values and directions
    # I am going to load data simulated with an SNR of 20 in the myocardial wall
    data = pd.read_pickle(data_dir / "cdti_table_snr_20.zip")
    data["slice_position"] = np.repeat(["0.0"], len(data))
    data["b_value_original"] = data["b_value"]
    data["slice_integer"] = 0
    data["diffusion_direction"] = data["direction"]

    simulated_gain = 1000
    snr_true = 20
    lv_myocardial_noise_true = simulated_gain / snr_true

    settings = {"debug": False}

    info = {}

    # Slice index for testing
    slices = [0]

    img_shape = data["image"][0].shape
    # Keep one average image per slice, with shape (num_slices, height, width).
    average_images = np.zeros((1, img_shape[0], img_shape[1]))

    _, noise_calculated, snr_b0_lv_calculated, _ = get_snr_maps(
        data, mask_3c, average_images, slices, settings, logger, info
    )

    # Check if SNR in the LV myo matches the simulated SNR
    assert np.allclose(snr_b0_lv_calculated[0]["median"], snr_true, atol=0.5 * snr_true)

    # Check also if the noise in the LV myocardium matches the simulated value
    noise_mat = noise_calculated[0]["0_0.58_0.58_0.58"]
    noise_lv_calculated = np.mean(noise_mat[mask_3c[0] == 1])
    assert np.allclose(
        noise_lv_calculated,
        lv_myocardial_noise_true,
        atol=lv_myocardial_noise_true * 0.5,
    )


def test_get_cylindrical_coordinates_short_axis(mask_3c: np.ndarray, data_dir: Path):
    """Confirm cardiac coordinate vectors match phantom references."""
    # Repeat mask 3 times for XYZ vector components
    mask_xyz = np.repeat(mask_3c[..., np.newaxis], 3, axis=3)
    mask_xyz[mask_xyz == 2] = 0

    # Load phantom cardiac coordinates
    with np.load(data_dir / "cardiac_coordinates.npz") as data:
        cardiac_coordinates = {key: data[key] for key in data.files}
    long_true = mask_xyz * cardiac_coordinates["longitudinal_component"]
    circ_true = mask_xyz * cardiac_coordinates["circumferential_component"]
    radial_true = mask_xyz * cardiac_coordinates["radial_component"]

    local_cardiac_coordinates = get_cylindrical_coordinates_short_axis(mask_3c)
    long_calculated = mask_xyz * local_cardiac_coordinates["long"]
    circ_calculated = mask_xyz * local_cardiac_coordinates["circ"]
    radial_calculated = mask_xyz * local_cardiac_coordinates["radi"]

    assert np.allclose(long_true, long_calculated)
    assert np.allclose(circ_true, circ_calculated)
    assert np.allclose(radial_true, radial_calculated)
