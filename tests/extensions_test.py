import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from extensions.extensions import (
    convert_array_to_dict_of_arrays,
    convert_dict_of_arrays_to_array,
    get_cardiac_coordinates_short_axis,
    get_snr_maps,
)
from extensions.get_fa_md import get_fa_md
from extensions.segmentation.manual_segmentation import get_sa_contours
from extensions.segmentation.polygon_selector import spline_interpolate_contour


@pytest.fixture
def cardiac_coordinates():
    coords = np.load(os.path.join("tests", "data", "cardiac_coordinates.npz"))
    return coords


@pytest.fixture
def mask():
    mask_3c = np.load(os.path.join("tests", "data", "mask_3c.npz"))
    return mask_3c["mask_3c"]


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


@pytest.mark.parametrize("ventricle", ["LV", "RV"])
def test_get_cylindrical_coordinates_short_axis(cardiac_coordinates, mask, ventricle):
    # Fails on the RV

    # repeat mask 3 times for XYZ vector components
    mask_lv = mask == 1
    mask_rv = mask == 2

    if ventricle == "RV":
        mask = mask > 0
    else:
        mask = mask == 1
    coords = np.where(mask == 1)

    v_centre_true_x = np.mean(coords[2])
    v_centre_true_y = np.mean(coords[1])

    if ventricle == "RV":
        mask_xyz = np.expand_dims(mask_rv, axis=3)
    else:
        mask_xyz = np.expand_dims(mask_lv, axis=3)

    mask_xyz = np.repeat(mask_xyz, 3, axis=3)

    # load phantom cardiac coordinates
    long_true = mask_xyz * cardiac_coordinates["longitudinal_component"]
    circ_true = mask_xyz * cardiac_coordinates["circumferential_component"]
    radial_true = mask_xyz * cardiac_coordinates["radial_component"]

    # mock settings
    settings = {}

    # mock slices
    slices = np.arange(len(mask))

    settings["debug"] = False
    dti = {}
    info = {}
    if ventricle == "LV":
        mask[mask == 2] = 0
    mask = convert_array_to_dict_of_arrays(mask, slices)
    segmentation = {}
    for slice_idx in slices:
        segmentation[slice_idx] = {}
        if ventricle == "LV":
            epi_contour, endo_contour = get_sa_contours(mask_lv[slice_idx])
            epi_len = len(epi_contour)
            endo_len = len(endo_contour)
            epi_contour = spline_interpolate_contour(epi_contour, 20, join_ends=False)
            epi_contour = spline_interpolate_contour(epi_contour, epi_len, join_ends=False)
            segmentation[slice_idx]["epicardium"], segmentation[slice_idx]["endocardium"] = epi_contour, endo_contour
            if endo_len > 10:
                endo_contour = spline_interpolate_contour(endo_contour, 20, join_ends=False)
                endo_contour = spline_interpolate_contour(endo_contour, endo_len, join_ends=False)
        else:
            epi_contour_rv, endo_contour_rv = get_sa_contours((mask_lv + mask_rv)[slice_idx])
            epi_len = len(epi_contour_rv)
            endo_len = len(endo_contour_rv)
            epi_contour_rv = spline_interpolate_contour(epi_contour_rv, 20, join_ends=False)
            epi_contour_rv = spline_interpolate_contour(epi_contour_rv, epi_len, join_ends=False)
            if endo_len > 10:
                endo_contour_rv = spline_interpolate_contour(endo_contour_rv, 20, join_ends=False)
                endo_contour_rv = spline_interpolate_contour(endo_contour_rv, endo_len, join_ends=False)
            segmentation[slice_idx]["epicardium_rv"] = epi_contour_rv
            segmentation[slice_idx]["endocardium_rv"] = endo_contour_rv

    local_cardiac_coordinates, v_center, _ = get_cardiac_coordinates_short_axis(
        mask, segmentation, slices, len(slices), settings, dti, None, info, ventricle
    )

    long_calculated = mask_xyz * convert_dict_of_arrays_to_array(local_cardiac_coordinates["long"])
    circ_calculated = mask_xyz * convert_dict_of_arrays_to_array(local_cardiac_coordinates["circ"])
    radial_calculated = mask_xyz * convert_dict_of_arrays_to_array(local_cardiac_coordinates["radi"])

    # assert np.isclose(v_centre_true_x, v_center[0][1], atol=2), f"v_centre_true_x: {v_centre_true_x}, v_center[0][1]: {v_center[0][1]}"
    # assert np.isclose(v_centre_true_y, v_center[0][0], atol=2), f"v_centre_true_y: {v_centre_true_y}, v_center[0][0]: {v_center[0][0]}"

    plt.subplot(121)
    plt.imshow(long_calculated[0, :, :, 2])
    plt.title("long calc z")
    plt.colorbar()
    plt.subplot(122)
    plt.title("long true z")
    plt.imshow(long_true[0, :, :, 2])
    plt.colorbar()
    plt.show()

    plt.subplot(221)
    plt.imshow(circ_calculated[0, :, :, 0])
    plt.title("Circumferential calc x")
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()
    plt.subplot(222)
    plt.title("Circumferential calc y")
    plt.imshow(circ_calculated[0, :, :, 1])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()

    plt.subplot(223)
    plt.title("Circumferential true x")
    plt.imshow(circ_true[0, :, :, 0])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()
    plt.subplot(224)
    plt.title("Circumferential true y")
    plt.imshow(circ_true[0, :, :, 1])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()

    plt.show()

    plt.subplot(221)
    plt.title("Radial calc x")
    plt.imshow(radial_calculated[0, :, :, 0])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()
    plt.subplot(222)
    plt.title("Radial calc y")
    plt.imshow(radial_calculated[0, :, :, 1])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(radial_true[0, :, :, 0])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.title("Radial true x")
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(radial_true[0, :, :, 1])
    plt.plot(v_center[0][1], v_center[0][0], "ro")
    plt.plot([v_centre_true_x], [v_centre_true_y], "bx")
    plt.title("Radial true y")
    plt.colorbar()

    plt.show()

    assert np.allclose(long_true, long_calculated), f"Longitudinal component is not correct for {ventricle}"
    assert np.allclose(
        circ_true, circ_calculated, atol=0.5
    ), f"Circumferential component is not correct for {ventricle}"
    assert np.allclose(radial_true, radial_calculated, atol=0.5), f"Radial component is not correct for {ventricle}"
