import logging

import numpy as np

from indi.extensions import ha_line_profiles


def _circle_contour(center_x, center_y, radius, n_points=360):
    """Create integer contour coordinates for a circle."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xs = (center_x + radius * np.cos(angles)).astype(np.int32)
    ys = (center_y + radius * np.sin(angles)).astype(np.int32)
    return np.column_stack((xs, ys))


def test_get_ha_line_profiles_(monkeypatch, logger, results_settings):
    """Exercise HA line profiles workflow with deterministic contours."""

    logger.setLevel(logging.DEBUG)

    # Image dimensions and center
    image_height = image_width = 101
    center = (50, 50)  # (y, x)

    # Two slices: one where endocardium is provided, one where it's not
    slice_count = 2
    slices = [0, 1]

    # Create HA maps: simple radial gradient so values exist inside mask
    ha_values = np.zeros((slice_count, image_height, image_width), dtype=float)
    yy, xx = np.meshgrid(
        np.arange(image_height),
        np.arange(image_width),
        indexing="ij",
    )
    radial_distance = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    ha_values[:] = radial_distance

    # Create mask_3c with myocardium region as 1 (big disk)
    mask_3c = np.zeros((slice_count, image_height, image_width), dtype=np.uint8)
    radius_epi = 30
    myocardium_mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius_epi**2
    mask_3c[:, myocardium_mask] = 1  # myocardium labeled as 1

    # segmentation: make slice 0 have endocardium (non-empty), slice 1 empty
    segmentation = {0: {"endocardium": np.array([1])}, 1: {"endocardium": np.array([])}}

    # LV centers for each slice (y, x)
    lv_centres = {0: center, 1: center}

    # average images (not used for assertions)
    average_images = np.zeros((slice_count, image_height, image_width), dtype=float)

    # info: pixel spacing
    info = {"pixel_spacing": (1.0, 1.0)}

    # Provide deterministic contours by monkeypatching the contour/spline functions
    epi_contour = _circle_contour(center[1], center[0], radius_epi, n_points=360)
    endo_contour = _circle_contour(center[1], center[0], radius_epi // 3, n_points=180)

    def fake_get_sa_contours(mask_uint8):
        # return epi, endo contours (x,y pairs)
        return epi_contour, endo_contour

    def fake_get_epi_contour(mask_uint8):
        return epi_contour

    def fake_spline_interpolate_contour(contour, n_points, join_ends=False):
        # return a denser version of the provided contour (float)
        if contour.shape[0] == 0:
            return contour
        idxs = np.linspace(0, contour.shape[0] - 1, n_points).astype(int)
        return contour[idxs].astype(float)

    monkeypatch.setattr(ha_line_profiles, "get_sa_contours", fake_get_sa_contours)
    monkeypatch.setattr(ha_line_profiles, "get_epi_contour", fake_get_epi_contour)
    monkeypatch.setattr(ha_line_profiles, "spline_interpolate_contour", fake_spline_interpolate_contour)

    # Call the function under test
    (
        ha_profiles,
        wall_thickness,
        bullseye_maps,
        distance_endo_maps,
        distance_epi_maps,
        distance_transmural_maps,
        ha_line_profiles_summary,
    ) = ha_line_profiles.get_ha_line_profiles_and_distance_maps(
        HA=ha_values,
        lv_centres=lv_centres,
        slices=slices,
        mask_3c=mask_3c,
        segmentation=segmentation,
        settings=results_settings,
        info=info,
        average_images=average_images,
        logger=logger,
        ventricle="LV",
    )

    # Basic assertions about returned structures
    assert set(ha_profiles.keys()) == set(slices)
    assert set(wall_thickness.keys()) == set(slices)
    assert bullseye_maps.shape == mask_3c.shape
    for distance_map in (
        distance_endo_maps,
        distance_epi_maps,
        distance_transmural_maps,
    ):
        assert distance_map.shape == mask_3c.shape
    assert set(ha_line_profiles_summary.keys()) == set(slices)

    # For each slice, lp_matrix should have rows == number of epi contour points used
    for slice_index in slices:
        lp_matrix = ha_profiles[slice_index]["lp_matrix"]
        # number of rows equals number of points in epi_contour returned earlier
        assert lp_matrix.shape[1] == 50  # interpolation length fixed in function
        assert lp_matrix.shape[0] == epi_contour.shape[0]

        # wall thickness list length should match number of epi points
        wt_list = wall_thickness[slice_index]["wt"]
        assert len(wt_list) == epi_contour.shape[0]

        # distance maps: inside mask should be finite, outside NaN
        inside_mask = mask_3c[slice_index] != 0
        assert np.all(np.isnan(distance_endo_maps[slice_index][~inside_mask]))
        assert np.all(np.isnan(distance_epi_maps[slice_index][~inside_mask]))
        assert np.all(np.isnan(distance_transmural_maps[slice_index][~inside_mask]))
        # inside should contain at least some finite values
        assert np.isfinite(distance_endo_maps[slice_index][inside_mask]).sum() > 0
        assert np.isfinite(distance_epi_maps[slice_index][inside_mask]).sum() > 0
        assert np.isfinite(distance_transmural_maps[slice_index][inside_mask]).sum() > 0

    # ha_lines_profiles_2 entries should contain median/q25/q75 arrays and slope/r_sq keys
    for slice_index in slices:
        summary = ha_line_profiles_summary[slice_index]
        assert all(key in summary for key in ("median", "q25", "q75", "iqr"))
        assert all(key in summary for key in ("r_sq", "slope", "y_pred"))
        # median should be a 1D numpy array
        assert isinstance(summary["median"], np.ndarray)
        # r_sq should be a float (or convertible)
        assert np.isscalar(summary["r_sq"])


def test_fix_angle_wrap_no_jump():
    """Ensure angle series without jumps remains unchanged."""
    angles = np.array([0.0, 10.0, 20.0, 25.0, 30.0])
    corrected_angles = ha_line_profiles.fix_angle_wrap(angles, angle_jump=45)
    # no change expected
    assert np.allclose(corrected_angles, angles, equal_nan=True)


def test_fix_angle_wrap_with_jump():
    """Ensure large angle jump zeros out subsequent values."""
    angles = np.array([0.0, 10.0, 20.0, 120.0, 130.0])
    corrected_angles = ha_line_profiles.fix_angle_wrap(angles, angle_jump=45)
    # after detected jump (20 -> 120) all subsequent values should be NaN
    assert np.all(np.isnan(corrected_angles[3:]))
    # earlier values preserved
    assert np.array_equal(corrected_angles[:3], angles[:3])
