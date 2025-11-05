import logging

import numpy as np

from indi.extensions import ha_line_profiles


def _circle_contour(center_x, center_y, radius, n_points=360):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xs = (center_x + radius * np.cos(angles)).astype(np.int32)
    ys = (center_y + radius * np.sin(angles)).astype(np.int32)
    return np.column_stack((xs, ys))


def test_get_ha_line_profiles_(monkeypatch, tmp_path):
    # Create temporary folders expected by the function
    results_dir = tmp_path / "results"
    results_b = results_dir / "results_b"
    debug_folder = tmp_path / "debug"
    results_b.mkdir(parents=True)
    debug_folder.mkdir(parents=True)

    settings = {"results": str(results_dir), "debug_folder": str(debug_folder), "debug": False}
    # logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    # Image dimensions and center
    H = W = 101
    center = (50, 50)  # (y, x)

    # Two slices: one where endocardium is provided, one where it's not
    n_slices = 2
    slices = [0, 1]

    # Create HA maps: simple radial gradient so values exist inside mask
    HA = np.zeros((n_slices, H, W), dtype=float)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    HA_val = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    HA[:] = HA_val

    # Create mask_3c with myocardium region as 1 (big disk)
    mask_3c = np.zeros((n_slices, H, W), dtype=np.uint8)
    radius_epi = 30
    rr = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius_epi**2
    mask_3c[:, rr] = 1  # myocardium labeled as 1

    # segmentation: make slice 0 have endocardium (non-empty), slice 1 empty
    segmentation = {0: {"endocardium": np.array([1])}, 1: {"endocardium": np.array([])}}

    # LV centers for each slice (y, x)
    lv_centres = {0: center, 1: center}

    # average images (not used for assertions)
    average_images = np.zeros((n_slices, H, W), dtype=float)

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
        ha_lines_profiles,
        wall_thickness,
        bullseye_maps,
        distance_endo_maps,
        distance_epi_maps,
        distance_transmural_maps,
        ha_lines_profiles_2,
    ) = ha_line_profiles.get_ha_line_profiles_and_distance_maps(
        HA=HA,
        lv_centres=lv_centres,
        slices=slices,
        mask_3c=mask_3c,
        segmentation=segmentation,
        settings=settings,
        info=info,
        average_images=average_images,
        logger=logger,
        ventricle="LV",
    )

    # Basic assertions about returned structures
    assert set(ha_lines_profiles.keys()) == set(slices)
    assert set(wall_thickness.keys()) == set(slices)
    assert bullseye_maps.shape == mask_3c.shape
    assert distance_endo_maps.shape == mask_3c.shape
    assert distance_epi_maps.shape == mask_3c.shape
    assert distance_transmural_maps.shape == mask_3c.shape
    assert set(ha_lines_profiles_2.keys()) == set(slices)

    # For each slice, lp_matrix should have rows == number of epi contour points used
    for s in slices:
        lp_matrix = ha_lines_profiles[s]["lp_matrix"]
        # number of rows equals number of points in epi_contour returned earlier
        assert lp_matrix.shape[1] == 50  # interpolation length fixed in function
        assert lp_matrix.shape[0] == epi_contour.shape[0]

        # wall thickness list length should match number of epi points
        wt_list = wall_thickness[s]["wt"]
        assert len(wt_list) == epi_contour.shape[0]

        # distance maps: inside mask should be finite, outside NaN
        inside_mask = mask_3c[s] != 0
        assert np.all(np.isnan(distance_endo_maps[s][~inside_mask]))
        assert np.all(np.isnan(distance_epi_maps[s][~inside_mask]))
        assert np.all(np.isnan(distance_transmural_maps[s][~inside_mask]))
        # inside should contain at least some finite values
        assert np.isfinite(distance_endo_maps[s][inside_mask]).sum() > 0
        assert np.isfinite(distance_epi_maps[s][inside_mask]).sum() > 0
        assert np.isfinite(distance_transmural_maps[s][inside_mask]).sum() > 0

    # ha_lines_profiles_2 entries should contain median/q25/q75 arrays and slope/r_sq keys
    for s in slices:
        d = ha_lines_profiles_2[s]
        assert "median" in d and "q25" in d and "q75" in d and "iqr" in d
        assert "r_sq" in d and "slope" in d and "y_pred" in d
        # median should be a 1D numpy array
        assert isinstance(d["median"], np.ndarray)
        # r_sq should be a float (or convertible)
        assert np.isscalar(d["r_sq"])


def test_fix_angle_wrap_no_jump():
    lp = np.array([0.0, 10.0, 20.0, 25.0, 30.0])
    out = ha_line_profiles.fix_angle_wrap(lp, angle_jump=45)
    # no change expected
    assert np.allclose(out, lp, equal_nan=True)


def test_fix_angle_wrap_with_jump():
    lp = np.array([0.0, 10.0, 20.0, 120.0, 130.0])
    out = ha_line_profiles.fix_angle_wrap(lp, angle_jump=45)
    # after detected jump (20 -> 120) all subsequent values should be NaN
    assert np.isnan(out[3])
    assert np.isnan(out[4])
    # earlier values preserved
    assert out[0] == 0.0 and out[1] == 10.0 and out[2] == 20.0
