import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from extensions.rotation.rotation import Rotation, rotate_vector


@pytest.mark.parametrize(
    "vector, angle, axis, expected",
    [
        (np.asarray([1, 0, 0]), 90, "z", np.asarray([0, -1, 0])),
        (np.asarray([0, 1, 0]), 90, "z", np.asarray([1, 0, 0])),
        (np.asarray([1, 0, 0]), 90, "y", np.asarray([0, 0, 1])),
        (np.asarray([0, 0, 1]), 90, "y", np.asarray([-1, 0, 0])),
        (np.asarray([0, 1, 0]), 90, "x", np.asarray([0, 0, -1])),
        (np.asarray([0, 0, 1]), 90, "x", np.asarray([0, 1, 0])),
        (np.asarray([1, 0, 0]), 180, "z", np.asarray([-1, 0, 0])),
        (np.asarray([0, 1, 0]), 180, "z", np.asarray([0, -1, 0])),
        (np.asarray([1, 0, 0]), 180, "y", np.asarray([-1, 0, 0])),
        (np.asarray([0, 0, 1]), 180, "y", np.asarray([0, 0, -1])),
        (np.asarray([0, 1, 0]), 180, "x", np.asarray([0, -1, 0])),
        (np.asarray([0, 0, 1]), 180, "x", np.asarray([0, 0, -1])),
        (np.asarray([1, 0, 0]), -90, "z", np.asarray([0, 1, 0])),
        (np.asarray([0, 1, 0]), -90, "z", np.asarray([-1, 0, 0])),
        (np.asarray([1, 0, 0]), -90, "y", np.asarray([0, 0, -1])),
        (np.asarray([0, 0, 1]), -90, "y", np.asarray([1, 0, 0])),
        (np.asarray([0, 1, 0]), -90, "x", np.asarray([0, 0, 1])),
        (np.asarray([0, 0, 1]), -90, "x", np.asarray([0, -1, 0])),
    ],
)
def test_rotate_vector(vector, angle, axis, expected):
    rotated_vector = rotate_vector(vector, angle, axis)
    assert np.allclose(rotated_vector, expected)


def test_rotate_vector_error():
    with pytest.raises(ValueError):
        rotate_vector(np.asarray([1, 0, 0]), 90, "w")


def get_rotation_object(rotation=True, axis="x", angle=90, n_configs=2, n_average=2, n_slices=3, im_size=10):
    indices_one_slice = sum([[i] * n_average for i in range(n_configs)], start=[])
    indices = indices_one_slice * n_slices
    slices = sum([[i] * n_configs * n_average for i in range(n_slices)], start=[])

    n_images = len(indices)
    assert n_images == n_configs * n_average * n_slices

    images = [np.random.rand(im_size, im_size) for _ in range(n_images)]

    im_pos = sum([[(0, 0, i)] * n_configs * n_average for i in range(n_slices)], start=[])

    # Check that the dataframe is correctly created
    assert len(indices) == len(slices) == len(images) == len(im_pos)
    data = pd.DataFrame(
        {
            "image": images,
            "diff_config": indices,
            "slice_integer": slices,
            "b_value": [i for i in range(n_images)],
            "b_value_original": [i for i in range(n_images)],
            "diffusion_direction": [np.random.rand(3) for _ in range(n_images)],
            "diffusion_direction_original": [np.random.rand(3) for _ in range(n_images)],
            "image_position": im_pos,
        }
    )

    info = {
        "img_size": (im_size, im_size),
        "integer_to_image_positions": {i: np.random.rand(3) for i in np.arange(n_slices)},
        "pixel_spacing": (0.5, 0.5, 1),
    }

    context = {
        "data": data.copy(),
        "info": info,
        "slices": np.arange(n_slices),
        "ref_images": {i: {"image": np.random.rand(im_size, im_size)} for i in np.arange(n_slices)},
        "dti": {
            "snr": {i: {"image": np.random.rand(im_size, im_size)} for i in np.arange(n_slices)},
        },
    }

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)
    settings = {
        "complex_data": False,
        "rotate": rotation,
        "rotation_axis": axis,
        "rotation_angle": angle,
        "code_path": os.path.abspath("./"),
        "session": tmp,
        "results": tmp,
        "debug_folder": tmp,
        "debug": True,
        "ex_vivo": True,
        "registration_mask_scale": 1.0,
    }

    return Rotation(context, settings, logging.getLogger(__name__)), images, indices, slices, context, data, info


@pytest.mark.parametrize(
    "rotation, axis, angle",
    [
        (True, "x", 90),
        (True, "y", 90),
        (True, "z", 90),
        (True, "x", 180),
        (True, "y", 180),
        (True, "z", 180),
        (True, "x", -90),
        (True, "y", -90),
        (True, "z", -90),
        (False, "x", 90),
    ],
)
def test_rotation(rotation, axis, angle):
    rotation, _, _, _, context, data, info = get_rotation_object(rotation=rotation, axis=axis, angle=angle)

    rotation.run()

    # check all the columns are the same
    assert set(context["data"].columns) == set(data.columns), "Missing columns in the data"

    # It should change the image spacing if rotation is applied
    # if rotation:
    #     if axis == "x":
    #         assert context["info"]["pixel_spacing"] == (0.5, 0.5, 1), "Pixel spacing is not the same, when rotating around x axis"

    #     if axis == "y":
    #         assert context["info"]["pixel_spacing"] == (0.5, 1.0, 0.5), "Pixel spacing is not the same when rotating around y axis"

    #     if axis == "z":
    #         assert context["info"]["pixel_spacing"] == (1.0, 0.5, 0.5), "Pixel spacing is not the same when rotating around z axis"

    # also check that the image positions are correctly updated
