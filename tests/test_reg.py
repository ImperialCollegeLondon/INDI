import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from extensions.registration_ex_vivo.registration import RegistrationExVivo


def get_reg_object(reg_mode="none", n_configs=2, n_average=2, n_slices=3, im_size=10):
    indices_one_slice = sum([[i] * n_average for i in range(n_configs)], start=[])
    indices = indices_one_slice * n_slices
    slices = sum([[i] * n_configs * n_average for i in range(n_slices)], start=[])

    n_images = len(indices)
    assert n_images == n_configs * n_average * n_slices

    images = [np.random.rand(im_size, im_size) for _ in range(n_images)]

    # Check that the dataframe is correctly created
    assert len(indices) == len(slices) == len(images)

    data = pd.DataFrame(
        {
            "image": images,
            "diff_config": indices,
            "slice_integer": slices,
            "b_value": [i for i in range(n_images)],
            "b_value_original": [i for i in range(n_images)],
            "diffusion_direction": [np.random.rand(3) for _ in range(n_images)],
            "diffusion_direction_original": [np.random.rand(3) for _ in range(n_images)],
        }
    )

    context = {
        "data": data.copy(),
        "info": {
            "img_size": (im_size, im_size),
        },
    }

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)
    settings = {
        "complex_data": False,
        "ex_vivo_registration": reg_mode,
        "code_path": os.path.abspath("./"),
        "session": tmp,
        "results": tmp,
        "debug_folder": tmp,
        "debug": True,
        "ex_vivo": True,
        "registration_mask_scale": 1.0,
    }

    return RegistrationExVivo(context, settings, logging.getLogger(__name__)), images, indices, slices, context, data


@pytest.mark.parametrize("reg_mode", ["none", "rigid", "non_rigid"])
def test_registration_ex_vivo(reg_mode):
    registration, _, _, _, context, data = get_reg_object(reg_mode=reg_mode)

    registration.run()

    # check all the columns are the same
    assert set(context["data"].columns) == set(data.columns), "Missing columns in the data"


def test_registration_average_image():
    im_size = 10
    n_configs = 2
    n_slices = 3
    registration, images, indices, *_ = get_reg_object(
        im_size=im_size, n_configs=n_configs, n_average=2, n_slices=n_slices
    )

    lower_b_value_index = 0
    registered_images = [(img, idx) for img, idx in zip(images, indices)]
    average_image, rigid_reg_images, _ = registration._calculate_average_image(
        indices, lower_b_value_index, registered_images
    )

    assert average_image[0].shape == (im_size, im_size), "Average image shape is wrong"
    assert len(rigid_reg_images) == n_configs * n_slices, "Number of images after registration is wrong"
