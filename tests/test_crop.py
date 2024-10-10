import logging
import os
import random
import tempfile

import numpy as np
import pandas as pd
import yaml

from extensions.crop.crop import Crop


def get_crop_object(n_configs=2, n_average=2, n_slices=3, im_size=10):
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
        "slices": np.arange(n_slices),
    }

    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "results_b"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "results_a"), exist_ok=True)
    settings = {
        "complex_data": False,
        "code_path": os.path.abspath("./"),
        "session": tmp,
        "results": tmp,
        "debug_folder": tmp,
        "debug": True,
        "ex_vivo": True,
        "registration_mask_scale": 1.0,
    }

    return Crop(context, settings, logging.getLogger(__name__)), images, indices, slices, context, data


def test_crop():
    crop = get_crop_object(n_slices=10, im_size=256)[0]
    slice_a, slice_b = random.randint(0, 10), random.randint(0, 10)
    slice_a, slice_b = min(slice_a, slice_b), max(slice_a, slice_b)
    slice_b = slice_b if slice_b != slice_a else (slice_b + 1) % 10

    row_a, row_b = random.randint(0, 256), random.randint(0, 256)
    row_a, row_b = min(row_a, row_b), max(row_a, row_b)
    row_b = row_b if row_b != row_a else (row_b + 1) % 256

    col_a, col_b = random.randint(0, 256), random.randint(0, 256)
    col_b = col_b if col_b != col_a else (col_b + 1) % 256
    col_a, col_b = min(col_a, col_b), max(col_a, col_b)

    crop_settings = dict(col=[col_a, col_b], row=[row_a, row_b], slice=[slice_a, slice_b])

    with open(os.path.join(crop.settings["session"], "crop.yaml"), "w") as f:
        yaml.dump(crop_settings, f)

    crop.run()
    data = crop.context["data"]

    assert len(data["slice_integer"].unique()) == slice_b - slice_a, "Number of slices is not correct"
    assert data["image"][data["slice_integer"] == slice_a].values[0].shape == (
        row_b - row_a,
        col_b - col_a,
    ), "Image size is not correct"

    assert crop.context["info"]["img_size"] == (row_b - row_a, col_b - col_a), "Image size is updated correctly"
    assert crop.context["info"]["n_slices"] == slice_b - slice_a, "Number of slices is updated correctly"

    slices = np.arange(10)
    assert all(crop.context["slices"] == slices[slice_a:slice_b]), "Slices are not updated correctly"
