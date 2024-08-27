import logging
import os
import tempfile

import numpy as np
import pandas as pd

from extensions.registration_ex_vivo.registration import RegistrationExVivo


def test_registration_ex_vivo():
    n_images = 12
    im_size = 10
    # 12 total images = 3 slices * 2 repetitions in 2 different directions
    indices = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    slices = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    images = [np.random.rand(im_size, im_size) for _ in range(n_images)]

    # Check that the dataframe is correctly created
    assert len(indices) == len(slices) == len(images)

    data = pd.DataFrame(
        {
            "image": images,
            "index": indices,
            "slice_integer": slices,
            "b_value": [i for i in range(n_images)],
            "diffusion_direction": [np.random.rand(3) for _ in range(n_images)],
        }
    )

    context = {
        "data": data.copy(),
        "info": {
            "img_size": (im_size, im_size),
        },
    }

    tmp = tempfile.mkdtemp()
    settings = {
        "complex_data": False,
        "code_path": os.path.abspath("./"),
        "session": tmp,
        "debug_folder": tmp,
        "registration_mask_scale": 1.0,
    }

    registration = RegistrationExVivo(context, settings, logging.getLogger(__name__))

    registration.run()

    # check all the columns are the same
    assert set(context["data"].columns) == set(data.columns)
