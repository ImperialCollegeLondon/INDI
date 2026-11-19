from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indi.extensions.tensor_fittings import dipy_tensor_fit

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("method", ["NLLS", "LS", "WLS", "RESTORE", "RNLLS", "RWLS"])
def test_dipy_tensor_fit(tmp_path, debug, method, logger, mask_3c):
    """Run tensor fitting pipeline and compare with ground-truth tensors.

    Args:
        tmp_path (Path): Temporary directory for debug and results outputs.
        debug (bool): Whether to enable debug outputs in settings.
        method (str): Tensor fitting method under test.
        logger (logging.Logger): Logger for diagnostics.
    """
    # Here I am loading data from DTI_numerical_phantom_3D repo
    # load table with pixel values, b_values and directions
    # For this test there is no noise, so SNR is infinite
    table = pd.read_pickle(DATA_DIR / "cdti_table.zip")
    # Rename some columns to match what is expected in the function
    table = table.rename(columns={"direction": "diffusion_direction"})
    table["slice_position"] = "0.0"
    table["slice_integer"] = 0
    table["to_be_removed"] = False
    table["bmatrix"] = None

    # load tensor that was used to create these pixel values
    tensor_true = np.load(DATA_DIR / "DT.npz")["DT"]
    slices = [0]
    average_images = {0: table["image"].mean(axis=0)}

    # mock settings
    debug_folder = tmp_path / "debug"
    (tmp_path / "results_b").mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)
    settings = {
        "debug": debug,
        "debug_folder": str(debug_folder),
        "results": str(tmp_path),
        "tensor_fit_method": method,
    }
    # mock info
    info = {
        "img_size": table["image"][0].shape,
        "n_slices": len(slices),
    }

    # calculate tensors with DiPy functions
    tensor, _, _, residuals_map, _, _, _ = dipy_tensor_fit(
        slices=slices,
        data=table,
        info=info,
        settings=settings,
        mask_3c=mask_3c,
        average_images=average_images,
        logger=logger,
        method=settings["tensor_fit_method"],
        quick_mode=False,
    )

    # assert that the tensors are the same
    mask = ~(np.isnan(tensor) | np.isnan(tensor_true))
    assert np.allclose(tensor[mask], tensor_true[mask], atol=1e-2)

    # check if residuals are close to zero
    # create a mask to ignore nan values
    mask = ~(np.isnan(residuals_map[0]))
    # because there is no noise, I expect the residuals to be very small
    assert np.isclose(np.median(residuals_map[0][mask]), 0, atol=1e-1)
