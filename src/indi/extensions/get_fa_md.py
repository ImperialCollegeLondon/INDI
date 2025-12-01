import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import moment


def get_fa_md(
    eigv: NDArray, info: dict[str, Any], mask_3c: NDArray, slices: NDArray, logger: logging.Logger
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, dict]:
    """
    Calculate eigenvalue-based DTI scalars: MD, FA, mode, Frobenius norm, and magnitude of anisotropy.

    Args:
        eigv: eigenvalues
        info: dict
        mask_3c: segmentation mask
        slices: array with slice indices
        logger: logger

    Returns:
        md: mean diffusivity
        fa: fractional anisotropy
        mode: tensor mode
        frob_norm: Frobenius norm
        mag_anisotropy: magnitude of anisotropy
        info: dict
    """
    # get MD and FA
    md = np.expand_dims(np.mean(eigv, axis=-1), axis=-1)
    adjusted_norms = np.linalg.norm(eigv, axis=-1)  # adjust norms to "inf" to avoid division by 0
    adjusted_norms[adjusted_norms == 0] = np.inf
    fa = np.sqrt(3 / 2) * np.linalg.norm(eigv - md, axis=-1) / adjusted_norms
    md = np.squeeze(md, axis=-1)

    # get tensor mode
    # For voxels with abnormally low eigenvalues (probably originally a negative
    # eigenvalue), mode becomes unstable and goes beyond the [-1, 1] limit due to
    # rounding errors. We set these values to NaN.
    # Also, sometimes, mode is NaN because the moments are very close to 0.
    second_moment = moment(eigv, moment=2, axis=-1)
    third_moment = moment(eigv, moment=3, axis=-1)
    second_moment[second_moment == 0] = np.nan
    third_moment[third_moment == 0] = np.nan
    mode = np.sqrt(2) * third_moment * second_moment ** (-3 / 2)
    mode[(-1 > mode) | (mode > 1)] = np.nan

    # get Frobenius norm of the tensor
    frob_norm = np.linalg.norm(eigv, axis=-1)

    # get the magnitude of anisotropy
    mag_anisotropy = np.sqrt(3 * second_moment)

    # turn values to nan where mask is different from 1
    md[mask_3c != 1] = np.nan
    fa[mask_3c != 1] = np.nan
    mode[mask_3c != 1] = np.nan
    frob_norm[mask_3c != 1] = np.nan
    mag_anisotropy[mask_3c != 1] = np.nan

    # get mean and std of MD, FA in the myocardium
    var_names = ["MD", "FA"]
    for var in var_names:
        for slice_idx in slices:
            vals = eval(var.lower())[slice_idx][mask_3c[slice_idx] == 1]
            if var == "MD":
                vals = 1e3 * vals
            logger.debug(
                "Median "
                + var
                + " for slice "
                + str(slice_idx).zfill(2)
                + ": "
                + "%.2f" % np.nanmedian(vals)
                + " ["
                + "%.2f" % np.nanpercentile(vals, 25)
                + ", "
                + "%.2f" % np.nanpercentile(vals, 75)
                + "]"
            )

    return md, fa, mode, frob_norm, mag_anisotropy, info
