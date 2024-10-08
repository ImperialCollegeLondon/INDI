import numpy as np
from numpy.typing import NDArray

from extensions.extensions import convert_array_to_dict_of_arrays, convert_dict_of_arrays_to_array


def get_fa_md(eigv: NDArray, info, mask_3c, slices, logger) -> tuple[NDArray, NDArray, dict]:
    """
    Calculate FA and MD maps

    Parameters
    ----------
    eigv: eigenvalues

    Returns
    -------
    MD and FA arrays

    """
    eigv_array = convert_dict_of_arrays_to_array(eigv)
    mask_3c_array = convert_dict_of_arrays_to_array(mask_3c)

    md = np.expand_dims(np.mean(eigv_array, axis=-1), axis=-1)
    adjusted_norms = np.linalg.norm(eigv_array, axis=-1)  # adjust norms to "inf" to avoid division by 0
    adjusted_norms[adjusted_norms == 0] = np.inf
    fa = np.sqrt(3 / 2) * np.linalg.norm(eigv_array - md, axis=-1) / adjusted_norms
    md = np.squeeze(md, axis=-1)

    # turn values to nan where mask is 0
    md[mask_3c_array == 0] = np.nan
    fa[mask_3c_array == 0] = np.nan

    # get mean and std of dti["md"] and dti["fa"] in the myocardium
    var_names = ["MD", "FA"]
    for var in var_names:
        vals = eval(var.lower())[mask_3c_array > 0]
        if var == "MD":
            vals = 1e3 * vals
        logger.debug(
            "Median "
            + var
            + ": "
            + "%.2f" % np.nanmedian(vals)
            + " ["
            + "%.2f" % np.nanpercentile(vals, 25)
            + ", "
            + "%.2f" % np.nanpercentile(vals, 75)
            + "]"
        )

    md = convert_array_to_dict_of_arrays(md, slices)
    fa = convert_array_to_dict_of_arrays(fa, slices)

    return md, fa, info
