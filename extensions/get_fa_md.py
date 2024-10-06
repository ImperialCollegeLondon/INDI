import numpy as np
from numpy.typing import NDArray


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
    md = np.expand_dims(np.mean(eigv, axis=-1), axis=-1)
    adjusted_norms = np.linalg.norm(eigv, axis=-1)  # adjust norms to "inf" to avoid division by 0
    adjusted_norms[adjusted_norms == 0] = np.inf
    fa = np.sqrt(3 / 2) * np.linalg.norm(eigv - md, axis=-1) / adjusted_norms
    md = np.squeeze(md, axis=-1)

    # turn values to nan where mask is 0
    md[mask_3c == 0] = np.nan
    fa[mask_3c == 0] = np.nan

    # get mean and std of dti["md"] and dti["fa"] in the myocardium
    var_names = ["MD", "FA"]
    for var in var_names:
        vals = eval(var.lower())[mask_3c > 0]
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

    return md, fa, info
