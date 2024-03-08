import numpy as np
from numpy.typing import NDArray


def get_fa_md(eigv: NDArray, info, mask_3c, slices, logger) -> [NDArray, NDArray, dict]:
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
    md[mask_3c != 1] = np.nan
    fa[mask_3c != 1] = np.nan

    # get mean and std of dti["md"] and dti["fa"] in the myocardium
    var_names = ["MD", "FA"]
    for var in var_names:
        info["DTI_" + var] = {}
        for slice_idx in slices:
            vals = eval(var.lower())[slice_idx][mask_3c[slice_idx] == 1]
            if var == "MD":
                vals = 1e3 * vals
            info["DTI_" + var][str(slice_idx).zfill(2)] = (
                "%.2f" % np.nanmean(vals) + " +/- " + "%.2f" % np.nanstd(vals)
            )
            logger.debug(
                "Mean "
                + var
                + " for slice "
                + str(slice_idx).zfill(2)
                + ": "
                + str("%.2f" % np.nanmean(vals) + " +/- " + "%.2f" % np.nanstd(vals))
            )

    return md, fa, info
