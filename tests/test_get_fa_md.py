from logging import Logger

import numpy as np
from scipy.stats import moment

from indi.extensions.get_fa_md import get_fa_md


def test_get_fa_md(mask_3c: np.ndarray, tensor_true: np.ndarray, logger: Logger):
    """Verify FA and MD calculations against phantom ground truth."""

    tensor_true = np.nan_to_num(tensor_true)
    slices = [0]
    info = {}

    eigenvalues, _ = np.linalg.eigh(tensor_true)
    (
        md_calculated,
        fa_calculated,
        mode_calculated,
        frob_norm_calculated,
        mag_anisotropy_calculated,
        _,
    ) = get_fa_md(eigenvalues, info, mask_3c, slices, logger)

    e1, e2, e3 = (2 * 1e-3, 1 * 1e-3, 0.5 * 1e-3)
    eigv = np.array([[[e1, e2, e3]]])

    second_moment = moment(eigv, moment=2, axis=-1)
    third_moment = moment(eigv, moment=3, axis=-1)

    mode_true = np.sqrt(2) * third_moment * second_moment ** (-3 / 2)
    frob_norm_true = np.linalg.norm(eigv, axis=-1)
    mag_anisotropy_true = np.sqrt(3 * second_moment)
    md_true = (e1 + e2 + e3) / 3
    fa_true = np.sqrt(1 / 2) * (
        np.sqrt((e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2) / np.sqrt(e1**2 + e2**2 + e3**2)
    )

    metrics = {
        "MD": (md_calculated, md_true),
        "FA": (fa_calculated, fa_true),
        "mode": (mode_calculated, mode_true),
        "Frobenius norm": (frob_norm_calculated, frob_norm_true),
        "magnitude anisotropy": (mag_anisotropy_calculated, mag_anisotropy_true),
    }
    mask = mask_3c == 1
    for metric, (calculated, expected_mean) in metrics.items():
        values = calculated[mask]
        assert np.allclose(np.mean(values), expected_mean), f"Unexpected {metric} mean"
        assert np.allclose(np.std(values), 0.0), f"Unexpected {metric} standard deviation"
