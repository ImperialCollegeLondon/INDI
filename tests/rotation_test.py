import numpy as np
import pytest

from extensions.rotation import rotate_vector


@pytest.mark.parametrize(
    "vector, angle, axis, expected",
    [
        (np.asarray([1, 0, 0]), 90, "z", np.asarray([0, 1, 0])),
        (np.asarray([0, 1, 0]), 90, "z", np.asarray([-1, 0, 0])),
        (np.asarray([1, 0, 0]), 90, "y", np.asarray([0, 0, -1])),
        (np.asarray([0, 0, 1]), 90, "y", np.asarray([1, 0, 0])),
        (np.asarray([0, 1, 0]), 90, "x", np.asarray([0, 0, 1])),
        (np.asarray([0, 0, 1]), 90, "x", np.asarray([0, -1, 0])),
        (np.asarray([1, 0, 0]), 180, "z", np.asarray([-1, 0, 0])),
        (np.asarray([0, 1, 0]), 180, "z", np.asarray([0, -1, 0])),
        (np.asarray([1, 0, 0]), 180, "y", np.asarray([-1, 0, 0])),
        (np.asarray([0, 0, 1]), 180, "y", np.asarray([0, 0, -1])),
        (np.asarray([0, 1, 0]), 180, "x", np.asarray([0, -1, 0])),
        (np.asarray([0, 0, 1]), 180, "x", np.asarray([0, 0, -1])),
        (np.asarray([1, 0, 0]), -90, "z", np.asarray([0, -1, 0])),
        (np.asarray([0, 1, 0]), -90, "z", np.asarray([1, 0, 0])),
        (np.asarray([1, 0, 0]), -90, "y", np.asarray([0, 0, 1])),
        (np.asarray([0, 0, 1]), -90, "y", np.asarray([-1, 0, 0])),
        (np.asarray([0, 1, 0]), -90, "x", np.asarray([0, 0, -1])),
        (np.asarray([0, 0, 1]), -90, "x", np.asarray([0, 1, 0])),
    ],
)
def test_rotate_vector(vector, angle, axis, expected):
    rotated_vector = rotate_vector(vector, angle, axis)
    assert np.allclose(rotated_vector, expected)
