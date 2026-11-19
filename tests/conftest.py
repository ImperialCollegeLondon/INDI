import logging
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def logger():
    """Provide a logger instance for tests."""
    return logging.getLogger("test_logger")


@pytest.fixture
def mask_3c():
    """Load the shared three-class phantom mask."""
    # 0 background, 1 left ventricle, 2 right ventricle
    return np.load(DATA_DIR / "mask_3c.npz")["mask_3c"]
