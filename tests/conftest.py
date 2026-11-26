import logging
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def logger() -> Logger:
    """Provide a logger instance for tests."""
    return logging.getLogger("test_logger")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Provide the directory containing shared test data."""
    return DATA_DIR


@pytest.fixture(scope="session")
def tensor_true(data_dir: Path) -> np.ndarray:
    """Load the shared ground-truth tensor."""
    with np.load(data_dir / "DT.npz") as data:
        return data["DT"]


@pytest.fixture(scope="module")
def tensor_fit_table(data_dir: Path) -> pd.DataFrame:
    """Load the noiseless DTI_numerical_phantom_3D table used by tensor-fitting tests."""
    # Here I am loading data from DTI_numerical_phantom_3D repo
    # Load table with pixel values, b_values and directions
    return pd.read_pickle(data_dir / "cdti_table.zip")


@pytest.fixture
def mask_3c(data_dir: Path) -> np.ndarray:
    """Load the shared three-class phantom mask."""
    # 0 background, 1 left ventricle, 2 right ventricle
    with np.load(data_dir / "mask_3c.npz") as data:
        return data["mask_3c"]


@pytest.fixture
def results_settings(tmp_path: Path) -> dict:
    """Create shared results/results_b and debug folders with their settings."""
    results_dir = tmp_path / "results"
    (results_dir / "results_b").mkdir(parents=True)
    debug_folder = tmp_path / "debug"
    debug_folder.mkdir()
    return {
        "results": str(results_dir),
        "debug_folder": str(debug_folder),
        "debug": False,
    }
