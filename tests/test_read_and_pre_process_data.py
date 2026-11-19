from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from indi.extensions.read_data.read_and_pre_process_data import read_and_process_pandas


@pytest.fixture
def basic_settings(tmp_path):
    """Set up a temporary diffusion_images folder with minimal settings."""
    dicom_folder = tmp_path / "diffusion_images"
    dicom_folder.mkdir()
    return {"dicom_folder": str(dicom_folder), "complex_data": False}


@pytest.fixture
def magnitude_data_setup(tmp_path):
    """Create the common magnitude folder and settings for complex-data tests."""
    dicom_folder = tmp_path / "diffusion_images"
    mag_folder = dicom_folder / "mag"
    mag_folder.mkdir(parents=True)
    return {
        "settings": {"dicom_folder": str(dicom_folder), "complex_data": False},
        "mag_folder": mag_folder,
        "dicom_folder": dicom_folder,
    }


def create_test_data_gz(folder, info=None, manufacturer="siemens", n=3, columns=None):
    """Create a gzipped dataframe resembling diffusion metadata.

    Args:
        folder (Path | str): Destination directory for the pickle file.
        info (dict | None): Optional metadata stored in ``attrs``.
        manufacturer (str): Manufacturer name to inject in the dataframe.
        n (int): Number of rows to generate.
        columns (dict | None): Optional dataframe columns to write instead of defaults.

    Returns:
        pd.DataFrame: Generated dataframe with attached ``info`` metadata.
    """
    if columns is None:
        columns = {"Manufacturer": [manufacturer] * n, "some_col": np.arange(n)}
    df = pd.DataFrame(columns)
    if info is None:
        info = {"Rows": 2, "Columns": 2}
    df.attrs["info"] = info
    save_path = Path(folder) / "data.gz"
    df.to_pickle(save_path)
    return df


def create_test_images_h5(folder, n=3, shape=(2, 2)):
    """Create an HDF5 file with dummy image data.

    Args:
        folder (Path | str): Destination directory for the H5 file.
        n (int): Number of images to create.
        shape (tuple[int, int]): Spatial dimensions of each image.

    Returns:
        np.ndarray: Array of generated images.
    """
    images = np.arange(n * shape[0] * shape[1]).reshape(n, *shape)
    save_path = Path(folder) / "images.h5"
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("pixel_values", data=images)
    return images


def test_read_and_process_pandas_basic(logger, basic_settings):
    """Validate successful load of magnitude-only datasets."""
    # Setup
    dicom_folder = basic_settings["dicom_folder"]

    # Create mag data
    create_test_data_gz(dicom_folder)
    create_test_images_h5(dicom_folder)

    # Run
    data, data_phase, info = read_and_process_pandas(logger, basic_settings)

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert data_phase.empty
    assert isinstance(info, dict)
    assert "image" in data.columns
    assert len(data) == 3
    assert all(isinstance(image, np.ndarray) for image in data["image"])
    assert info["Rows"] == 2
    assert info["Columns"] == 2
    assert info["manufacturer"] == "siemens"


def test_read_and_process_pandas_missing_data_gz(logger, basic_settings):
    """Ensure missing data.gz raises FileNotFoundError."""
    # No data.gz file
    with pytest.raises(FileNotFoundError):
        read_and_process_pandas(logger, basic_settings)


def test_read_and_process_pandas_missing_images_h5(logger, basic_settings):
    """Ensure missing images.h5 triggers an OSError."""
    dicom_folder = basic_settings["dicom_folder"]
    create_test_data_gz(dicom_folder)
    # No images.h5
    with pytest.raises(OSError):
        read_and_process_pandas(logger, basic_settings)


def test_read_and_process_pandas_complex_data(logger, magnitude_data_setup):
    """Handle complex data paths and infer manufacturer."""
    settings = magnitude_data_setup["settings"]
    mag_folder = magnitude_data_setup["mag_folder"]
    phase_folder = magnitude_data_setup["dicom_folder"] / "phase"
    phase_folder.mkdir()

    # Create mag data
    create_test_data_gz(mag_folder)
    create_test_images_h5(mag_folder)
    # Create phase data
    create_test_data_gz(phase_folder)
    create_test_images_h5(phase_folder)

    # Run
    data, phase_data, info = read_and_process_pandas(logger, settings)

    # Assert
    assert settings["complex_data"] is True
    assert isinstance(data, pd.DataFrame)
    assert isinstance(phase_data, pd.DataFrame)
    assert "image" in data.columns
    assert "image" in phase_data.columns
    assert info["manufacturer"] == "siemens"


def test_read_and_process_pandas_missing_phase_folder(logger, magnitude_data_setup):
    """Raise when complex data is requested without a phase folder."""
    settings = magnitude_data_setup["settings"]
    mag_folder = magnitude_data_setup["mag_folder"]
    create_test_data_gz(mag_folder)
    create_test_images_h5(mag_folder)
    # No phase folder
    with pytest.raises(FileNotFoundError):
        read_and_process_pandas(logger, settings)


def test_read_and_process_pandas_missing_manufacturer_column(logger, basic_settings):
    """Error when manufacturer column is absent in saved dataframe."""
    dicom_folder = basic_settings["dicom_folder"]
    create_test_data_gz(dicom_folder, info={}, columns={"some_col": [1, 2, 3]})
    create_test_images_h5(dicom_folder)
    with pytest.raises(ValueError, match="The 'Manufacturer' column is missing"):
        read_and_process_pandas(logger, basic_settings)


def test_read_and_process_pandas_empty_dataframe(logger, basic_settings):
    """Error when dataframe contains no rows."""
    dicom_folder = basic_settings["dicom_folder"]
    create_test_data_gz(dicom_folder, info={}, columns={"Manufacturer": []})
    create_test_images_h5(dicom_folder, n=0)
    with pytest.raises(ValueError, match="The data DataFrame is empty"):
        read_and_process_pandas(logger, basic_settings)


def test_read_and_process_pandas_unsupported_manufacturer(logger, basic_settings):
    """Reject unsupported manufacturer strings."""
    dicom_folder = basic_settings["dicom_folder"]
    create_test_data_gz(dicom_folder, info={}, columns={"Manufacturer": ["unknown"]})
    create_test_images_h5(dicom_folder, n=1)
    with pytest.raises(ValueError, match="Manufacturer not supported"):
        read_and_process_pandas(logger, basic_settings)
