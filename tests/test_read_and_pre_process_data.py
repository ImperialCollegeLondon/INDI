import os
import logging
import numpy as np
import pandas as pd
import h5py
import pytest

from indi.extensions.read_data.read_and_pre_process_data import read_and_process_pandas

@pytest.fixture
def logger():
    return logging.getLogger("test_logger")

@pytest.fixture
def basic_settings(tmp_path):
    dicom_folder = tmp_path / "diffusion_images"
    dicom_folder.mkdir()
    return {
        "dicom_folder": str(dicom_folder),
        "complex_data": False
    }

def create_test_data_gz(folder, info=None, manufacturer="siemens", n=3):
    df = pd.DataFrame({
        "Manufacturer": [manufacturer] * n,
        "some_col": np.arange(n)
    })
    if info is None:
        info = {"Rows": 2, "Columns": 2}
    df.attrs["info"] = info
    save_path = os.path.join(folder, "data.gz")
    df.to_pickle(save_path)
    return df

def create_test_images_h5(folder, n=3, shape=(2,2)):
    arr = np.arange(n * shape[0] * shape[1]).reshape(n, *shape)
    save_path = os.path.join(folder, "images.h5")
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("pixel_values", data=arr)
    return arr

def test_read_and_process_pandas_basic(logger, basic_settings):
    # Setup
    dicom_folder = basic_settings["dicom_folder"]
    df = create_test_data_gz(dicom_folder)
    arr = create_test_images_h5(dicom_folder, n=3, shape=(2,2))

    # Run
    data, data_phase, info = read_and_process_pandas(logger, basic_settings)

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert data_phase.empty
    assert isinstance(info, dict)
    assert "image" in data.columns
    assert len(data) == 3
    assert np.all([isinstance(img, np.ndarray) for img in data["image"]])
    assert info["Rows"] == 2
    assert info["Columns"] == 2
    assert info["manufacturer"] == "siemens"

def test_read_and_process_pandas_missing_data_gz(logger, basic_settings):
    # No data.gz file
    with pytest.raises(FileNotFoundError):
        read_and_process_pandas(logger, basic_settings)

def test_read_and_process_pandas_missing_images_h5(logger, basic_settings):
    dicom_folder = basic_settings["dicom_folder"]
    create_test_data_gz(dicom_folder)
    # No images.h5
    with pytest.raises(OSError):
        read_and_process_pandas(logger, basic_settings)

def test_read_and_process_pandas_complex_data(tmp_path, logger):
    # Setup mag and phase folders
    dicom_folder = tmp_path / "diffusion_images"
    mag_folder = dicom_folder / "mag"
    phase_folder = dicom_folder / "phase"
    mag_folder.mkdir(parents=True)
    phase_folder.mkdir()
    settings = {
        "dicom_folder": str(dicom_folder),
        "complex_data": False
    }
    # Create mag data
    create_test_data_gz(mag_folder)
    create_test_images_h5(mag_folder)
    # Create phase data
    create_test_data_gz(phase_folder)
    create_test_images_h5(phase_folder)

    # Run
    data, data_phase, info = read_and_process_pandas(logger, settings)

    # Assert
    assert settings["complex_data"] is True
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data_phase, pd.DataFrame)
    assert "image" in data.columns
    assert "image" in data_phase.columns
    assert info["manufacturer"] == "siemens"

def test_read_and_process_pandas_missing_phase_folder(tmp_path, logger):
    dicom_folder = tmp_path / "diffusion_images"
    mag_folder = dicom_folder / "mag"
    mag_folder.mkdir(parents=True)
    settings = {
        "dicom_folder": str(dicom_folder),
        "complex_data": False
    }
    create_test_data_gz(mag_folder)
    create_test_images_h5(mag_folder)
    # No phase folder
    with pytest.raises(FileNotFoundError):
        read_and_process_pandas(logger, settings)

def test_read_and_process_pandas_missing_manufacturer_column(logger, basic_settings):
    dicom_folder = basic_settings["dicom_folder"]
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    df.attrs["info"] = {}
    df.to_pickle(os.path.join(dicom_folder, "data.gz"))
    create_test_images_h5(dicom_folder)
    with pytest.raises(ValueError, match="The 'Manufacturer' column is missing"):
        read_and_process_pandas(logger, basic_settings)

def test_read_and_process_pandas_empty_dataframe(logger, basic_settings):
    dicom_folder = basic_settings["dicom_folder"]
    df = pd.DataFrame({"Manufacturer": []})
    df.attrs["info"] = {}
    df.to_pickle(os.path.join(dicom_folder, "data.gz"))
    create_test_images_h5(dicom_folder, n=0)
    with pytest.raises(ValueError, match="The data DataFrame is empty"):
        read_and_process_pandas(logger, basic_settings)

def test_read_and_process_pandas_unsupported_manufacturer(logger, basic_settings):
    dicom_folder = basic_settings["dicom_folder"]
    df = pd.DataFrame({"Manufacturer": ["unknown"]})
    df.attrs["info"] = {}
    df.to_pickle(os.path.join(dicom_folder, "data.gz"))
    create_test_images_h5(dicom_folder, n=1)
    with pytest.raises(ValueError, match="Manufacturer not supported"):
        read_and_process_pandas(logger, basic_settings)