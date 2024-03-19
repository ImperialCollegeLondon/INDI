"""
Script to fish the DTCMR DICOMs from a bunch of folders
and organise them in three folders:
- DTCMR magnitude
- DTCMR phase
- other non diffusion DICOMs
"""

import glob
import os
import sys

import pydicom
from tqdm.auto import tqdm

# root folder that will contain a subfolder per subject
root_folder = "/Users/pf/Desktop/WHOLE_HEART_STUDY/original_data/"


def move_dicom(folder, dicom_file):
    """
    Move a DICOM file to another folder
    """
    # how many files already in the folder
    n_files = len([entry for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))])
    dicom_new_filename = "MR_" + str(n_files + 1).zfill(5) + ".dcm"

    new_file = os.path.join(folder, dicom_new_filename)
    os.rename(dicom_file, new_file)


# get all the subfolders
subfolders = glob.glob(root_folder + "**")
subfolders.sort()

# loop over the subfolders
for subfolder in tqdm(subfolders, desc="Subfolders", position=0, leave=False):
    # create subfolders for the DTCMR DICOMs
    # make sure the folders do not exist already
    mag_folder = os.path.join(subfolder, "diffusion_images")
    phase_folder = os.path.join(subfolder, "diffusion_phase_images")
    other_folder = os.path.join(subfolder, "other_images")

    if not os.path.exists(mag_folder):
        os.makedirs(mag_folder)
    else:
        print(f"Folder already exists: {mag_folder}")
        sys.exit()

    if not os.path.exists(phase_folder):
        os.makedirs(phase_folder)
    else:
        print(f"Folder already exists: {phase_folder}")
        sys.exit()

    if not os.path.exists(other_folder):
        os.makedirs(other_folder)
    else:
        print(f"Folder already exists: {other_folder}")
        sys.exit()

    # get all the DICOMs in the subfolder
    dicom_files = glob.glob(subfolder + "/**/*.dcm", recursive=True)
    dicom_files.sort()

    # loop over the DICOMs and fish the DTCMR ones
    for dicom_file in tqdm(dicom_files, desc="DICOMs", position=1, leave=False):
        ds = pydicom.dcmread(dicom_file)

        if "AcquisitionContrast" not in ds or "ComplexImageComponent" not in ds:
            # if DICOM is not a diffusion image
            move_dicom(other_folder, dicom_file)

        else:
            if ds.AcquisitionContrast == "DIFFUSION" and ds.ComplexImageComponent == "MAGNITUDE":
                # if DICOM is a diffusion and magnitude image
                move_dicom(mag_folder, dicom_file)

            elif ds.AcquisitionContrast == "DIFFUSION" and ds.ComplexImageComponent == "PHASE":
                # if DICOM is a diffusion and phase image
                move_dicom(phase_folder, dicom_file)

            else:
                # if DICOM is not a diffusion image
                move_dicom(other_folder, dicom_file)
