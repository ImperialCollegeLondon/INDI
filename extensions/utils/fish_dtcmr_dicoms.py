"""
Script to fish the DTCMR DICOMs from a bunch of folders
and organise them in three folders:
- DTCMR magnitude
- DTCMR phase
- other non diffusion DICOMs
"""

import glob
import os
import shutil
import sys

import pydicom
from tqdm.auto import tqdm

# root folder that will contain a subfolder per subject
root_folder = "/Users/pf/Desktop/HLH/DICOMS_from_circle/UR75"


def copy_dicom(folder, dicom_file):
    """
    Copy a DICOM file to another folder
    """
    # how many files already in the folder
    n_files = len([entry for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))])
    dicom_new_filename = "MR_" + str(n_files + 1).zfill(5) + ".dcm"

    new_file = os.path.join(folder, dicom_new_filename)
    shutil.copy2(dicom_file, new_file)


# get all the subfolders
subfolders = glob.glob(root_folder + "**")
subfolders.sort()

# loop over the subfolders
for subfolder in tqdm(subfolders, desc="Subfolders", position=0, leave=False):
    # create subfolders for the DTCMR DICOMs
    # make sure the folders do not exist already
    mag_folder = os.path.join(subfolder, "diffusion_images")
    phase_folder = os.path.join(subfolder, "diffusion_phase_images")
    map_folder = os.path.join(subfolder, "diffusion_maps")

    if not os.path.exists(mag_folder):
        os.makedirs(os.path.join(mag_folder, "systole"))
        os.makedirs(os.path.join(mag_folder, "diastole"))
    else:
        print(f"Folder already exists: {mag_folder}")
        sys.exit()

    if not os.path.exists(phase_folder):
        os.makedirs(os.path.join(phase_folder, "systole"))
        os.makedirs(os.path.join(phase_folder, "diastole"))
    else:
        print(f"Folder already exists: {phase_folder}")
        sys.exit()

    if not os.path.exists(map_folder):
        os.makedirs(map_folder)
    else:
        print(f"Folder already exists: {map_folder}")
        sys.exit()

    # get all the DICOMs in the subfolder
    dicom_files = glob.glob(subfolder + "/**/*.dcm", recursive=True)
    dicom_files.sort()

    # loop over the DICOMs and fish the DTCMR ones
    for dicom_file in tqdm(dicom_files, desc="DICOMs", position=1, leave=False):
        ds = pydicom.dcmread(dicom_file)

        if "PerFrameFunctionalGroupsSequence" in ds:
            # modern DICOMs
            if "AcquisitionContrast" not in ds or "ComplexImageComponent" not in ds:
                # if DICOM is not a diffusion image
                pass
            else:
                if ds.AcquisitionContrast == "DIFFUSION" and ds.ComplexImageComponent == "MAGNITUDE":
                    # if DICOM is a diffusion and magnitude image
                    copy_dicom(mag_folder, dicom_file)

                elif ds.AcquisitionContrast == "DIFFUSION" and ds.ComplexImageComponent == "PHASE":
                    # if DICOM is a diffusion and phase image
                    copy_dicom(phase_folder, dicom_file)

                else:
                    # if DICOM is not a diffusion image
                    pass

        else:
            if "ImageType" not in ds:
                # if DICOM is not a diffusion image
                pass

            else:
                # old DICOMs
                image_type = ds.ImageType = [x.upper() for x in ds.ImageType]

                if image_type[2] == "DIFFUSION":
                    if image_type[0] == "ORIGINAL" and image_type[1] == "PRIMARY":
                        # determine cardiac phase
                        trig_time = ds.TriggerTime
                        delta_diastole = abs(trig_time - 0)
                        delta_diastole2 = abs(trig_time - 1000)
                        delta_diastole = delta_diastole if delta_diastole < delta_diastole2 else delta_diastole2
                        delta_systole = abs(trig_time - 300)
                        if delta_diastole < delta_systole:
                            cardiac_phase = "diastole"
                        else:
                            cardiac_phase = "systole"

                        # if DICOM is a diffusion and magnitude image
                        copy_dicom(os.path.join(mag_folder, cardiac_phase), dicom_file)
                    elif image_type[0] == "DERIVED" and image_type[1] == "PRIMARY":
                        # if DICOM is a diffusion derived map
                        copy_dicom(map_folder, dicom_file)
                else:
                    # if DICOM is not a diffusion image
                    pass
