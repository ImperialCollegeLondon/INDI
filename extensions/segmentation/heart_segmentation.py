import copy
import pathlib
import subprocess

import cv2 as cv
import nrrd
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from skimage import morphology
from tqdm import tqdm

from extensions.extension_base import ExtensionBase
from extensions.extensions import (
    close_small_holes,
    convert_array_to_dict_of_arrays,
    convert_dict_of_arrays_to_array,
    get_cylindrical_coordinates_short_axis,
)
from extensions.get_tensor_orientation_maps import get_ha_e2a_maps
from extensions.segmentation.manual_segmentation import (
    get_epi_contour,
    get_mask_from_poly,
    get_sa_contours,
    manual_lv_segmentation,
    plot_manual_lv_segmentation,
)
from extensions.segmentation.polygon_selector import spline_interpolate_contour
from extensions.tensor_fittings.tensor_fittings import quick_tensor_fit

# Ideas
# Use a dynamically set tool for segmentation.
# The ida would be that this class is just a wrapper for a segmentation tool


def get_preliminary_ha_md_maps(slices, average_images, data, info, settings, logger):
    # =========================================================
    # Preliminary HA map
    # =========================================================

    # n_slices = len(slices)
    session = pathlib.Path(settings["session"])

    # check if LV manual segmentation has been previously saved
    # if not calculate a prelim HA map
    prelim_ha = {}
    prelim_md = {}

    # loop over the slices
    for slice_idx in slices:
        if not (session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz").exists():
            # mask is all ones here for now.
            thr_mask = {slice_idx: np.ones((info["img_size"][0], info["img_size"][1]))}

            # get cylindrical coordinates
            local_cylindrical_coordinates = get_cylindrical_coordinates_short_axis(
                thr_mask[slice_idx],
                slice_idx,
            )
            # get basic tensor
            tensor = quick_tensor_fit(
                slice_idx,
                data,
                info,
            )
            # get basic HA and MD maps
            prelim_eigenvalues, prelim_eigenvectors = np.linalg.eigh(tensor)
            prelim_eigenvectors_dict = {slice_idx: prelim_eigenvectors}

            prelim_ha_, _, _ = get_ha_e2a_maps(
                thr_mask,
                local_cylindrical_coordinates,
                prelim_eigenvectors_dict,
            )
            prelim_ha[slice_idx] = prelim_ha_[0]
            prelim_md[slice_idx] = np.mean(prelim_eigenvalues, axis=-1)

            # threshold preliminary MD and HA maps
            prelim_ha[slice_idx] = prelim_ha[slice_idx] * thr_mask[slice_idx]
            prelim_md[slice_idx] = 1e3 * prelim_md[slice_idx] * thr_mask[slice_idx]

    return prelim_ha, prelim_md


class HeartSegmentation(ExtensionBase):
    """Heart Segmentation Extension
    If a segmentation file is available, this extension will load it and use it otherwise it will open a window with
    manual segmentation tool.
    """

    def run(self) -> None:
        self.logger.info("Running Heart Segmentation")
        # Get preliminary HA and MD maps
        prelim_ha, prelim_md = get_preliminary_ha_md_maps(
            self.context["slices"],
            self.context["average_images"],
            self.context["data"],
            self.context["info"],
            self.settings,
            self.logger,
        )
        session = pathlib.Path(self.settings["session"])

        n_slices = len(self.context["slices"])

        mask_3c = np.zeros(
            (
                self.context["info"]["n_slices"],
                self.context["info"]["img_size"][0],
                self.context["info"]["img_size"][1],
            ),
            dtype="uint8",
        )
        thr_mask = np.ones(
            (
                self.context["info"]["n_slices"],
                self.context["info"]["img_size"][0],
                self.context["info"]["img_size"][1],
            ),
            dtype="uint8",
        )

        segmentation = {}

        for slice_idx in tqdm(self.context["slices"], desc="Segmentation slices:"):
            # check if LV manual segmentation has been previously saved
            if (session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz").exists():
                # # load segmentations
                # self.logger.info(
                #     "Manual LV segmentation previously saved for slice: " + str(slice_idx) + ", loading mask..."
                # )
                npzfile = np.load(
                    session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz", allow_pickle=True
                )

                mask_3c[slice_idx] = npzfile["mask_3c"]
                segmentation[slice_idx] = npzfile["segmentation"]
                segmentation[slice_idx] = segmentation[slice_idx].item()

                # if there is no epicardial border defined, mark this slice to be removed in the dataframe
                if segmentation[slice_idx]["epicardium"].size == 0:
                    self.context["data"].loc[
                        self.context["data"]["slice_integer"] == slice_idx, "to_be_removed"
                    ] = True

            else:
                # manual LV segmentation
                # self.logger.info("Manual LV segmentation for slice: " + str(slice_idx))
                segmentation[slice_idx], thr_mask[slice_idx] = manual_lv_segmentation(
                    mask_3c[slice_idx],
                    self.context["average_images"][slice_idx],
                    prelim_ha[slice_idx],
                    prelim_md[slice_idx],
                    100,
                    self.settings,
                    self.context["colormaps"],
                    slice_idx,
                    self.context["slices"],
                )

                # define the final mask_3c
                if segmentation[slice_idx]["epicardium"].size != 0:
                    mask_epi = get_mask_from_poly(
                        segmentation[slice_idx]["epicardium"].astype(np.int32),
                        mask_3c[slice_idx].shape,
                    )
                else:
                    mask_epi = np.zeros(mask_3c[slice_idx].shape, dtype="uint8")
                    # mark this slice to be removed in the dataframe
                    self.context["data"].loc[
                        self.context["data"]["slice_integer"] == slice_idx, "to_be_removed"
                    ] = True

                if segmentation[slice_idx]["epicardium"].size != 0:
                    # only do the following if there is an epicardial border defined, otherwise this slice will be removed
                    if segmentation[slice_idx]["endocardium"].size != 0:
                        mask_endo = get_mask_from_poly(
                            segmentation[slice_idx]["endocardium"].astype(np.int32),
                            mask_3c[slice_idx].shape,
                        )
                    else:
                        mask_endo = np.zeros(mask_3c[slice_idx].shape, dtype="uint8")

                    # we need to remove the mask pixels that have been thresholded out
                    mask_epi *= thr_mask[slice_idx]

                    if segmentation[slice_idx]["endocardium"].size != 0:
                        # erode endo mask in order to keep the endo line inside the myocardial ROI
                        kernel = np.ones((2, 2), np.uint8)
                        mask_endo = cv.erode(mask_endo, kernel, iterations=1)
                        mask_endo *= thr_mask[slice_idx]

                    mask_lv = mask_epi - mask_endo
                    if segmentation[slice_idx]["endocardium"].size != 0:
                        epi_contour, endo_contour = get_sa_contours(mask_lv)
                    else:
                        epi_contour = get_epi_contour(mask_lv)
                        endo_contour = np.array([])

                    epi_len = len(epi_contour)
                    endo_len = len(endo_contour)
                    epi_contour = spline_interpolate_contour(epi_contour, 20, join_ends=False)
                    epi_contour = spline_interpolate_contour(epi_contour, epi_len, join_ends=False)

                    if segmentation[slice_idx]["endocardium"].size != 0:
                        endo_contour = spline_interpolate_contour(endo_contour, 20, join_ends=False)
                        endo_contour = spline_interpolate_contour(endo_contour, endo_len, join_ends=False)

                    segmentation[slice_idx]["epicardium"] = epi_contour
                    if segmentation[slice_idx]["endocardium"].size != 0:
                        segmentation[slice_idx]["endocardium"] = endo_contour

                    all_channel_mask = mask_3c[slice_idx].copy()
                    all_channel_mask[all_channel_mask == 1] = 0
                    all_channel_mask = all_channel_mask + mask_lv
                    all_channel_mask[all_channel_mask == 3] = 1
                    mask_3c[slice_idx] = all_channel_mask

                    # sometimes there are holes between the myocardium and rest of the heart mask, fill them here
                    mask_3c[slice_idx] = close_small_holes(mask_3c[slice_idx])

                # save mask and segmentation
                np.savez_compressed(
                    session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz",
                    mask_3c=mask_3c[slice_idx],
                    segmentation=segmentation[slice_idx],
                )

        if self.settings["debug"]:
            plot_manual_lv_segmentation(
                n_slices,
                self.context["slices"],
                segmentation,
                self.context["average_images"],
                mask_3c,
                self.settings,
                "lv_manual_mask",
                self.settings["debug_folder"],
            )

        # Export segmentation to context
        self.context["segmentation"] = segmentation
        self.context["mask_3c"] = mask_3c

        self.logger.info("Heart Segmentation Completed")


python_code = """


def exportLabelmap():

    filepath = outputPath + "/label.seg.nrrd"
    segmentationNode = getNode('vtkMRMLSegmentationNode1')
    storageNode = segmentationNode.CreateDefaultStorageNode()
    storageNode.SetFileName(filepath)
    storageNode.WriteData(segmentationNode)

    points = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
    for i, p in enumerate(points):
        fname = outputPath + "/insertion_point_" + str(i+1) + ".csv"
        slicer.modules.markups.logic().ExportControlPointsToCSV(p, fname)

    slicer.util.delayDisplay("Segmentation saved to " + filepath)

shortcut = qt.QShortcut(slicer.util.mainWindow())
shortcut.setKey(qt.QKeySequence("Ctrl+Shift+s"))
shortcut.connect( "activated()", exportLabelmap)

"""


class ExternalSegmentation(ExtensionBase):
    def run(self) -> None:
        self.logger.info("Running Heart Segmentation")

        session = pathlib.Path(self.settings["session"])

        if not (session / "label.seg.nrrd").exists():
            # if no 3D segmentation from slicer found, then run slicer

            prelim_ha, prelim_md = get_preliminary_ha_md_maps(
                self.context["slices"],
                self.context["average_images"],
                self.context["data"],
                self.context["info"],
                self.settings,
                self.logger,
            )

            prelim_ha_array = convert_dict_of_arrays_to_array(prelim_ha)
            prelim_md_array = convert_dict_of_arrays_to_array(prelim_md)
            average_images = self.context["average_images"]
            average_images_array = convert_dict_of_arrays_to_array(average_images)

            output_path_code = "outputPath = '" + self.settings["session"] + "'"
            script = output_path_code + python_code

            self.logger.info("Opening Slicer for manual segmentation")
            self.logger.info("Segment the LV and press Ctrl+Shift+s (Cmd+Shift+s) and close Slicer once done")

            nrrd.write((session / "average_images.nrrd").as_posix(), average_images_array)
            nrrd.write((session / "MD_map.nrrd").as_posix(), prelim_md_array)
            nrrd.write((session / "HA_map.nrrd").as_posix(), prelim_ha_array)

            subprocess.run(
                [
                    "/Applications/Slicer.app/Contents/MacOS/Slicer",
                    (session / "average_images.nrrd").as_posix(),
                    (session / "MD_map.nrrd").as_posix(),
                    (session / "HA_map.nrrd").as_posix(),
                    "--python-code",
                    script,
                ]
            )

        # load slicer 3D segmentation
        seg_mask, segmentation_info = nrrd.read((session / "label.seg.nrrd").as_posix())

        # retrieve label value and layer value from the segmentation masks
        def find_label_value_and_layer(input_dict, value):
            for key, val in input_dict.items():
                if isinstance(val, str):
                    if val == value:
                        label_value_key = key.replace("Name", "LabelValue")
                        layer_value_key = key.replace("Name", "Layer")
                        return (int(input_dict[label_value_key]), int(input_dict[layer_value_key]))
            return "None"

        lv_label_layer = find_label_value_and_layer(segmentation_info, "LV")
        whole_heart_layer = find_label_value_and_layer(segmentation_info, "whole_heart")

        # mark slices with no segmentation to be removed
        mask = np.any(seg_mask[0, ...], axis=(1, 2))
        slices_to_be_removed = np.asarray(self.context["slices"])[~mask]
        slices_to_keep = np.asarray(self.context["slices"])[mask]
        for slice_idx in slices_to_be_removed:
            self.context["data"].loc[self.context["data"]["slice_integer"] == slice_idx, "to_be_removed"] = True

        if whole_heart_layer:
            # LV segmentation
            lv_mask = np.copy(seg_mask[lv_label_layer[1], ...])
            lv_mask[lv_mask != lv_label_layer[0]] = 0
            lv_mask[lv_mask == lv_label_layer[0]] = 1

            # Whole heart segmentation
            whole_heart_mask = np.copy(seg_mask[whole_heart_layer[1], ...])
            whole_heart_mask[whole_heart_mask != whole_heart_layer[0]] = 0
            whole_heart_mask[whole_heart_mask == whole_heart_layer[0]] = 1

            # RV segmentation defined as the whole_heart - lv
            rv_mask = whole_heart_mask - lv_mask
            rv_mask[rv_mask > 2] = 0
            rv_mask[rv_mask < 0] = 0

            # remove all but the biggest island of the RV
            def remove_small_objects(img):
                binary = copy.copy(img)
                binary[binary > 0] = 1
                labels = morphology.label(binary)
                labels_num = [len(labels[labels == each]) for each in np.unique(labels)]
                rank = np.argsort(np.argsort(labels_num))
                index = list(rank).index(len(rank) - 2)
                new_img = copy.copy(img)
                new_img[labels != index] = 0
                return new_img

            rv_mask = remove_small_objects(rv_mask)

            mask_3c = lv_mask + (2 * rv_mask)

            # convert the 3D numpy array of masks to dictionary
            mask_3c_dict = convert_array_to_dict_of_arrays(mask_3c, self.context["slices"])
            lv_mask = convert_array_to_dict_of_arrays(lv_mask, self.context["slices"])
            rv_mask = convert_array_to_dict_of_arrays(rv_mask, self.context["slices"])

            self.settings["RV-segmented"] = True
            self.logger.info("RV segmentation detected")

        else:
            mask_3c = seg_mask
            mask_3c_dict = convert_array_to_dict_of_arrays(mask_3c, self.context["slices"])
            lv_mask = convert_array_to_dict_of_arrays(seg_mask, self.context["slices"])

            self.settings["RV-segmented"] = False
            self.logger.info("No RV segmentation detected")

        self.context["mask_3c"] = mask_3c_dict

        # insertion points
        points = [pd.read_csv(p) for p in session.glob("insertion_point_*.csv") if "schema" not in p.name]
        assert len(points) == 2, "Two insertion points needed, instead got: " + str(len(points))
        points_interp = []
        u_fine = np.linspace(0, 1, len(slices_to_keep))

        # The first IP is the inferior and the second is the anterior
        if points[0]["label"].values[0] == "anterior":
            points = points[::-1]

        for p in points:
            arr = np.stack([p["s"].values, p["p"].values, p["l"].values], axis=0)
            if len(arr[0]) < 4:
                k = len(arr[0]) - 1
            else:
                k = 3
            tck, _ = splprep(arr, k=k)
            x_fine, y_fine, _ = splev(u_fine, tck)
            points_interp.append([x_fine, y_fine])

        # segmentation dictionary
        segmentation = {}
        for i, slice_idx in enumerate(slices_to_keep):
            segmentation[slice_idx] = {}

            # get the contour points for the LV
            epi_contour, endo_contour = get_sa_contours(lv_mask[slice_idx])
            # interpolate with splines, so we do not have a ladder effect when calculating the
            # cardiac coordinates
            epi_len = len(epi_contour)
            endo_len = len(endo_contour)
            epi_contour = spline_interpolate_contour(epi_contour, 20, join_ends=False)
            epi_contour = spline_interpolate_contour(epi_contour, epi_len, join_ends=False)
            if endo_len > 10:
                endo_contour = spline_interpolate_contour(endo_contour, 20, join_ends=False)
                endo_contour = spline_interpolate_contour(endo_contour, endo_len, join_ends=False)

            # do the same for the RV
            if self.settings["RV-segmented"]:
                epi_contour_rv, endo_contour_rv = get_sa_contours(lv_mask[slice_idx] + rv_mask[slice_idx])
                epi_len = len(epi_contour_rv)
                endo_len = len(endo_contour_rv)
                epi_contour_rv = spline_interpolate_contour(epi_contour_rv, 20, join_ends=False)
                epi_contour_rv = spline_interpolate_contour(epi_contour_rv, epi_len, join_ends=False)
                if endo_len > 10:
                    endo_contour_rv = spline_interpolate_contour(endo_contour_rv, 20, join_ends=False)
                    endo_contour_rv = spline_interpolate_contour(endo_contour_rv, endo_len, join_ends=False)
            else:
                epi_contour_rv = np.array([])
                endo_contour_rv = np.array([])

            segmentation[slice_idx]["epicardium"] = epi_contour
            segmentation[slice_idx]["endocardium"] = endo_contour
            segmentation[slice_idx]["epicardium_rv"] = epi_contour_rv
            segmentation[slice_idx]["endocardium_rv"] = endo_contour_rv
            segmentation[slice_idx]["anterior_ip"] = np.asarray([points_interp[0][0][i], points_interp[0][1][i]])
            segmentation[slice_idx]["inferior_ip"] = np.asarray([points_interp[1][0][i], points_interp[1][1][i]])

        # remove slices not segmented
        self.context["data"] = self.context["data"][self.context["data"]["slice_integer"].isin(self.context["slices"])]
        self.context["segmentation"] = segmentation
