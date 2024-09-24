import pathlib
import subprocess

import cv2 as cv
import nrrd
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep

from extensions.extension_base import ExtensionBase
from extensions.extensions import close_small_holes, get_cylindrical_coordinates_short_axis
from extensions.get_tensor_orientation_maps import get_ha_e2a_maps
from extensions.segmentation.manual_segmentation import (
    get_epi_contour,
    get_mask_from_poly,
    get_sa_contours,
    manual_lv_segmentation,
    plot_manual_lv_segmentation,
)
from extensions.segmentation.polygon_selector import spline_interpolate_contour
from extensions.tensor_fittings.tensor_fittings import dipy_tensor_fit

# Ideas
# Use a dynamically set tool for segmentation.
# The ida would be that this class is just a wrapper for a segmentation tool


def get_premliminary_ha_md_maps(slices, average_images, data, info, settings, logger):
    # =========================================================
    # Preliminary HA map
    # =========================================================

    # n_slices = len(slices)
    session = pathlib.Path(settings["session"])
    # check if LV manual segmentation has been previously saved
    # if not calculate a prelim HA map
    prelim_ha = np.zeros((info["n_slices"], info["img_size"][0], info["img_size"][1]))
    prelim_md = np.zeros((info["n_slices"], info["img_size"][0], info["img_size"][1]))
    # mask is all ones here for now.
    thr_mask = np.ones((info["n_slices"], info["img_size"][0], info["img_size"][1]))
    # loop over the slices
    for slice_idx in slices:
        if not (session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz").exists():
            # get cylindrical coordinates
            local_cylindrical_coordinates = get_cylindrical_coordinates_short_axis(
                thr_mask[[slice_idx], ...],
            )

            # get basic tensor
            tensor, _, _, _, info = dipy_tensor_fit(
                [slice_idx],
                data,
                info,
                settings,
                thr_mask,
                average_images,
                logger,
                "LS",
                quick_mode=True,
            )
            # get basic HA and MD maps
            prelim_eigenvalues, prelim_eigenvectors = np.linalg.eigh(tensor[[slice_idx], ...])
            prelim_ha[slice_idx], _, _ = get_ha_e2a_maps(
                thr_mask[[slice_idx], ...],
                local_cylindrical_coordinates,
                prelim_eigenvectors,
            )
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
        prelim_ha, prelim_md = get_premliminary_ha_md_maps(
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

        for slice_idx in self.context["slices"]:
            # check if LV manual segmentation has been previously saved
            if (session / f"manual_lv_segmentation_slice_{str(slice_idx).zfill(3)}.npz").exists():
                # load segmentations
                self.logger.info(
                    "Manual LV segmentation previously saved for slice: " + str(slice_idx) + ", loading mask..."
                )
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
                self.logger.info("Manual LV segmentation for slice: " + str(slice_idx))
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
            prelim_ha, prelim_md = get_premliminary_ha_md_maps(
                self.context["slices"],
                self.context["average_images"],
                self.context["data"],
                self.context["info"],
                self.settings,
                self.logger,
            )

            output_path_code = "outputPath = '" + self.settings["session"] + "'"

            script = output_path_code + python_code

            self.logger.info("Opening Slicer for manual segmentation")
            self.logger.info("Segment the LV and press Ctrl+Shift+s (Cmd+Shift+s) and close Slicer once done")

            nrrd.write((session / "average_images.nrrd").as_posix(), self.context["average_images"])
            nrrd.write((session / "MD_map.nrrd").as_posix(), prelim_md)
            nrrd.write((session / "HA_map.nrrd").as_posix(), prelim_ha)

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

        mask_3c, _ = nrrd.read((session / "label.seg.nrrd").as_posix())
        # mask_3c, header = nrrd.read((session / "Segmentation.seg.nrrd").as_posix())

        assert mask_3c.shape[0] == 2, "Segment both the LV and RV, both including the septum"

        self.context["mask_3c"] = mask_3c[0, ...]
        self.context["mask_rv"] = mask_3c[1, ...]

        points = [pd.read_csv(p) for p in session.glob("insertion_point_*.csv") if "schema" not in p.name]

        assert len(points) == 2, "No intersection points selected"

        points_interp = []

        u_fine = np.linspace(0, 1, len(self.context["slices"]))
        for p in points:
            arr = np.stack([p["s"].values, p["p"].values, p["l"].values], axis=0)

            tck, _ = splprep(arr)
            x_fine, y_fine, _ = splev(u_fine, tck)
            points_interp.append([x_fine, y_fine])

        segmentation = {}

        for i, slice_idx in enumerate(self.context["slices"]):
            segmentation[slice_idx] = {}

            epi_contour, endo_contour = get_sa_contours(self.context["mask_3c"][slice_idx])

            segmentation[slice_idx]["epicardium"] = epi_contour
            segmentation[slice_idx]["endocardium"] = endo_contour
            segmentation[slice_idx]["anterior_ip"] = np.asarray([points_interp[0][0][i], points_interp[0][1][i]])
            segmentation[slice_idx]["inferior_ip"] = np.asarray([points_interp[1][0][i], points_interp[1][1][i]])

        self.context["segmentation"] = segmentation
