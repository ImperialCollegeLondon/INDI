import bisect
import pathlib
import subprocess

import nrrd
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import splev, splprep

from extensions.extension_base import ExtensionBase
from extensions.extensions import convert_array_to_dict_of_arrays, convert_dict_of_arrays_to_array
from extensions.segmentation.heart_segmentation import get_preliminary_ha_md_maps


def build_curves(points, mask):
    n_poly_points = 100

    curves = []
    curves_fine = []
    zs = []
    for p in points:
        control_points = np.stack([[p["s"][i], p["p"][i], p["l"][i]] for i in range(len(p))], axis=-1)
        control_points = control_points[:, np.argsort(control_points[0])][:, ::-1]
        z = np.mean(control_points[2, :])  # Assume each curve varies little in the z direction
        u_fine = np.linspace(0, 1, n_poly_points)
        tck, _ = splprep(control_points[:2, :], k=2 if len(control_points[0]) > 3 else 2, s=0)
        x_fine, y_fine = splev(u_fine, tck)
        curves_fine.append([x_fine, y_fine])
        zs.append(z)
        curves.append(control_points[:2, :])

    zs = np.asarray(zs)
    curves_fine = np.stack(curves_fine, axis=0)
    arg = np.argsort(zs)
    zs = zs[arg]
    curves_fine = curves_fine[arg, :, :]
    curves = {idx: curves[k] for idx, k in enumerate(arg)}

    curves_fine_fine = np.zeros(curves_fine.shape)
    for idx in range(n_poly_points):
        x = curves_fine[:, 0, idx]
        y = curves_fine[:, 1, idx]
        z = zs
        tck, u = interpolate.splprep([x, y, z], k=1, s=10)
        u_fine = np.linspace(0, 1, len(z))
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        curves_fine_fine[:, 0, idx] = x_fine
        curves_fine_fine[:, 1, idx] = y_fine

    # TODO debug plot with 3D mask and points

    return curves_fine_fine, zs


def interpolate_curves(zs, curves, zs_desired):
    curves_interpolated = []
    # Make sure there is no division by zero
    # zs_desired += 1
    zs += 1
    for z in zs_desired:
        i = min(bisect.bisect(zs, z), len(zs) - 1)
        curve1 = curves[i]
        curve2 = curves[i - 1]
        alpha = (z - zs[i - 1]) / (zs[i] - zs[i - 1])
        curves_interpolated.append(alpha * curve1 + (1 - alpha) * curve2)

    return curves_interpolated


python_code = """

def exportLabelmap():

    filepath = outputPath + "/label.seg.nrrd"
    segmentationNode = getNode('vtkMRMLSegmentationNode1')
    storageNode = segmentationNode.CreateDefaultStorageNode()
    storageNode.SetFileName(filepath)
    storageNode.WriteData(segmentationNode)

    points = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
    for i, p in enumerate(points):
        fname = outputPath + "/curves" + str(i+1) + ".csv"
        slicer.modules.markups.logic().ExportControlPointsToCSV(p, fname)

    slicer.util.delayDisplay("Segmentation saved to " + filepath)

shortcut = qt.QShortcut(slicer.util.mainWindow())
shortcut.setKey(qt.QKeySequence("Ctrl+Shift+s"))
shortcut.connect( "activated()", exportLabelmap)

"""


class ExternalTissueBlockSegmentation(ExtensionBase):
    def run(self) -> None:
        self.logger.info("Running Heart Segmentation")

        session = pathlib.Path(self.settings["session"])

        if not (session / "label.seg.nrrd").exists():
            prelim_ha, prelim_md = get_preliminary_ha_md_maps(
                self.context["slices"],
                self.context["average_images"],
                self.context["data"],
                self.context["info"],
                self.settings,
                self.logger,
            )

            output_path_code = "outputPath = '" + self.settings["session"] + "'"
            script = output_path_code + python_code

            prelim_ha_array = convert_dict_of_arrays_to_array(prelim_ha)
            prelim_md_array = convert_dict_of_arrays_to_array(prelim_md)
            average_images = self.context["average_images"]
            average_images_array = convert_dict_of_arrays_to_array(average_images)

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

        mask_3c, _ = nrrd.read((session / "label.seg.nrrd").as_posix())
        # mask_3c, header = nrrd.read((session / "Segmentation.seg.nrrd").as_posix())

        assert mask_3c.ndim == 3 or np.max(mask_3c) > 1, "Select only one segment"

        mask_3c_dict = convert_array_to_dict_of_arrays(mask_3c, self.context["slices"])
        self.context["mask_3c"] = mask_3c_dict

        # TODO remove endo curves, I don't think I need them.

        points = [pd.read_csv(p) for p in session.glob("curves*.csv") if "schema" not in p.name]

        epi_points = list(filter(lambda p: "epi" in p["label"][0], points))
        endo_points = list(filter(lambda p: "endo" in p["label"][0], points))

        epi_curves, epi_zs = build_curves(epi_points, mask_3c)
        endo_curves, endo_zs = build_curves(endo_points, mask_3c)

        z_desired = np.arange(len(self.context["slices"])) + 1
        epi = interpolate_curves(epi_zs, epi_curves, z_desired)
        endo = interpolate_curves(endo_zs, endo_curves, z_desired)

        segmentation = {}
        for i, slice_idx in enumerate(self.context["slices"]):
            segmentation[slice_idx] = {
                "epicardium": epi[i].T,
                "endocardium": endo[i].T,
                "anterior_ip": np.array([]),
                "inferior_ip": np.array([]),
            }

        self.context["segmentation"] = segmentation

        self.settings["RV-segmented"] = False
