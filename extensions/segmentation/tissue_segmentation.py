import bisect
import pathlib
import subprocess

import nrrd
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep

from extensions.extension_base import ExtensionBase
from extensions.segmentation.heart_segmentation import get_premliminary_ha_md_maps


def build_curves(points):
    curves = []
    zs = []
    for p in points:
        control_points = np.stack([[p["s"][i], p["p"][i], p["l"][i]] for i in range(len(p))], axis=-1)
        z = np.mean(control_points[2, :])  # Assume each curve varies little in the z direction
        u_fine = np.linspace(0, 1, 100)
        tck, _ = splprep(control_points[:2, :])
        x_fine, y_fine = splev(u_fine, tck)
        curves.append([x_fine, y_fine])
        zs.append(z)

    zs = np.asarray(zs)
    curves = np.stack(curves, axis=0)
    arg = np.argsort(zs)
    zs = zs[arg]
    curves = curves[arg, :, :]

    return curves, zs


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

        assert mask_3c.ndim == 3 or np.max(mask_3c) > 1, "Select only one segment"

        self.context["mask_3c"] = mask_3c

        points = [pd.read_csv(p) for p in session.glob("curves*.csv") if "schema" not in p.name]

        epi_points = list(filter(lambda p: "epi" in p["label"][0], points))
        endo_points = list(filter(lambda p: "endo" in p["label"][0], points))

        epi_curves, epi_zs = build_curves(epi_points)
        endo_curves, endo_zs = build_curves(endo_points)

        z_desired = np.arange(len(self.context["slices"])) + 1
        epi = interpolate_curves(epi_zs, epi_curves, z_desired)
        endo = interpolate_curves(endo_zs, endo_curves, z_desired)

        segmentation = {}

        import matplotlib.pyplot as plt

        for i, slice_idx in enumerate(self.context["slices"]):
            segmentation[slice_idx] = {
                "epicardium": epi[i],
                "endocardium": endo[i],
                "anterior_ip": np.array([]),
                "inferior_ip": np.array([]),
            }
            plt.imshow(self.context["average_images"][slice_idx], cmap="gray")
            plt.plot(endo[i][0], endo[i][1])
            plt.plot(epi[i][0], epi[i][1])
            plt.show()

        self.context["segmentation"] = segmentation
