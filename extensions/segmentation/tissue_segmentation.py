import bisect
import copy
import pathlib
import subprocess

import nrrd
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from skimage import morphology

from extensions.extension_base import ExtensionBase
from extensions.extensions import (
    close_small_holes,
    convert_array_to_dict_of_arrays,
    convert_dict_of_arrays_to_array,
    get_cylindrical_coordinates_short_axis,
)
from extensions.segmentation.heart_segmentation import get_preliminary_ha_md_maps


def build_curves(points, mask):
    # curves = []
    # zs = []
    # for p in points:
    #     control_points = np.stack([[p["s"][i], p["p"][i], p["l"][i]] for i in range(len(p))], axis=-1)
    #     control_points = control_points[:, np.argsort(control_points[0])][:, ::-1]
    #     z = np.mean(control_points[2, :])  # Assume each curve varies little in the z direction
    #     u_fine = np.linspace(0, 1, 100)
    #     tck, _ = splprep(control_points[:2, :], k=3 if len(control_points) > 3 else 2, s=0)
    #     x_fine, y_fine = splev(u_fine, tck)
    #
    #     x = control_points[0, :]
    #     y = control_points[1, :]
    #
    #     # calculate polynomial
    #     z = np.polyfit(x, y, 2)
    #     f = np.poly1d(z)
    #
    #     # calculate new x's and y's
    #     x_new = np.linspace(x[0], x[-1], 100)
    #     y_new = f(x_new)
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.plot(x, y, "o", x_new, y_new)
    #     plt.savefig("test.png")
    #     plt.close()
    #
    #     curves.append([x_fine, y_fine])
    #     zs.append(z)
    #
    # zs = np.asarray(zs)
    # curves = np.stack(curves, axis=0)
    # arg = np.argsort(zs)
    # zs = zs[arg]
    # curves = curves[arg, :, :]

    # TODO this code is not working correctly at the moment. It needs to be fixed
    x = []
    y = []
    z = []
    for point in points:
        x.extend(list(point.p.values))
        y.extend(list(point.s.values))
        z.extend(list(point.l.values))
    data = np.array([x, y, z]).T

    # Define mathematical function for curve fitting
    def func(xy, a, b, c, d, e, f):
        x, y = xy
        # return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y
        return a + b * x + c * y + f * x * y

    # Perform curve fitting
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(func, (x, y), z)

    # Print optimized parameters
    print(popt)

    # Create 3D plot of the data points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(np.transpose(mask, (1, 2, 0)), alpha=0.5)
    ax.scatter(x, y, z, color="blue")
    x_range = np.linspace(min(x), max(x), 50)
    y_range = np.linspace(min(y), max(y), 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func((X, Y), *popt)
    ax.plot_surface(X, Y, Z, color="red", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show(block=True)
    # plt.savefig("test.png")

    curves = 1
    zs = 1

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

        points = [pd.read_csv(p) for p in session.glob("curves*.csv") if "schema" not in p.name]

        epi_points = list(filter(lambda p: "epi" in p["label"][0], points))
        endo_points = list(filter(lambda p: "endo" in p["label"][0], points))

        epi_curves, epi_zs = build_curves(epi_points, mask_3c)
        endo_curves, endo_zs = build_curves(endo_points)

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
