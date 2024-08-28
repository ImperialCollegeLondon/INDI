import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import RectangleSelector

from extensions.extension_base import ExtensionBase


class ThreeDSelector:
    def __init__(self, nx, ny, nz) -> None:
        self.slice = [0, nx]
        self.row = [0, ny]
        self.col = [0, nz]

    def set_selectors(self, selector_front, selector_side, selector_top):
        self.selector_front = selector_front
        self.selector_side = selector_side
        self.selector_top = selector_top

    def update(self):
        self.selector_front.extents = (self.col[0], self.col[1], self.row[0], self.row[1])
        self.selector_side.extents = (self.col[0], self.col[1], self.slice[0], self.slice[1])
        self.selector_top.extents = (self.row[0], self.row[1], self.slice[0], self.slice[1])

    def select_front(self, eclick, erelease):
        self.col = [eclick.xdata, erelease.xdata]
        self.row = [eclick.ydata, erelease.ydata]
        self.update()

    def select_side(self, eclick, erelease):
        self.slice = [eclick.xdata, erelease.xdata]
        self.col = [eclick.ydata, erelease.ydata]
        self.update()

    def select_top(self, eclick, erelease):
        self.row = [eclick.xdata, erelease.xdata]
        self.slice = [eclick.ydata, erelease.ydata]
        self.update()


def manual_crop(image):
    """Use the mouse to select the ROI for cropping"""
    nx, ny, nz = image.shape

    roi = ThreeDSelector(nx, ny, nz)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    rect_props = dict(fill=False, linestyle="-", edgecolor="yellow")

    axs[0].imshow(image[nx // 2, :, :], cmap="gray")
    axs[0].set_title("Front view")
    axs[0].axis("off")
    front_rect = RectangleSelector(
        axs[0],
        roi.select_front,
        interactive=True,
        drag_from_anywhere=True,
        props=rect_props,
    )
    axs[0].axhline(ny // 2, color="r", linestyle="--")
    axs[0].axvline(nz // 2, color="r", linestyle="--")

    axs[1].imshow(image[:, ny // 2, :], cmap="gray")
    axs[1].set_title("Side view")
    axs[1].axis("off")
    side_rect = RectangleSelector(
        axs[1],
        roi.select_side,
        interactive=True,
        drag_from_anywhere=True,
        props=rect_props,
    )
    axs[1].axhline(nx // 2, color="r", linestyle="--")
    axs[1].axvline(nz // 2, color="r", linestyle="--")

    axs[2].imshow(image[:, :, nz // 2], cmap="gray")
    axs[2].set_title("Top view")
    axs[2].axis("off")
    top_rect = RectangleSelector(
        axs[2],
        roi.select_top,
        interactive=True,
        drag_from_anywhere=True,
        props=rect_props,
    )
    axs[2].axhline(nx // 2, color="r", linestyle="--")
    axs[2].axvline(ny // 2, color="r", linestyle="--")

    roi.set_selectors(front_rect, side_rect, top_rect)
    roi.update()
    plt.show()

    return roi.slice, roi.row, roi.col


class Crop(ExtensionBase):
    def run(self) -> None:
        """Crop data to the desired ROI

        Parameters
        ----------
        data : dict
            dictionary to hold data
        slices : list
            list of slices index
        settings : dict
        info : dict
        logger : logging
            logger for console

        Returns
        -------
        data : dict
            pd.Dataframe to hold data
        slices : list
            dictionary to hold slices
        """

        data = self.context["data"]
        image = np.asarray([np.asarray(data["image"][i], dtype=int) for i in self.context["slices"]])

        if os.path.exists(os.path.join(self.settings["session"], "crop.yaml")):
            with open(os.path.join(self.settings["session"], "crop.yaml"), "r") as handle:
                crop = yaml.safe_load(handle)
                self.slice = crop["slice"]
                self.row = crop["row"]
                self.col = crop["col"]
            self.logger.debug("ROI loaded from crop.yaml")

        else:
            self.logger.info("Select the ROI for cropping")
            slice, row, col = manual_crop(image)
            self.slice = [int(slice[0]), int(slice[1])]
            self.row = [int(row[0]), int(row[1])]
            self.col = [int(col[0]), int(col[1])]

        self.logger.info(f"ROI: {self.slice}, {self.row}, {self.col}")

        # crop the data
        data["image"] = data["image"].apply(lambda x: x[self.row[0] : self.row[1], self.col[0] : self.col[1]])
        slices = self.context["slices"][self.slice[0] : self.slice[1]]
        self._save_crop()

        self.context["data"], self.context["slices"] = data, slices

    def _save_crop(self):
        """Saves the crop values"""
        crop = dict(slice=self.slice, row=self.row, col=self.col)
        with open(os.path.join(self.settings["session"], "crop.yaml"), "w") as handle:
            yaml.dump(crop, handle)
