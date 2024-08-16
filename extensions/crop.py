import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector


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


def crop_data(data: pd.DataFrame, slices: List[int], settings: Dict, info: Dict, logger: logging.Logger):
    """
    Crop data to the desired ROI

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
        dictionary to hold data
    slices : list
        dictionary to hold slices
    """

    image = np.asarray([np.asarray(data["image"][i], dtype=int) for i in slices])
    nx, ny, nz = image.shape

    roi = ThreeDSelector(nx, ny, nz)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(image[nx // 2, :, :], cmap="gray")
    axs[0].set_title("Front view")
    axs[0].axis("off")
    front_rect = RectangleSelector(
        axs[0],
        roi.select_front,
        interactive=True,
        drag_from_anywhere=True,
    )

    axs[1].imshow(image[:, ny // 2, :], cmap="gray")
    axs[1].set_title("Side view")
    axs[1].axis("off")
    side_rect = RectangleSelector(
        axs[1],
        roi.select_side,
        interactive=True,
        drag_from_anywhere=True,
    )

    axs[2].imshow(image[:, :, nz // 2], cmap="gray")
    axs[2].set_title("Top view")
    axs[2].axis("off")
    top_rect = RectangleSelector(
        axs[2],
        roi.select_top,
        interactive=True,
        drag_from_anywhere=True,
    )

    roi.set_selectors(front_rect, side_rect, top_rect)
    roi.update()
    plt.show()

    logger.info(f"ROI: {roi.slice}, {roi.row}, {roi.col}")

    # crop the data
    data["image"].apply(lambda x: x[int(roi.row[0]) : int(roi.row[1]), int(roi.col[0]) : int(roi.col[1])])
    slices = slices[int(roi.slice[0]) : int(roi.slice[1])]
    return data, slices
