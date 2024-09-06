import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import RectangleSelector

from extensions.extension_base import ExtensionBase


class ThreeDSelector:
    def __init__(self, nx, ny, nz, img) -> None:
        self.slice = [0, nx]
        self.row = [0, ny]
        self.col = [0, nz]
        self.img = img

    def set_selectors(self, selector_front, selector_side, selector_top):
        self.selector_front = selector_front
        self.selector_side = selector_side
        self.selector_top = selector_top

    def set_img_plots(self, img_plots):
        self.img_plots = img_plots

    def set_line_selectors(self, lines):
        self.lines = lines

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

    def callback_top_horizontal(self, pos):
        self.img_plots["front"].set_data(self.img[int(pos), :, :])
        self.lines["side"]["horizontal"].pos = pos
        if self.slice[0] < pos < self.slice[1]:
            self.img_plots["front"].set_alpha(1)
        else:
            self.img_plots["front"].set_alpha(0.5)

    def callback_top_vertical(self, pos):
        self.img_plots["side"].set_data(self.img[:, int(pos), :])
        self.lines["front"]["vertical"].pos = pos
        if self.row[0] < pos < self.row[1]:
            self.img_plots["side"].set_alpha(1)
        else:
            self.img_plots["side"].set_alpha(0.5)

    def callback_side_horizontal(self, pos):
        self.img_plots["front"].set_data(self.img[int(pos), :, :])
        self.lines["top"]["horizontal"].pos = pos
        if self.slice[0] < pos < self.slice[1]:
            self.img_plots["front"].set_alpha(1)
        else:
            self.img_plots["front"].set_alpha(0.5)

    def callback_side_vertical(self, pos):
        self.img_plots["top"].set_data(self.img[:, :, int(pos)])
        self.lines["front"]["horizontal"].pos = pos
        if self.col[0] < pos < self.col[1]:
            self.img_plots["top"].set_alpha(1)
        else:
            self.img_plots["top"].set_alpha(0.5)

    def callback_front_horizontal(self, pos):
        self.img_plots["side"].set_data(self.img[:, int(pos), :])
        self.lines["top"]["vertical"].pos = pos
        if self.row[0] < pos < self.row[1]:
            self.img_plots["side"].set_alpha(1)
        else:
            self.img_plots["side"].set_alpha(0.5)

    def callback_front_vertical(self, pos):
        self.img_plots["top"].set_data(self.img[:, :, int(pos)])
        self.lines["side"]["vertical"].pos = pos
        if self.col[0] < pos < self.col[1]:
            self.img_plots["top"].set_alpha(1)
        else:
            self.img_plots["top"].set_alpha(0.5)


class LineSelector:
    def __init__(self, ax, pos, orientation, callback, range=5, interactive=False, props=None):
        self.ax = ax
        self.orientation = orientation
        self.callback = callback
        self.props = props
        self.interactive = interactive
        self.state = set()
        self.range = range
        self._pos = pos

        if self.orientation == "vertical":
            self.line = self.ax.axvline(self._pos, **self.props)
        else:
            self.line = self.ax.axhline(self._pos, **self.props)

        if self.interactive:
            self.connect()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if "move" in self.state:
            self.x0 = event.xdata
            self.y0 = event.ydata
            if self.orientation == "vertical":
                self.line.set_xdata([self.x0, self.x0])
                self.callback(self.x0)
            else:
                self.line.set_ydata([self.y0, self.y0])
                self.callback(self.y0)
            self.ax.figure.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:
            self.x0 = event.xdata
            self.y0 = event.ydata
            if self.orientation == "vertical":
                if self._pos - self.range < self.x0 < self._pos + self.range:
                    self.state.add("move")
            else:
                if self._pos - self.range < self.y0 < self._pos + self.range:
                    self.state.add("move")

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if "move" in self.state:
            if self.orientation == "vertical":
                self.line.set_xdata([self.x0, self.x0])
                self._pos = self.x0
                self.callback(self.x0)
                self.state.clear()
            else:
                self.line.set_ydata([self.y0, self.y0])
                self._pos = self.y0
                self.callback(self.y0)
                self.state.clear()
            self.ax.figure.canvas.draw()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        if self.orientation == "vertical":
            self.line.set_xdata([value, value])
        else:
            self.line.set_ydata([value, value])
        self._pos = value
        self.ax.figure.canvas.draw()


def manual_crop(image):
    """Use the mouse to select the ROI for cropping"""

    nx, ny, nz = image.shape

    roi = ThreeDSelector(nx, ny, nz, image)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    rect_props = dict(fill=False, linestyle="-", edgecolor="orange")
    line_props = dict(color="r", linestyle="--")

    lines = {
        "top": {"vertical": None, "horizontal": None},
        "side": {"vertical": None, "horizontal": None},
        "front": {"vertical": None, "horizontal": None},
    }

    img_plots = {
        "top": axs[2].imshow(image[:, :, nz // 2], cmap="gray"),
        "side": axs[1].imshow(image[:, ny // 2, :], cmap="gray"),
        "front": axs[0].imshow(image[nx // 2, :, :], cmap="gray"),
    }

    axs[0].set_title("Front view")
    axs[0].axis("off")
    front_rect = RectangleSelector(
        axs[0],
        roi.select_front,
        interactive=True,
        props=rect_props,
        button=1,
    )
    # vline = axs[0].axhline(ny // 2, color="r", linestyle="--", picker=True)
    # hline = axs[0].axvline(nz // 2, color="r", linestyle="--", picker=True)

    hline = LineSelector(
        axs[0], ny // 2, "horizontal", roi.callback_front_horizontal, interactive=True, props=line_props
    )
    vline = LineSelector(axs[0], nz // 2, "vertical", roi.callback_front_vertical, interactive=True, props=line_props)
    lines["front"]["horizontal"] = hline
    lines["front"]["vertical"] = vline

    axs[1].set_title("Side view")
    axs[1].axis("off")
    side_rect = RectangleSelector(axs[1], roi.select_side, interactive=True, props=rect_props, button=1)
    hline = LineSelector(
        axs[1], nx // 2, "horizontal", roi.callback_side_horizontal, interactive=True, props=line_props
    )
    vline = LineSelector(axs[1], nz // 2, "vertical", roi.callback_side_vertical, interactive=True, props=line_props)
    lines["side"]["horizontal"] = hline
    lines["side"]["vertical"] = vline

    axs[2].set_title("Top view")
    axs[2].axis("off")
    top_rect = RectangleSelector(axs[2], roi.select_top, interactive=True, props=rect_props, button=1)
    hline = LineSelector(
        axs[2], nx // 2, "horizontal", roi.callback_top_horizontal, interactive=True, props=line_props
    )
    vline = LineSelector(axs[2], ny // 2, "vertical", roi.callback_top_vertical, interactive=True, props=line_props)
    lines["top"]["horizontal"] = hline
    lines["top"]["vertical"] = vline

    roi.set_selectors(front_rect, side_rect, top_rect)
    roi.set_img_plots(img_plots)
    roi.set_line_selectors(lines)
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
        info = self.context["info"]
        images = []
        for slice in self.context["slices"]:
            ref_image = data[(data["index"] == 0) & (data["slice_integer"] == slice)]["image"].values[0]
            images.append(ref_image)

        image = np.stack(images, axis=0)

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

        info["n_slices"] = self.slice[1] - self.slice[0]
        info["img_size"] = (self.row[1] - self.row[0], self.col[1] - self.col[0])

        self.context["data"], self.context["slices"], self.context["info"] = data, slices, info

    def _save_crop(self):
        """Saves the crop values"""
        crop = dict(slice=self.slice, row=self.row, col=self.col)
        with open(os.path.join(self.settings["session"], "crop.yaml"), "w") as handle:
            yaml.dump(crop, handle)
