import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, PolygonSelector, Slider
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep


def spline_interpolate_contour(contour, n_points, join_ends=False):
    """
    Interpolate a contour to a spline curve with n_points
    """
    if join_ends:
        contour = np.append(contour, [contour[0]], axis=0)
    x = contour[:, 0]
    y = contour[:, 1]
    tck, _ = splprep([x, y], s=0, per=True)
    xi, yi = splev(np.linspace(0, 1, n_points), tck, der=0)
    return np.array([xi, yi]).T


def get_sa_contours(lv_mask):
    # get the contours of the epicardium
    # From stackoverflow: In the hierarchy (hier), the fourth index tells you, to which outer
    # (or parent) contour a possible inner (or child) contour is related. Most outer contours have an
    # index of -1, all others have non-negative values. So I should get 2 contours. The epicardium
    # should have the hierarchy x, x, x, -1. It should also be the longest contour.
    c_mask = np.array(lv_mask * 255, dtype=np.uint8)
    contours, hier = cv.findContours(c_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # there might be loose regions/point creating additional small contours
    # so we need to remove these and keep only the two longest
    # get the length of each contour
    contours_len = [len(contours[i]) for i in range(len(contours))]
    # get the indices of the two longest contours
    idx = np.argsort(contours_len)[-2:]
    # get the two longest contours
    contours = [contours[i] for i in idx]
    # get the hierarchy of the two longest contours
    hier = np.stack([hier[0, i] for i in idx], axis=0)

    # find epicardial contour
    hier_matches = list(np.where(hier[:, 3] == -1))
    # check there is only one epicardium
    assert len(hier_matches) == 1, "We have detected more than one epicardium!"
    hier_pos = hier_matches[0][0]
    epi_contour = np.squeeze(contours[hier_pos])

    # find endocardial contour
    hier_matches = list(np.where(hier[:, 3] != -1))
    # check there is only one epicardium
    assert len(hier_matches) == 1, "We have detected more than one endocardium!"
    hier_pos = int(hier_matches[0][0])
    endo_contour = np.squeeze(contours[hier_pos])

    return epi_contour, endo_contour


def get_epi_contour(lv_mask):
    # get the contours of the epicardium
    # From stackoverflow: In the hierarchy (hier), the fourth index tells you, to which outer
    # (or parent) contour a possible inner (or child) contour is related. Most outer contours have an
    # index of -1, all others have non-negative values. So I should get 2 contours. The epicardium
    # should have the hierarchy x, x, x, -1. It should also be the longest contour.
    c_mask = np.array(lv_mask * 255, dtype=np.uint8)
    contours, hier = cv.findContours(c_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # there might be loose regions/point creating additional small contours
    # so we need to remove these and keep only the two longest
    # get the length of each contour
    contours_len = [len(contours[i]) for i in range(len(contours))]
    # get the indices of the longest contours
    idx = np.argsort(contours_len)[-1:]
    # get the longest contours
    contours = [contours[i] for i in idx]
    # get the hierarchy of the longest contours
    hier = np.stack([hier[0, i] for i in idx], axis=0)

    # find epicardial contour
    hier_matches = list(np.where(hier[:, 3] == -1))
    # check there is only one epicardium
    assert len(hier_matches) == 1, "We have detected more than one epicardium!"
    hier_pos = hier_matches[0][0]
    epi_contour = np.squeeze(contours[hier_pos])

    return epi_contour


class define_roi_border(object):
    """
    Define the ROI border with a polygon which will then be interpolated to a spline.
    If there is a pre-countour, this would be used to define the initial polygon.
    """

    def __init__(self, ax, title_message, pre_contour, second_axis_lines, seg_mask, img_shape):
        # print("Select points in the figure by enclosing them within a polygon.")
        # print("Press the 'esc' key to start a new polygon.")
        # print("Try holding the 'shift' key to move all of the vertices.")
        # print("Try holding the 'ctrl' key to move a single vertex.")
        # initiate the class
        self.canvas = ax.figure.canvas
        # store image shape
        self.img_shape = img_shape
        # initiate segmentation mask
        self.mask = seg_mask
        # set the title message
        ax.set_title(title_message)
        # if there were previously defined lines or points, remove them
        for line in self.canvas.figure.axes[0].lines:
            line.remove()
        # set the line style
        line_style = dict(color="tab:brown", linestyle="None", linewidth=0.01, alpha=0.8)
        self.poly = PolygonSelector(ax, self.onselect, useblit=True, props=line_style)
        self.title_message = title_message
        self.second_axis_lines = second_axis_lines
        if pre_contour.any():
            # interpolate the points to a spline curve
            array = spline_interpolate_contour(pre_contour, 100, join_ends=True)
            self.spline_points = array
            # plot the spline interpolation
            self.canvas.figure.axes[0].plot(
                array[:, 0],
                array[:, 1],
                color="tab:brown",
                lw=1,
                alpha=0.9,
                label="ROI",
            )
            # add the contour points to the polygon
            self.poly.verts = pre_contour

            # secondary axis
            for line in self.canvas.figure.axes[1].lines:
                line.remove()
            if self.title_message == "Click epicardial points":
                self.second_axis_lines["epi"] = array
            elif self.title_message == "Click endocardial points":
                self.second_axis_lines["endo"] = array

            # draw mask in secondary axis
            # remove any previous drawn mask
            if len(self.canvas.figure.axes[1].images) > 1:
                self.canvas.figure.axes[1].images[-1].remove()
            # get mask from polygon
            mask = get_mask_from_poly(
                array.astype(np.int32),
                self.img_shape,
            )

            if self.title_message == "Click endocardial points":
                # erode endo mask in order to keep the endo line inside the myocardial ROI
                kernel = np.ones((2, 2), np.uint8)
                mask = cv.erode(mask, kernel, iterations=1)

            # store mask in dictionary and also subtract previous mask if exists
            if self.title_message == "Click epicardial points":
                if self.mask["endo"] is not None:
                    mask = mask - self.mask["endo"]
                self.mask["epi"] = mask
            elif self.title_message == "Click endocardial points":
                if self.mask["epi"] is not None:
                    mask = self.mask["epi"] - mask
                self.mask["endo"] = mask
            self.canvas.figure.axes[1].imshow(mask, alpha=0.5 * mask, cmap="Oranges")

            for idx, name in enumerate(self.second_axis_lines):
                if self.second_axis_lines[name] is not None:
                    c_array = self.second_axis_lines[name]
                    if len(c_array) > 2:
                        # self.canvas.figure.axes[1].plot(
                        #     np.array(c_array[:, 0]).astype(np.int32),
                        #     np.array(c_array[:, 1]).astype(np.int32),
                        #     color="tab:orange",
                        #     lw=2,
                        #     alpha=0.5,
                        # )
                        pass
                    else:
                        self.canvas.figure.axes[1].plot(
                            c_array[:, 0],
                            c_array[:, 1],
                            "o",
                            color="tab:orange",
                            markersize=10,
                        )
            # update canvas
            self.canvas.draw_idle()

    def onselect(self, verts):
        # function to run when the polygon closes
        # interpolate the polygon points to a spline curve with no end points
        # remove the previous spline if there is one.
        if len(self.canvas.figure.axes[0].lines) > 2:
            self.canvas.figure.axes[0].lines[-1].remove()

        # get the points from the polygon
        array = np.array([*verts])

        # interpolate the points to a spline curve
        array = spline_interpolate_contour(array, 100, join_ends=True)
        self.spline_points = array
        # plot the spline interpolation
        self.canvas.figure.axes[0].plot(array[:, 0], array[:, 1], color="tab:brown", lw=1, alpha=0.9, label="ROI")
        # secondary axis
        for line in self.canvas.figure.axes[1].lines:
            line.remove()
        if self.title_message == "Click epicardial points":
            self.second_axis_lines["epi"] = array
        elif self.title_message == "Click endocardial points":
            self.second_axis_lines["endo"] = array

        # draw mask
        # remove any previous drawn mask
        if len(self.canvas.figure.axes[1].images) > 1:
            self.canvas.figure.axes[1].images[-1].remove()
        # get mask from polygon
        mask = get_mask_from_poly(
            array.astype(np.int32),
            self.img_shape,
        )

        if self.title_message == "Click endocardial points":
            # erode endo mask in order to keep the endo line inside the myocardial ROI
            kernel = np.ones((2, 2), np.uint8)
            mask = cv.erode(mask, kernel, iterations=1)

        # store mask in dictionary and also subtract previous mask if exists
        if self.title_message == "Click epicardial points":
            if self.mask["endo"] is not None:
                mask = mask - self.mask["endo"]
            self.mask["epi"] = mask
        elif self.title_message == "Click endocardial points":
            if self.mask["epi"] is not None:
                mask = self.mask["epi"] - mask
            self.mask["endo"] = mask
        self.canvas.figure.axes[1].imshow(mask, alpha=0.5 * mask, cmap="Oranges")

        # draw all lines
        for idx, name in enumerate(self.second_axis_lines):
            if self.second_axis_lines[name] is not None:
                c_array = self.second_axis_lines[name]
                if len(c_array) > 2:
                    # self.canvas.figure.axes[1].plot(
                    #     c_array[:, 0],
                    #     c_array[:, 1],
                    #     color="tab:orange",
                    #     lw=2,
                    #     alpha=0.5,
                    # )
                    pass
                else:
                    self.canvas.figure.axes[1].plot(
                        c_array[:, 0],
                        c_array[:, 1],
                        "o",
                        color="tab:orange",
                        markersize=10,
                    )
        # update canvas
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()


def clean_image(img, factor):
    clean_img = np.copy(img)
    clean_img[img < factor] = 0
    mask = np.ones(img.shape)
    mask[img < factor] = 0

    return clean_img, mask


class scrool_slider:
    """
    Display image and a slider. When the slider is moved, the image is updated.
    slider can be moved with the mouse wheel or by clicking on the slider.
    """

    def __init__(self, fig, ax_0, img_0, ax_img_0, ax_1, img_1, ax_img_1):
        self.ax_0 = ax_0
        self.img_0 = img_0
        self.ax_img_0 = ax_img_0

        self.ax_1 = ax_1
        self.img_1 = img_1
        self.ax_img_1 = ax_img_1

        self.fig = fig

        # Make a vertically oriented slider to control the amplitude
        axamp = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.amp_slider = Slider(
            ax=axamp,
            label="Threshold: ",
            valmin=0,
            valmax=1,
            valinit=0.1,
            valstep=0.01,
            orientation="horizontal",
        )
        # define that the update function will run when the slider is moved
        self.amp_slider.on_changed(self.update)
        # update the image when the slider is moved
        self.update(self.amp_slider.val)

    def on_scroll(self, event):
        # scroll event will move the slider up to its limits, after that it won't move anymore
        # scroll up event
        if event.button == "up" and (self.amp_slider.val + self.amp_slider.valstep) < (
            self.amp_slider.valmax + self.amp_slider.valstep
        ):
            self.amp_slider.set_val(self.amp_slider.val + self.amp_slider.valstep)
        # scroll down event
        if event.button == "down" and (self.amp_slider.val - self.amp_slider.valstep) > (
            self.amp_slider.valmin - self.amp_slider.valstep
        ):
            self.amp_slider.set_val(self.amp_slider.val - self.amp_slider.valstep)
        # update the image when the slider is moved
        self.update(self.amp_slider.val)

    def update(self, val):
        # when slider moves, this function is called
        #  we need to aply the clean image on the magnitude image not on the HA map
        # so we need to find out which axis contains that image
        c_lims = self.fig.axes[0].images[0].get_clim()
        self.updated_img_0, self.mask = clean_image(self.img_0, val)
        self.updated_img_1 = self.img_1 * self.mask
        if c_lims == (0.0, 0.85):
            self.ax_img_0.set_array(self.updated_img_0)
            self.ax_img_1.set_array(self.updated_img_1)
        else:
            self.ax_img_1.set_array(self.updated_img_0)
            self.ax_img_0.set_array(self.updated_img_1)

        self.fig.canvas.draw_idle()

    def get_img_and_mask(self):
        # function to retrieve the current image and mask
        return self.updated_img, self.mask


class click_insertion_points(object):
    """
    Click on the insertion points.
    They can be moved after both are defined by clicking close to them
    """

    def __init__(self, ax, second_axis_lines):
        self.canvas = ax.figure.canvas
        self.binding_id = self.canvas.mpl_connect("button_press_event", self.on_click)
        self.ip_x = []
        self.ip_y = []
        self.ip_dict = {}
        self.second_axis_lines = second_axis_lines
        # if there were previously defined lines or points, remove them
        for line in self.canvas.figure.axes[0].lines:
            line.remove()

    def on_click(self, event):
        # limit to just the main axis
        if event.inaxes == self.canvas.figure.axes[0]:
            if len(self.ip_x) == 0:
                # first point is the superior insertion point
                self.ip_x.append(event.xdata)
                self.ip_y.append(event.ydata)
                self.ip_dict["superior"] = self.canvas.figure.axes[0].plot(
                    event.xdata, event.ydata, "^", color="tab:red", markersize=10
                )
                self.canvas.draw_idle()
            elif len(self.ip_x) == 1:
                # second point is the inferior insertion point
                self.ip_x.append(event.xdata)
                self.ip_y.append(event.ydata)
                self.ip_dict["inferior"] = self.canvas.figure.axes[0].plot(
                    event.xdata, event.ydata, "v", color="tab:red", markersize=10
                )
                self.canvas.draw_idle()
            else:
                # if two points have already been defined, move the closest one
                # to the position clicked
                c_sip = [self.ip_x[0], self.ip_y[0]]
                c_iip = [self.ip_x[1], self.ip_y[1]]
                dist_sip = np.sqrt((event.xdata - c_sip[0]) ** 2 + (event.ydata - c_sip[1]) ** 2)
                dist_iip = np.sqrt((event.xdata - c_iip[0]) ** 2 + (event.ydata - c_iip[1]) ** 2)
                if dist_sip < dist_iip:
                    self.ip_x[0] = event.xdata
                    self.ip_y[0] = event.ydata
                    self.ip_dict["superior"][0].remove()
                    self.ip_dict["superior"] = self.canvas.figure.axes[0].plot(
                        event.xdata, event.ydata, "^", color="tab:red", markersize=10
                    )
                    self.canvas.draw_idle()
                else:
                    self.ip_x[1] = event.xdata
                    self.ip_y[1] = event.ydata
                    self.ip_dict["inferior"][0].remove()
                    self.ip_dict["inferior"] = self.canvas.figure.axes[0].plot(
                        event.xdata, event.ydata, "v", color="tab:red", markersize=10
                    )
                    self.canvas.draw_idle()
            # secondary axis
            for line in self.canvas.figure.axes[1].lines:
                line.remove()
            if self.ip_x:
                self.second_axis_lines["ip"] = np.column_stack([self.ip_x, self.ip_y])
            for idx, name in enumerate(self.second_axis_lines):
                if self.second_axis_lines[name] is not None:
                    c_array = self.second_axis_lines[name]
                    if len(c_array) > 2:
                        self.canvas.figure.axes[1].plot(
                            c_array[:, 0],
                            c_array[:, 1],
                            color="tab:orange",
                            lw=2,
                            alpha=0.5,
                        )
                    else:
                        self.canvas.figure.axes[1].plot(
                            c_array[:, 0],
                            c_array[:, 1],
                            "X",
                            color="tab:orange",
                            markersize=10,
                            alpha=0.5,
                        )
            # update canvas
            self.canvas.draw_idle()


def plot_manual_lv_segmentation(
    n_slices: int,
    slices: NDArray,
    segmentation: dict,
    average_maps: NDArray,
    mask_3c: NDArray,
    settings: dict,
    filename: str,
    save_path: str,
):
    """
    Plot the manual segmentation LV contours and insertion points

    Parameters
    ----------
    n_slices : int
        number of slices
    segmentation : dict
        segmentation information
    average_images : NDArray
        average denoised and normalised images for each slice
    mask_3c : NDArray
        mask with  segmentation
    settings : dict
    filename: str,
    save_path: str,
    """

    for slice_idx in slices:
        # alpha mask
        alphas_myocardium = np.copy(mask_3c[slice_idx])
        alphas_myocardium[alphas_myocardium == 2] = 0
        alphas_myocardium[alphas_myocardium > 0.1] = 0.3

        # plot average images and respective masks
        plt.figure(figsize=(10, 10))
        plt.imshow(average_maps[slice_idx], cmap="Greys_r")
        plt.imshow(alphas_myocardium, alpha=alphas_myocardium, vmin=0, vmax=1, cmap="hot")
        plt.axis("off")
        if segmentation[slice_idx]["epicardium"].size != 0:
            plt.scatter(
                segmentation[slice_idx]["epicardium"][:, 0],
                segmentation[slice_idx]["epicardium"][:, 1],
                marker=".",
                s=2,
                color="tab:blue",
                alpha=0.5,
            )
        if segmentation[slice_idx]["endocardium"].size != 0:
            plt.scatter(
                segmentation[slice_idx]["endocardium"][:, 0],
                segmentation[slice_idx]["endocardium"][:, 1],
                marker=".",
                s=2,
                color="tab:red",
                alpha=0.5,
            )
        if segmentation[slice_idx]["anterior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["anterior_ip"][0],
                segmentation[slice_idx]["anterior_ip"][1],
                "2",
                color="tab:orange",
                markersize=10,
                alpha=0.5,
            )
        if segmentation[slice_idx]["inferior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["inferior_ip"][0],
                segmentation[slice_idx]["inferior_ip"][1],
                "1",
                color="tab:orange",
                markersize=10,
                alpha=0.5,
            )
        plt.savefig(
            os.path.join(
                save_path,
                filename + "_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
        plt.close()


def get_mask_from_poly(poly, dims):
    """
    Get a mask from a polygon
    """
    # create a mask with zeros
    mask = np.zeros(dims)
    # fill the polygon with ones
    mask = cv.fillPoly(mask, np.array([poly]).astype(np.int32), color=1)
    return mask


def manual_lv_segmentation(
    mask_3c: NDArray,
    average_maps: NDArray,
    ha_maps: NDArray,
    n_points: int,
    settings: dict,
    colormaps: dict,
    slice_idx: int,
    slices: NDArray,
):
    """
    Manually define the epicardial and endocardial contours and the insertion points
    for the LV.
    The epicardial border is compulsory, the endocardial border and insertion points are optional.
    If no epicardial segmentation slice is going to be marked to be removed.
    An initial LV mask is passed into the function. This mask is used to define the
    epicardial and endocardial initial contours. If the mask is all zeros, then no initial
    contour will be used, and we need to manually draw one.
    This initial mask can be for example a U-Net prediction.

    Parameters
    ----------
    mask_3c : NDArray
        initial mask
    average_maps : NDArray
        average denoised and normalised images for each slice
    ha_maps : NDArray
        HA maps for each slice
    n_points : int
        number of points to interpolate the contours
    settings : dict
    colormaps : dict
    slice_idx : int
        slice index
    slices : NDArray
        slice indices
    """
    lv_masks = mask_3c.copy()
    lv_masks[lv_masks == 2] = 0

    # get the contours of the epicardium and endocardium if not all zeros
    all_zeros = not np.any(lv_masks)
    if all_zeros:
        epi_contour = np.array([])
        endo_contour = np.array([])
        second_axis_lines = {"epi": None, "endo": None, "ip": None}
        seg_mask = {"epi": None, "endo": None}
    else:
        try:
            # get endo and epi contours from mask
            # interpolate the contours to a small number of points and
            # delete the last point, which is equal to the first one
            # use the points for the initial polygon
            epi_contour, endo_contour = get_sa_contours(lv_masks)
            epi_contour = spline_interpolate_contour(epi_contour, n_points=n_points, join_ends=False)
            epi_contour = np.delete(epi_contour, -1, 0)
            endo_contour = spline_interpolate_contour(endo_contour, n_points=n_points, join_ends=False)
            endo_contour = np.delete(endo_contour, -1, 0)
            second_axis_lines = {"epi": None, "endo": None, "ip": None}
            seg_mask = {"epi": None, "endo": None}
        except:
            # if something goes wrong in finding the endo and epi lines. Most likely the U-Net
            # mask does not have the right shape. So we need to manually draw the contours from scratch.
            epi_contour = np.array([])
            endo_contour = np.array([])
            second_axis_lines = {"epi": None, "endo": None, "ip": None}
            seg_mask = {"epi": None, "endo": None}

    class buttons:
        """matplotlib figure buttons"""

        def epi(self, event):
            # draw epicardial contour
            # disconnect previous polygons or on_click events if they exist
            if hasattr(self, "epi_spline"):
                self.epi_spline.disconnect()
            if hasattr(self, "endo_spline"):
                self.endo_spline.disconnect()
            if hasattr(self, "ip"):
                plt.disconnect(self.ip.binding_id)

            """draw the epicardial border"""
            self.epi_spline = define_roi_border(
                event.canvas.figure.axes[0],
                "Click epicardial points",
                epi_contour,
                second_axis_lines,
                seg_mask,
                lv_masks.shape,
            )

        def endo(self, event):
            # draw endocardial contour
            # disconnect previous polygons or on_click events if they exist
            if hasattr(self, "epi_spline"):
                self.epi_spline.disconnect()
            if hasattr(self, "endo_spline"):
                self.endo_spline.disconnect()
            if hasattr(self, "ip"):
                plt.disconnect(self.ip.binding_id)

            """draw the endocardial border"""
            self.endo_spline = define_roi_border(
                event.canvas.figure.axes[0],
                "Click endocardial points",
                endo_contour,
                second_axis_lines,
                seg_mask,
                lv_masks.shape,
            )

        def click(self, event, second_axis_lines):
            # click on the two insertion points
            # first anterior, then inferior
            # disconnect previous polygons if they exist
            event.canvas.figure.axes[0].set_title("Click on the anterior and then inferior insertion points.")
            if hasattr(self, "epi_spline"):
                self.epi_spline.disconnect()
            if hasattr(self, "endo_spline"):
                self.endo_spline.disconnect()
            self.ip = click_insertion_points(event.canvas.figure.axes[0], second_axis_lines)

        def swap_images(self, event):
            # swap the images in the figure
            a = event.canvas.figure.axes[0].images[0].get_array()
            b = event.canvas.figure.axes[1].images[0].get_array()

            for idx in range(2):
                if idx == 0:
                    event.canvas.figure.axes[idx].images[0].set_array(b)
                elif idx == 1:
                    event.canvas.figure.axes[idx].images[0].set_array(a)

                c_lims = event.canvas.figure.axes[idx].images[0].get_clim()
                if c_lims == (-90.0, 90.0):
                    event.canvas.figure.axes[idx].images[0].set_clim((0, 0.85))
                    event.canvas.figure.axes[idx].images[0].set_alpha(1.0)
                    event.canvas.figure.axes[idx].images[0].set_cmap("Greys_r")
                elif c_lims == (0.0, 0.85):
                    event.canvas.figure.axes[idx].images[0].set_clim((-90, 90))
                    event.canvas.figure.axes[idx].images[0].set_alpha(0.5)
                    event.canvas.figure.axes[idx].images[0].set_cmap(colormaps["HA"])

            event.canvas.draw_idle()

    # plot the magnitude image to be ROI'd
    # retina screen resolution
    my_dpi = 192
    if mask_3c.shape[0] > mask_3c.shape[1]:
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
            num="Slice " + str(slice_idx) + " of " + str(len(slices) - 1),
        )
    else:
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
            num="Slice " + str(slice_idx) + " of " + str(len(slices) - 1),
        )
    # leave some space for the buttons
    fig.subplots_adjust(left=0.2, bottom=0.2)
    # axis where ROIs will be drawn
    ax_img_0 = ax[0].imshow(average_maps, cmap="Greys_r", vmin=0, vmax=0.85)
    ax[0].axis("off")
    # axis where latest ROIs will be shown
    ax_img_1 = ax[1].imshow(ha_maps, colormaps["HA"], vmin=-90, vmax=90, alpha=0.5)
    ax[1].axis("off")

    # add the buttons to the figure
    callback = buttons()
    # epi button
    epi_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "epicardium.png"))
    ax_epi = fig.add_axes([0.05, 0.75, 0.10, 0.10])
    ax_epi.axis("off")
    button_epi = Button(ax_epi, label="", image=epi_icon)
    button_epi.on_clicked(callback.epi)
    # endo button
    endo_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "endocardium.png"))
    ax_endo = fig.add_axes([0.05, 0.60, 0.10, 0.10])
    ax_endo.axis("off")
    button_endo = Button(ax_endo, label="", image=endo_icon)
    button_endo.on_clicked(callback.endo)
    # insertion points button
    ip_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "insertion_points.png"))
    ax_ip = fig.add_axes([0.05, 0.45, 0.10, 0.10])
    ax_ip.axis("off")
    button_ip = Button(ax_ip, label="", image=ip_icon)
    button_ip.on_clicked(lambda x: callback.click(x, second_axis_lines))
    # swap images button
    si_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "swap_images.png"))
    ax_swap_images = fig.add_axes([0.05, 0.20, 0.10, 0.10])
    ax_swap_images.axis("off")
    button_si = Button(ax_swap_images, label="", image=si_icon)
    button_si.on_clicked(callback.swap_images)

    # slider stuff
    threshold_slider_and_scroll = scrool_slider(
        fig,
        ax[0],
        average_maps,
        ax_img_0,
        ax[1],
        ha_maps,
        ax_img_1,
    )
    fig.canvas.mpl_connect("scroll_event", threshold_slider_and_scroll.on_scroll)

    plt.show()

    # retrieve the threshold mask
    thr_mask = threshold_slider_and_scroll.mask

    # store segmentation information from the buttons' callbacks
    segmentation = {}
    # The epicardium needs to be defined. If not slice will be removed.
    # The endocardium and the two insertion points is optional.
    if hasattr(callback, "epi_spline"):
        segmentation["epicardium"] = callback.epi_spline.spline_points
    else:
        segmentation["epicardium"] = np.array([])

    if hasattr(callback, "endo_spline"):
        segmentation["endocardium"] = callback.endo_spline.spline_points
    else:
        segmentation["endocardium"] = np.array([])

    if hasattr(callback, "ip"):
        segmentation["anterior_ip"] = np.array(
            [
                callback.ip.ip_x[0],
                callback.ip.ip_y[0],
            ]
        )
        segmentation["inferior_ip"] = np.array(
            [
                callback.ip.ip_x[1],
                callback.ip.ip_y[1],
            ]
        )
    else:
        segmentation["anterior_ip"] = np.array([])
        segmentation["inferior_ip"] = np.array([])

    return segmentation, thr_mask
