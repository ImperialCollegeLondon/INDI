import os

import cv2 as cv
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray

from .polygon_selector import PolygonSelectorSpline


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

    # catch the case where there is no endocardium
    try:
        # check there is only one epicardium
        assert len(hier_matches) == 1, "We have detected more than one endocardium!"
        hier_pos = int(hier_matches[0][0])
        endo_contour = np.squeeze(contours[hier_pos])
    except IndexError:
        endo_contour = np.array([])
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


def clean_image(img, factor):
    clean_img = np.copy(img)
    clean_img[img < factor] = 0
    mask = np.ones(img.shape)
    mask[img < factor] = 0
    return clean_img, mask


def get_mask_from_poly(poly, dims):
    """
    Get a mask from a polygon
    """
    # create a mask with zeros
    mask = np.zeros(dims)
    # fill the polygon with ones
    mask = cv.fillPoly(mask, np.array([poly]).astype(np.int32), color=1)
    return mask


def reduce_polygon(polygon_coords: NDArray, angle_th: float = 0, distance_th: float = 0):
    """
    Reduce the number of points in a polygon by removing points that are close to each other
    and have a small angle between them.

    Code from here: https://stackoverflow.com/questions/48562739/reducing-number-of-nodes-in-polygons-using-python

    Parameters
    ----------
    polygon_coords
    angle_th
    distance_th

    Returns
    -------

    reduced polygon

    """
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]

    # convert the polygon coordinates into a NumPy array
    polygon = np.array(polygon_coords)

    while len(points_removed):
        points_removed = list()
        for i in range(0, len(polygon) - 2, 2):
            v01 = polygon[i - 1] - polygon[i]
            v12 = polygon[i] - polygon[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i)
                continue
            if d01 * d12 != 0:
                x = np.sum(v01 * v12) / (d01 * d12)
                if np.abs(x) <= 1:
                    angle = np.arccos(x)
                    if angle < angle_th_rad:
                        points_removed.append(i)
        polygon = np.delete(polygon, points_removed, axis=0)

    # Convert the reduced polygon back to a list
    reduced_polygon = polygon.tolist()

    return reduced_polygon


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
        average denoised and normalized images for each slice
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
                markersize=5,
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


class Actions:
    def __init__(
        self,
        fig: matplotlib.figure.Figure,
        ax_seg: matplotlib.axes.Axes,
        ax_preview: matplotlib.axes.Axes,
        ax_slider: matplotlib.axes.Axes,
        cont_img: NDArray,
        map1: NDArray,
        map2: NDArray,
        *,
        initial_endo_poly=None,
        initial_epi_poly=None,
        cont_img_props={},
        map1_props={},
        map2_props={},
        slider_props={},
        num_points=100,
        spline_props=None,
        line_props=None,
    ) -> None:
        self.fig = fig
        self.ax_seg = ax_seg
        self.ax_preview = ax_preview
        self.ax_slider = ax_slider

        self.epi_selected = False

        def select_epi(_):
            self.draw_preview()
            self.epi_selected = True

        self.epi_poly = PolygonSelectorSpline(
            ax_seg, select_epi, useblit=True, num_points=num_points, curve_props=spline_props, props=line_props
        )

        (self.endo_preview,) = self.ax_preview.plot([], [], color="tab:orange", lw=2, alpha=0.5)
        (self.epi_preview,) = self.ax_preview.plot([], [], color="tab:orange", lw=2, alpha=0.5)
        (self.ip_preview,) = self.ax_preview.plot([], [], "X", color="tab:green", markersize=8, alpha=0.8)

        if initial_epi_poly is not None:
            self.epi_poly.verts = initial_epi_poly
            self.epi_selected = True
        self.epi_poly.set_active(False)
        self.epi_poly.set_visible(False)

        self.endo_selected = False

        def select_endo(_):
            self.draw_preview()
            self.endo_selected = True

        self.endo_poly = PolygonSelectorSpline(
            ax_seg, select_endo, useblit=True, num_points=num_points, curve_props=spline_props, props=line_props
        )
        if initial_endo_poly is not None:
            self.endo_poly.verts = initial_endo_poly
            self.endo_selected = True
        self.endo_poly.set_active(False)
        self.endo_poly.set_visible(False)

        slider_props.setdefault("label", "Threshold")
        slider_props.setdefault("valmin", 0.0)
        slider_props.setdefault("valmax", 1.0)
        self.slider = Slider(ax_slider, **slider_props)
        # define that the update function will run when the slider is moved
        self.slider.on_changed(self.update_slider)

        self.ip_selected = False
        self.ip = {"inferior": np.full(2, np.nan), "anterior": np.full(2, np.nan)}
        self.ip_current = "anterior"  # 'inferior' | 'anterior'
        (self.ip_plot_aip,) = self.ax_seg.plot([], [], "^", color="tab:red", markersize=8, alpha=0.5)
        (self.ip_plot_iip,) = self.ax_seg.plot([], [], "v", color="tab:red", markersize=8, alpha=0.5)
        self.ip_plot_aip.set_visible(False)
        self.ip_plot_iip.set_visible(False)
        self.ip_event_id = self.fig.canvas.mpl_connect("button_press_event", self.update_ip)
        self.fig.canvas.mpl_disconnect(self.ip_event_id)

        if initial_epi_poly is not None and initial_endo_poly is not None:
            # If we are here is because we have the contours already, likely from an AI model.
            # Set them both to visible, and if no editing needs to be done, we just need to
            # define the insertion points, otherwise we can press the buttons to edit the contours.
            self.draw_preview()
            self.epi_poly.set_visible(True)
            self.endo_poly.set_visible(True)

        self.mask = np.ones_like(cont_img)
        self.cont_img = cont_img
        self.cont_img_org = cont_img.copy()
        self.maps_org = [map1.copy(), map2.copy()]
        self.maps = [map1, map2]
        self.maps_props = [map1_props, map2_props]
        self.map_use = 0
        self.cont_img_props = cont_img_props
        self.seg_on_map = False

        self.fig.canvas.draw_idle()
        self.bg_seg = self.fig.canvas.copy_from_bbox(self.ax_seg.get_tightbbox())
        self.bg_preview = self.fig.canvas.copy_from_bbox(self.ax_preview.get_tightbbox())

    def draw_images(self):
        if self.seg_on_map:
            imgs = self.ax_seg.images
            if len(imgs) > 0:
                imgs[0].set_data(self.maps[self.map_use])
                imgs[0].set(**self.maps_props[self.map_use])
            else:
                self.ax_seg.imshow(self.maps[self.map_use], **self.maps_props[self.map_use])

            imgs = self.ax_preview.images
            if len(imgs) > 0:
                imgs[0].set_data(self.cont_img)
                imgs[0].set(**self.cont_img_props)
            else:
                self.ax_preview.imshow(self.cont_img, **self.cont_img_props)
        else:
            imgs = self.ax_seg.images
            if len(imgs) > 0:
                imgs[0].set_data(self.cont_img)
                imgs[0].set(**self.cont_img_props)
            else:
                self.ax_seg.imshow(self.cont_img, **self.cont_img_props)

            imgs = self.ax_preview.images
            if len(imgs) > 0:
                imgs[0].set_data(self.maps[self.map_use])
                imgs[0].set(**self.maps_props[self.map_use])
            else:
                self.ax_preview.imshow(self.maps[self.map_use], **self.maps_props[self.map_use])

    def draw_preview(self):
        self.endo_preview.set_data(self.endo_poly.curve_points[:, 0], self.endo_poly.curve_points[:, 1])
        self.epi_preview.set_data(self.epi_poly.curve_points[:, 0], self.epi_poly.curve_points[:, 1])
        self.ip_preview.set_data(
            [[self.ip["inferior"][0], self.ip["anterior"][0]], [self.ip["inferior"][1], self.ip["anterior"][1]]]
        )
        self.fig.canvas.draw_idle()

    def remove_border_btn(self):
        btn_ax = filter(lambda ax: "btn" in ax.get_label(), self.fig.get_axes())
        for ax in btn_ax:
            ax.set_axis_off()

    def set_border_btn(self, ax):
        ax.set_axis_on()
        ax.set_facecolor("#FFFFFF00")
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2.5)  # change width
            ax.spines[axis].set_color("#ED4700")  # change color

    def segment_epi(self, event):
        """Deactivate the endocardium contour and activate the epicardium contour"""
        self.fig.canvas.mpl_disconnect(self.ip_event_id)
        self.ip_plot_aip.set_visible(False)
        self.ip_plot_iip.set_visible(False)
        self.remove_border_btn()
        self.set_border_btn(event.inaxes)

        self.epi_poly.set_active(True)
        self.epi_poly.set_visible(True)

        self.endo_poly.set_active(False)
        self.endo_poly.set_visible(False)

        self.ax_seg.set_title("Click epicardial points")
        self.fig.canvas.draw_idle()
        # self.draw()

    def segment_endo(self, event):
        """Deactivate the epicardium contour and activate the endocardium contour"""
        self.fig.canvas.mpl_disconnect(self.ip_event_id)
        self.ip_plot_aip.set_visible(False)
        self.ip_plot_iip.set_visible(False)
        self.remove_border_btn()
        self.set_border_btn(event.inaxes)

        self.epi_poly.set_active(False)
        self.epi_poly.set_visible(False)

        self.endo_poly.set_active(True)
        self.endo_poly.set_visible(True)

        self.ax_seg.set_title("Click endocardial points")
        self.fig.canvas.draw_idle()
        # self.draw()

    def swap_images(self, _):
        """Swap between segmenting the contrast image and the map"""
        self.seg_on_map = not self.seg_on_map
        self.draw_images()
        self.bg_seg = self.fig.canvas.copy_from_bbox(self.ax_seg.get_tightbbox())
        self.bg_preview = self.fig.canvas.copy_from_bbox(self.ax_preview.get_tightbbox())
        # self.draw()
        self.fig.canvas.draw_idle()

    def swap_maps(self, _):
        """Swap between the two maps"""
        self.map_use = (self.map_use + 1) % 2
        self.draw_images()
        self.bg_seg = self.fig.canvas.copy_from_bbox(self.ax_seg.get_tightbbox())
        self.bg_preview = self.fig.canvas.copy_from_bbox(self.ax_preview.get_tightbbox())
        self.fig.canvas.draw_idle()
        # self.draw()

    def on_scroll(self, event):
        """Use the scroll wheel to update the threshold"""
        # scroll event will move the slider up to its limits, after that it won't move anymore
        # scroll up event
        if event.button == "up" and (self.slider.val + self.slider.valstep) < (
            self.slider.valmax + self.slider.valstep
        ):
            self.slider.set_val(self.slider.val + self.slider.valstep)
        # scroll down event
        if event.button == "down" and (self.slider.val - self.slider.valstep) > (
            self.slider.valmin - self.slider.valstep
        ):
            self.slider.set_val(self.slider.val - self.slider.valstep)
        self.update_slider(None)

    def update_slider(self, _):
        """Apply the mask"""
        self.cont_img, self.mask = clean_image(self.cont_img_org, self.slider.val)

        self.maps[0] = self.maps_org[0] * self.mask
        self.maps[1] = self.maps_org[1] * self.mask

        self.draw_images()

        self.bg_seg = self.fig.canvas.copy_from_bbox(self.ax_seg.get_tightbbox())
        self.bg_preview = self.fig.canvas.copy_from_bbox(self.ax_preview.get_tightbbox())
        self.fig.canvas.draw_idle()

        # self.draw()

    def pick_ip(self, event):
        """Activate the intersection point code"""
        self.remove_border_btn()
        self.set_border_btn(event.inaxes)

        self.epi_poly.set_active(False)
        self.epi_poly.set_visible(False)

        self.endo_poly.set_active(False)
        self.endo_poly.set_visible(False)

        self.fig.canvas.draw_idle()

        self.ax_seg.set_title("Click on the anterior and then inferior insertion points")
        self.ip_plot_aip.set_visible(True)
        self.ip_plot_iip.set_visible(True)
        self.ip_event_id = self.fig.canvas.mpl_connect("button_press_event", self.update_ip)

    def update_ip(self, event):
        """Click on the figure to register the intersection points"""
        if event.inaxes is self.ax_seg:
            self.ip[self.ip_current][0] = event.xdata
            self.ip[self.ip_current][1] = event.ydata
            self.ip_plot_aip.set_data([self.ip["anterior"][0]], [self.ip["anterior"][1]])
            self.ip_plot_iip.set_data([self.ip["inferior"][0]], [self.ip["inferior"][1]])
            if self.ip_current == "anterior":
                self.ip_current = "inferior"
                self.ip_selected = True
            else:
                self.ip_current = "anterior"

        self.draw_preview()
        self.fig.canvas.draw_idle()
        # self.draw()


def manual_lv_segmentation(
    mask_3c: NDArray,
    average_map: NDArray,
    ha_map: NDArray,
    md_map: NDArray,
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
    average_map : NDArray
        average denoised and normalized images for each slice
    ha_map : NDArray
        HA maps for each slice
    md_map : NDArray
        MD maps for each slice
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
        epi_contour = None
        endo_contour = None
    else:
        try:
            # get endo and epi contours from mask
            # interpolate the contours to a small number of points and
            # delete the last point, which is equal to the first one
            # use the points for the initial polygon
            epi_contour, endo_contour = get_sa_contours(lv_masks)
            epi_contour = np.delete(epi_contour, -1, 0)
            endo_contour = np.delete(endo_contour, -1, 0)

            # reduce the number of points in the contours in order to be easily editable
            # epi_contour = reduce_polygon(epi_contour, angle_th=10, distance_th=10)
            # endo_contour = reduce_polygon(endo_contour, angle_th=10, distance_th=10)

            n_poly_points = 14
            if len(epi_contour) > n_poly_points:
                steps = np.linspace(0, len(epi_contour) - 1, n_poly_points, dtype=int)
                epi_contour = epi_contour[steps, :]
            if len(endo_contour) > n_poly_points:
                steps = np.linspace(0, len(endo_contour) - 1, n_poly_points, dtype=int)
                endo_contour = endo_contour[steps, :]

        except:
            # if something goes wrong in finding the endo and epi lines. Most likely the U-Net
            # mask does not have the right shape. So we need to manually draw the contours from scratch.
            epi_contour = None
            endo_contour = None

    # plot the magnitude image to be ROI'd
    # retina screen resolution
    # It should be detected automatically otherwise when using a non retina display the app will look too large
    my_dpi = 100
    if mask_3c.shape[0] > mask_3c.shape[1]:
        fig, ax = plt.subplots(
            1,
            2,
            dpi=my_dpi,
            figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
            num="Slice " + str(slice_idx) + " of " + str(len(slices) - 1),
        )
    else:
        fig, ax = plt.subplots(
            2,
            1,
            dpi=my_dpi,
            figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
            num="Slice " + str(slice_idx) + " of " + str(len(slices) - 1),
        )

    # leave some space for the buttons
    fig.subplots_adjust(left=0.2, bottom=0.2)
    map1_props = dict(cmap=colormaps["HA"], clim=[-90, 90], alpha=0.5, interpolation="none")
    map2_props = dict(cmap=colormaps["MD"], clim=[0.0, 2.5], alpha=0.5, interpolation="none")
    cont_img_props = dict(cmap="Greys_r", clim=[0.0, 0.85], alpha=0.8, interpolation="none")

    # axis where the magnitude image will be shown initially
    ax[0].imshow(average_map, **cont_img_props)
    ax[0].axis("off")
    ax[0].label = "mag"
    # axis where the HA map will be shown initially
    ax[1].imshow(ha_map, **map1_props)
    ax[1].axis("off")
    ax[1].label = "ha"

    ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    slider_props = dict(label="Threshold: ", valmin=0, valmax=1, valinit=0.01, valstep=0.01, orientation="horizontal")
    line_style = dict(color="tab:brown", linestyle="None", linewidth=0.01, alpha=0.8, markersize=3)
    curve_style = dict(color="tab:brown", lw=1, alpha=0.9, label="ROI")
    actions = Actions(
        fig,
        ax[0],
        ax[1],
        ax_slider,
        average_map,
        ha_map,
        md_map,
        num_points=n_points,
        initial_epi_poly=epi_contour,
        initial_endo_poly=endo_contour,
        slider_props=slider_props,
        line_props=line_style,
        spline_props=curve_style,
        map1_props=map1_props,
        map2_props=map2_props,
        cont_img_props=cont_img_props,
    )

    # add the buttons to the figure

    # epi button
    epi_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "epicardium.png"))
    ax_epi = fig.add_axes([0.05, 0.75, 0.10, 0.10], label="epi_btn")
    button_epi = Button(ax_epi, label="", image=epi_icon, hovercolor="#FFFFFF00")
    button_epi.on_clicked(actions.segment_epi)

    # endo button
    endo_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "endocardium.png"))
    ax_endo = fig.add_axes([0.05, 0.60, 0.10, 0.10], label="endo_btn")
    button_endo = Button(ax_endo, label="", image=endo_icon, hovercolor="#FFFFFF00")
    button_endo.on_clicked(actions.segment_endo)

    # insertion points button
    ip_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "insertion_points.png"))
    ax_ip = fig.add_axes([0.05, 0.45, 0.10, 0.10], label="ip_btn")
    button_ip = Button(ax_ip, label="", image=ip_icon, hovercolor="#FFFFFF00")
    button_ip.on_clicked(actions.pick_ip)

    # swap images button
    si_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "swap_images.png"))
    ax_swap_images = fig.add_axes([0.05, 0.30, 0.10, 0.10], label="si_btn")
    ax_swap_images.axis("off")
    button_si = Button(ax_swap_images, label="", image=si_icon, hovercolor="#FFFFFF00")
    button_si.on_clicked(actions.swap_images)

    # ha/md map button
    hm_icon = plt.imread(os.path.join(settings["code_path"], "assets", "icons", "ha_md.png"))
    ax_ha_md = fig.add_axes([0.05, 0.15, 0.10, 0.10], label="hm_btn")
    ax_ha_md.axis("off")
    button_hm = Button(ax_ha_md, label="", image=hm_icon, hovercolor="#FFFFFF00")
    button_hm.on_clicked(actions.swap_maps)

    fig.canvas.mpl_connect("scroll_event", actions.on_scroll)
    actions.remove_border_btn()

    plt.show(block=True)

    # retrieve the threshold mask
    thr_mask = actions.mask

    # store segmentation information from the buttons' callbacks
    segmentation = {}
    # The epicardium needs to be defined. If not slice will be removed.
    # The endocardium and the two insertion points is optional.
    if actions.epi_selected:
        segmentation["epicardium"] = np.array(actions.epi_poly.curve_points)
    else:
        segmentation["epicardium"] = np.array([])

    if actions.endo_selected:
        segmentation["endocardium"] = np.array(actions.endo_poly.curve_points)

    else:
        segmentation["endocardium"] = np.array([])

    if actions.ip_selected:
        segmentation["anterior_ip"] = actions.ip["anterior"]
        segmentation["inferior_ip"] = actions.ip["inferior"]
    else:
        segmentation["anterior_ip"] = np.array([])
        segmentation["inferior_ip"] = np.array([])

    return segmentation, thr_mask
