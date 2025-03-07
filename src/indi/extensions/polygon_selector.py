import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.widgets import RectangleSelector, ToolHandles, _SelectorWidget
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


def dist_to_line(p1, p2, p3):
    """Calculates the distance of the line connecting the points p1 and p2 to point p3"""

    def dist_sqr(p, q):
        return (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2

    dist = dist_sqr(p1, p2)
    if dist == 0:
        raise ValueError("p1 and p2 are coincidental, there is no line")
    u = ((p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1])) / dist
    intersection = p1[0] + u * (p2[0] - p1[0]), p1[1] + u * (p2[1] - p1[1])

    return dist_sqr(intersection, p3) ** 0.5, u


class PolygonSelectorSpline(_SelectorWidget):
    """
    Select a polygon region of an Axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Once drawn individual vertices
    can be moved by clicking and dragging with the left mouse button, or
    removed by clicking the right mouse button.

    In addition, the following modifier keys can be used:

    - Hold *ctrl* and click and drag a vertex to reposition it before the
      polygon has been completed.
    - Hold the *shift* key and click and drag anywhere in the Axes to move
      all vertices.
    - Press the *esc* key to start a new polygon.

    For the selector to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function
        When a polygon is completed or modified after completion,
        the *onselect* function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.

    props : dict, optional
        Properties with which the line is drawn, see `.Line2D` for valid properties.
        Default::

            dict(color='k', linestyle='-', linewidth=2, alpha=0.5)

    handle_props : dict, optional
        Artist properties for the markers drawn at the vertices of the polygon.
        See the marker arguments in `.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``color`` property in *props*.

    grab_range : float, default: 10
        A vertex is selected (to complete the polygon or to move a vertex) if
        the mouse click is within *grab_range* pixels of the vertex.

    draw_bounding_box : bool, optional
        If `True`, a bounding box will be drawn around the polygon selector
        once it is complete. This box can be used to move and resize the
        selector.

    box_handle_props : dict, optional
        Properties to set for the box handles. See the documentation for the
        *handle_props* argument to `RectangleSelector` for more info.

    box_props : dict, optional
        Properties to set for the box. See the documentation for the *props*
        argument to `RectangleSelector` for more info.

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_simple`
    :doc:`/gallery/widgets/polygon_selector_demo`

    Notes
    -----
    If only one point remains after removing points, the selector reverts to an
    incomplete state and you can start drawing a new polygon from the existing
    point.
    """

    def __init__(
        self,
        ax,
        onselect,
        *,
        num_points=100,
        useblit=False,
        curve_props=None,
        props=None,
        handle_props=None,
        grab_range=10,
        draw_bounding_box=False,
        box_handle_props=None,
        box_props=None,
    ):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by PolygonSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.
        state_modifier_keys = dict(
            clear="escape",
            move_vertex="control",
            move_all="shift",
            move="not-applicable",
            square="not-applicable",
            center="not-applicable",
            rotate="not-applicable",
        )
        super().__init__(ax, onselect, useblit=useblit, state_modifier_keys=state_modifier_keys)

        self._xys = [(0, 0)]
        self.num_points = num_points

        if props is None:
            props = dict(color="k", linestyle="-", linewidth=2, alpha=0.5)
        props = {**props, "animated": self.useblit}
        self._selection_artist = line = Line2D([], [], **props)
        self.ax.add_line(line)

        if curve_props is None:
            curve_props = dict(color="k")
        curve_props = {**curve_props, "animated": self.useblit}
        (self.curve,) = self.ax.plot(np.zeros(num_points), np.zeros(num_points), **curve_props)
        # self.curve.set_visible(False)
        self.curve_points = np.array([[np.nan, np.nan]])

        if handle_props is None:
            handle_props = dict(markeredgecolor="k", markerfacecolor=props.get("color", "k"))
        self._handle_props = handle_props
        self._polygon_handles = ToolHandles(self.ax, [], [], useblit=self.useblit, marker_props=self._handle_props)

        self._active_handle_idx = -1
        self.grab_range = grab_range

        self.set_visible(True)
        self._draw_box = draw_bounding_box
        self._box = None

        if box_handle_props is None:
            box_handle_props = {}
        self._box_handle_props = self._handle_props.update(box_handle_props)
        self._box_props = box_props

    def _draw_curve(self):
        if self._selection_completed:
            self.curve_points = spline_interpolate_contour(self._xys[:-1], self.num_points, join_ends=True)
            self.curve.set_visible(True)
            self.curve.set_ydata(self.curve_points[:, 1])
            self.curve.set_xdata(self.curve_points[:, 0])

    @property
    def artists(self):
        """Tuple of the artists of the selector."""
        handles_artists = getattr(self, "_handles_artists", ())
        return (self._selection_artist,) + handles_artists + (self.curve,)

    def _get_bbox(self):
        return self._selection_artist.get_bbox()

    def _add_box(self):
        self._box = RectangleSelector(
            self.ax,
            onselect=lambda *args, **kwargs: None,
            useblit=self.useblit,
            grab_range=self.grab_range,
            handle_props=self._box_handle_props,
            props=self._box_props,
            interactive=True,
        )
        self._box._state_modifier_keys.pop("rotate")
        self._box.connect_event("motion_notify_event", self._scale_polygon)
        self._update_box()
        # Set state that prevents the RectangleSelector from being created
        # by the user
        self._box._allow_creation = False
        self._box._selection_completed = True
        self._draw_polygon()

    def _remove_box(self):
        if self._box is not None:
            self._box.set_visible(False)
            self._box = None

    def _update_box(self):
        # Update selection box extents to the extents of the polygon
        if self._box is not None:
            bbox = self._get_bbox()
            self._box.extents = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
            # Save a copy
            self._old_box_extents = self._box.extents

    def _scale_polygon(self, event):
        """
        Scale the polygon selector points when the bounding box is moved or
        scaled.

        This is set as a callback on the bounding box RectangleSelector.
        """
        if not self._selection_completed:
            return

        if self._old_box_extents == self._box.extents:
            return

        # Create transform from old box to new box
        x1, y1, w1, h1 = self._box._rect_bbox
        old_bbox = self._get_bbox()
        t = (
            transforms.Affine2D()
            .translate(-old_bbox.x0, -old_bbox.y0)
            .scale(1 / old_bbox.width, 1 / old_bbox.height)
            .scale(w1, h1)
            .translate(x1, y1)
        )

        # Update polygon verts.  Must be a list of tuples for consistency.
        new_verts = [(x, y) for x, y in t.transform(np.array(self.verts))]
        self._xys = [*new_verts, new_verts[0]]
        self._draw_polygon()
        self._old_box_extents = self._box.extents

    @property
    def _handles_artists(self):
        return self._polygon_handles.artists

    def _remove_vertex(self, i):
        """Remove vertex with index i."""
        if len(self._xys) > 2 and self._selection_completed and i in (0, len(self._xys) - 1):
            # If selecting the first or final vertex, remove both first and
            # last vertex as they are the same for a closed polygon
            self._xys.pop(0)
            self._xys.pop(-1)
            # Close the polygon again by appending the new first vertex to the
            # end
            self._xys.append(self._xys[0])
        else:
            self._xys.pop(i)
        if len(self._xys) <= 2:
            # If only one point left, return to incomplete state to let user
            # start drawing again
            self._selection_completed = False
            self._remove_box()

    def _press(self, event):
        """Button press event handler."""
        # Check for selection of a tool handle.
        if (self._selection_completed or "move_vertex" in self._state) and len(self._xys) > 0:
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.grab_range:
                self._active_handle_idx = h_idx
        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier).
        self._xys_at_press = self._xys.copy()

    def _release(self, event):
        """Button release event handler."""

        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Add a new vertex
        elif self._selection_completed and "move_all" not in self._state and "move_vertex" not in self._state:
            p3 = self._get_data_coords(event)
            min_dist = np.inf
            index = None
            for i, (p1, p2) in enumerate(zip(self._xys[:-1], self._xys[1:])):
                dist, u = dist_to_line(p1, p2, p3)
                if 1 > u > 0:
                    if min_dist > dist:
                        min_dist = dist
                        index = i
            self._xys.insert(index + 1, p3)
            self._draw_polygon()

        # Complete the polygon.
        elif len(self._xys) > 3 and self._xys[-1] == self._xys[0]:
            self._selection_completed = True
            if self._draw_box and self._box is None:
                self._add_box()
            self._draw_polygon()

        # Place new vertex.
        elif not self._selection_completed and "move_all" not in self._state and "move_vertex" not in self._state:
            self._xys.insert(-1, self._get_data_coords(event))

        if self._selection_completed:
            self.onselect(self.verts)

    def onmove(self, event):
        """Cursor move event handler and validator."""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # _eventpress is None.
        if not self.ignore(event):
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler."""
        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xys[idx] = self._get_data_coords(event)
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._selection_completed:
                self._xys[-1] = self._get_data_coords(event)

        # Move all vertices.
        elif "move_all" in self._state and self._eventpress:
            xdata, ydata = self._get_data_coords(event)
            dx = xdata - self._eventpress.xdata
            dy = ydata - self._eventpress.ydata
            for k in range(len(self._xys)):
                x_at_press, y_at_press = self._xys_at_press[k]
                self._xys[k] = x_at_press + dx, y_at_press + dy

        # Do nothing if completed or waiting for a move.
        elif self._selection_completed or "move_vertex" in self._state or "move_all" in self._state:
            return

        # Position pending vertex.
        else:
            # Calculate distance to the start vertex.
            x0, y0 = self._selection_artist.get_transform().transform(self._xys[0])
            v0_dist = np.hypot(x0 - event.x, y0 - event.y)
            # Lock on to the start vertex if near it and ready to complete.
            if len(self._xys) > 3 and v0_dist < self.grab_range:
                self._xys[-1] = self._xys[0]
            else:
                self._xys[-1] = self._get_data_coords(event)

        self._draw_polygon()

    def _on_key_press(self, event):
        """Key press event handler."""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if not self._selection_completed and ("move_vertex" in self._state or "move_all" in self._state):
            self._xys.pop()
            self._draw_polygon()

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if not self._selection_completed and (
            event.key == self._state_modifier_keys.get("move_vertex")
            or event.key == self._state_modifier_keys.get("move_all")
        ):
            self._xys.append(self._get_data_coords(event))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xys = [self._get_data_coords(event)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)

    def _draw_polygon_without_update(self):
        """Redraw the polygon based on new vertex positions, no update()."""
        xs, ys = zip(*self._xys) if self._xys else ([], [])
        self._selection_artist.set_data(xs, ys)
        self._update_box()
        # Only show one tool handle at the start and end vertex of the polygon
        # if the polygon is completed or the user is locked on to the start
        # vertex.
        if self._selection_completed or (len(self._xys) > 3 and self._xys[-1] == self._xys[0]):
            self._polygon_handles.set_data(xs[:-1], ys[:-1])
        else:
            self._polygon_handles.set_data(xs, ys)

    def _draw_polygon(self):
        """Redraw the polygon based on the new vertex positions."""
        self._draw_polygon_without_update()
        self._draw_curve()
        self.update()

    @property
    def verts(self):
        """The polygon vertices, as a list of ``(x, y)`` pairs."""
        return self._xys[:-1]

    @verts.setter
    def verts(self, xys):
        """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
        self._xys = [*xys, xys[0]]
        self._selection_completed = True
        self.set_visible(True)
        if self._draw_box and self._box is None:
            self._add_box()
        self._draw_polygon()

    def _clear_without_update(self):
        self._selection_completed = False
        self._xys = [(0, 0)]
        self._draw_polygon_without_update()
