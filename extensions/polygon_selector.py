import numpy as np
from matplotlib.widgets import PolygonSelector


class PolygonSelectorBetter(PolygonSelector):
    def _release(self, event):
        """Button release event handler."""

        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Add a new vertex
        elif self._selection_completed and event.inaxes is self.ax:
            nodes = np.asarray(self._xys)
            dist_2 = np.sum((nodes - self._get_data_coords(event)) ** 2, axis=1)
            index = np.argmin(dist_2)
            self._xys.insert(index, self._get_data_coords(event))
            self._draw_polygon()

        # Complete the polygon.
        elif len(self._xys) > 3 and self._xys[-1] == self._xys[0]:
            self._selection_completed = True
            if self._draw_box and self._box is None:
                self._add_box()

        # Place new vertex.
        elif not self._selection_completed and "move_all" not in self._state and "move_vertex" not in self._state:
            self._xys.insert(-1, self._get_data_coords(event))

        if self._selection_completed:
            self.onselect(self.verts)
