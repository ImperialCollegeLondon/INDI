from extensions.extension_base import ExtensionBase
from extensions.extensions import get_cardiac_coordinates_short_axis, get_ha_line_profiles, get_lv_segments
from extensions.get_eigensystem import get_eigensystem
from extensions.get_fa_md import get_fa_md
from extensions.get_tensor_orientation_maps import get_tensor_orientation_maps


class Metrics(ExtensionBase):
    def run(self):
        dti = self.context["dti"]
        slices = self.context["slices"]
        info = self.context["info"]
        average_images = self.context["average_images"]
        mask_3c = self.context["mask_3c"]
        segmentation = self.context["segmentation"]
        # =========================================================
        # Get Eigensystems
        # =========================================================
        dti, info = get_eigensystem(
            dti,
            slices,
            info,
            average_images,
            self.settings,
            mask_3c,
            self.logger,
        )

        # =========================================================
        # Get dti["fa"] and dti["md"] maps
        # =========================================================
        dti["md"], dti["fa"], info = get_fa_md(dti["eigenvalues"], info, mask_3c, slices, self.logger)

        # =========================================================
        # Get cardiac coordinates
        # =========================================================
        local_cardiac_coordinates, lv_centres, phi_matrix = get_cardiac_coordinates_short_axis(
            mask_3c, segmentation, slices, info["n_slices"], self.settings, dti, average_images, info
        )

        # =========================================================
        # Segment heart
        # =========================================================
        dti["lv_sectors"] = get_lv_segments(segmentation, phi_matrix, mask_3c, lv_centres, slices, self.logger)

        # =========================================================
        # Get dti["ha"] and dti["e2a"] maps
        # =========================================================
        dti["ha"], dti["ta"], dti["e2a"], info = get_tensor_orientation_maps(
            slices, mask_3c, local_cardiac_coordinates, dti, self.settings, info, self.logger
        )

        # =========================================================
        # Get HA line profiles
        # =========================================================
        dti["ha_line_profiles"], dti["wall_thickness"] = get_ha_line_profiles(
            dti["ha"], lv_centres, slices, mask_3c, segmentation, self.settings, info
        )

        self.context["dti"] = dti
        self.context["info"] = info
