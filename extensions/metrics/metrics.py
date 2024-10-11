from extensions.extension_base import ExtensionBase
from extensions.extensions import (
    get_cardiac_coordinates_short_axis,
    get_coordinates_tissue_block,
    get_ha_line_profiles,
    get_heart_segments,
)
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
        if self.settings["tissue_block"]:
            local_cardiac_coordinates, lv_centres, phi_matrix = get_coordinates_tissue_block(
                mask_3c, segmentation, slices, info["n_slices"], self.settings, dti, average_images, info
            )
        else:
            local_cardiac_coordinates, lv_centres, phi_matrix = get_cardiac_coordinates_short_axis(
                mask_3c, segmentation, slices, info["n_slices"], self.settings, dti, average_images, info
            )
            if self.settings["RV-segmented"]:
                local_cardiac_coordinates_rv, rv_centres, phi_matrix_rv = get_cardiac_coordinates_short_axis(
                    mask_3c, segmentation, slices, info["n_slices"], self.settings, dti, average_images, info, "RV"
                )

        # =========================================================
        # Segment heart
        # =========================================================
        dti["lv_sectors"] = get_heart_segments(
            segmentation, phi_matrix, mask_3c, lv_centres, slices, self.logger, "LV"
        )
        if self.settings["RV-segmented"]:
            dti["rv_sectors"] = get_heart_segments(
                segmentation, phi_matrix_rv, mask_3c, rv_centres, slices, self.logger, "RV"
            )

        # =========================================================
        # Get dti["ha"] and dti["e2a"] maps
        # =========================================================
        dti["ha"], dti["ta"], dti["e2a"] = get_tensor_orientation_maps(
            slices, mask_3c, local_cardiac_coordinates, dti, self.settings, self.logger, "LV"
        )
        if self.settings["RV-segmented"]:
            dti["ha_rv"], dti["ta_rv"], dti["e2a_rv"] = get_tensor_orientation_maps(
                slices, mask_3c, local_cardiac_coordinates_rv, dti, self.settings, self.logger, "RV"
            )

        # =========================================================
        # Get HA line profiles
        # =========================================================
        if not self.settings["tissue_block"]:
            dti["ha_line_profiles"], dti["wall_thickness"] = get_ha_line_profiles(
                dti["ha"], lv_centres, slices, mask_3c, segmentation, self.settings, info
            )

        # =========================================================
        # Combine metric maps for orientation measures
        # =========================================================
        maps_lv = ["ha", "ta", "e2a"]
        maps_rv = ["ha_rv", "ta_rv", "e2a_rv"]
        for idx in range(len(maps_lv)):
            for slice_ in slices:
                dti[maps_lv[idx]][slice_][mask_3c[slice_] == 2] = dti[maps_rv[idx]][slice_][mask_3c[slice_] == 2]
            del dti[maps_rv[idx]]

        self.context["dti"] = dti
        self.context["info"] = info


# class MetricsRV(ExtensionBase):
#     def run(self):
#         dti = self.context["dti"]
#         slices = self.context["slices"]
#         info = self.context["info"]
#         average_images = self.context["average_images"]
#         mask_3c = self.context["mask_whole_heart"]
#         mask_rv = self.context["mask_rv"]
#         segmentation = self.context["segmentation"]
#
#         self.logger.info("Calculating the RV maps")
#         # =========================================================
#         # Get Eigensystems
#         # =========================================================
#         dti, info = get_eigensystem(dti, slices, info, average_images, self.settings, mask_3c, self.logger, "RV")
#
#         # =========================================================
#         # Get dti["fa"] and dti["md"] maps
#         # =========================================================
#         dti["md-rv"], dti["fa-rv"], info = get_fa_md(dti["eigenvalues"], info, mask_3c, slices, self.logger)
#
#         # =========================================================
#         # Get cardiac coordinates
#         # =========================================================
#         local_cardiac_coordinates, rv_centres, phi_matrix = get_cardiac_coordinates_short_axis(
#             mask_3c, segmentation, slices, info["n_slices"], self.settings, dti, average_images, info, "RV"
#         )
#
#         # =========================================================
#         # Segment heart
#         # =========================================================
#         dti["rv_sectors"] = get_lv_segments(segmentation, phi_matrix, mask_3c, rv_centres, slices, self.logger)
#
#         # =========================================================
#         # Get dti["ha"] and dti["e2a"] maps
#         # =========================================================
#         dti["ha-rv"], dti["ta-rv"], dti["e2a-rv"], info = get_tensor_orientation_maps(
#             slices, mask_3c, local_cardiac_coordinates, dti, self.settings, info, self.logger
#         )
#
#         # =========================================================
#         # Get HA line profiles
#         # =========================================================
#         # dti["ha_line_profiles-rv"], dti["wall_thickness-rv"] = get_ha_line_profiles(
#         #     dti["ha-rv"], rv_centres, slices, mask_3c, segmentation, self.settings, info, "RV"
#         # )
#
#         # =========================================================
#         # Remove the LV mask from the RV mask
#         # =========================================================
#         maps = ["fa-rv", "md-rv", "ha-rv", "ta-rv", "e2a-rv"]
#         for map_name in maps:
#             dti[map_name][mask_rv == 0] = np.nan
#
#         self.context["dti"] = dti
#         self.context["info"] = info
