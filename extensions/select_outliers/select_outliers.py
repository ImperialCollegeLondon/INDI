import os
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from extensions.extension_base import ExtensionBase
from extensions.extensions import get_window
from extensions.read_data.read_and_pre_process_data import create_2d_montage_from_database


def manual_image_removal(
    data: pd.DataFrame,
    slices: NDArray,
    segmentation: Dict,
    mask: NDArray,
    settings: Dict,
    stage: str,
    info: Dict,
) -> List[int]:
    """
    Manual removal of images. A matplotlib window will open, and we can select images to be removed.

    Parameters
    ----------
    data: dataframe with all the dwi data
    slices: array with slice positions
    segmentation: dict with epicardium and endocardium masks
    mask: array with the mask of the heart
    settings: dict with useful info
    stage: pre or post segmentation
    info: dict with useful info

    Returns
    -------
    dataframe with all the data
    dataframe with data without the rejected images
    info: dict
    array with indices of rejected images in the original dataframe

    """
    # # max relative signal in the images
    # if settings["sequence_type"] == "steam":
    #     max_rel_signal = 0.75
    # elif settings["sequence_type"] == "se":
    #     max_rel_signal = 0.5
    # else:
    #     max_rel_signal = 1.0

    # now we need to collect all the index positions of the rejected frames
    stored_indices_all_slices = []
    # loop over the slices
    for slice_idx in slices:
        # dataframe with current slice
        c_df = data.loc[data["slice_integer"] == slice_idx].copy()
        c_df = c_df.reset_index()
        # initiate maximum number of images found for each b-val and dir combination
        max_number_of_images = 0

        # drop any images already marked to be removed
        c_df = c_df.loc[c_df["to_be_removed"] == False]

        # convert list of directions to a tuple
        c_df["diffusion_direction_original"] = c_df["diffusion_direction_original"].apply(tuple)

        # get unique b-values
        b_vals = c_df.b_value_original.unique()
        b_vals.sort()

        # initiate the stacks for the images and the highlight masks
        c_img_stack = {}
        c_img_indices = {}
        c_img_stack_series_description = {}

        # loop over sorted b-values
        for b_val in b_vals:
            # dataframe with current slice and current b-value
            c_df_b = c_df.loc[c_df["b_value_original"] == b_val].copy()

            # unique directions
            dirs = c_df_b["diffusion_direction_original"].unique()
            dirs.sort()

            # loop over directions
            for dir_idx, dir in enumerate(dirs):
                # dataframe with current slice, current b-value, and current direction images
                c_df_b_d = c_df_b.loc[c_df_b["diffusion_direction_original"] == dir].copy()

                # for each b_val and each dir collect all images
                c_img_stack[b_val, dir] = np.stack(c_df_b_d.image.values, axis=0)
                c_img_indices[b_val, dir] = c_df_b_d.index.values
                c_img_stack_series_description[b_val, dir] = c_df_b_d.series_description.values

                # record n_images if bigger than the values stored
                n_images = c_df_b_d.shape[0]
                if n_images > max_number_of_images:
                    max_number_of_images = n_images

        # plot all stored images,
        # y-axis b-value and direction combos
        # x-axis repetitions
        store_selected_images = []
        # retina screen resolution
        my_dpi = 192
        rows = len(c_img_stack)
        cols = max_number_of_images
        # check if FOV is vertical, if so rotate text
        if info["img_size"][0] < info["img_size"][1]:
            text_rotation = 0
        else:
            text_rotation = 90
        if stage == "pre":
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
                dpi=my_dpi,
                num="Slice " + str(slice_idx),
                squeeze=False,
            )
        elif stage == "post":
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(settings["screen_size"][0] / my_dpi, (settings["screen_size"][1] - 52) / my_dpi),
                dpi=my_dpi,
                num="Slice " + str(slice_idx),
                squeeze=False,
            )
        for idx, key in enumerate(c_img_stack):
            cc_img_stack = c_img_stack[key]
            for idx2, img in enumerate(cc_img_stack):
                vmin, vmax = get_window(img, mask)
                axs[idx, idx2].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                if segmentation:
                    axs[idx, idx2].scatter(
                        segmentation[slice_idx]["epicardium"][:, 0],
                        segmentation[slice_idx]["epicardium"][:, 1],
                        marker=".",
                        s=0.1,
                        color="tab:red",
                        alpha=0.7,
                    )
                    if segmentation[slice_idx]["endocardium"].size != 0:
                        axs[idx, idx2].scatter(
                            segmentation[slice_idx]["endocardium"][:, 0],
                            segmentation[slice_idx]["endocardium"][:, 1],
                            marker=".",
                            s=0.1,
                            color="tab:red",
                            alpha=0.7,
                        )
                if stage == "pre":
                    if not settings["print_series_description"]:
                        axs[idx, idx2].text(
                            2,
                            2,
                            str(int(key[0])),
                            fontsize=3,
                            color="tab:orange",
                            horizontalalignment="left",
                            verticalalignment="top",
                            bbox=dict(facecolor="black", pad=0, edgecolor="none"),
                        )
                    elif settings["print_series_description"]:
                        axs[idx, idx2].text(
                            2,
                            2,
                            c_img_stack_series_description[key][idx2],
                            fontsize=3,
                            color="tab:orange",
                            horizontalalignment="left",
                            verticalalignment="top",
                            bbox=dict(facecolor="black", pad=0, edgecolor="none"),
                            rotation=text_rotation,
                        )
                elif stage == "post":
                    axs[idx, idx2].text(
                        2,
                        2,
                        str(int(key[0])),
                        fontsize=3,
                        color="tab:orange",
                        horizontalalignment="left",
                        verticalalignment="top",
                        bbox=dict(facecolor="black", pad=0, edgecolor="none"),
                    )

                axs[idx, idx2].set_xticks([])
                axs[idx, idx2].set_yticks([])
                axs[idx, idx2].values = *key, idx2

        # Setting the values for all axes.
        plt.setp(axs, xticks=[], yticks=[])
        plt.tight_layout(pad=0.1)
        # remove axes with no image
        [p.set_axis_off() for p in [i for i in axs.flatten() if len(i.images) < 1]]

        def onclick_select(event):
            """function to record the axes of the selected images in subplots"""
            if event.inaxes is not None:
                if event.inaxes in store_selected_images:
                    store_selected_images.remove(event.inaxes)
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(1)
                else:
                    event.inaxes.set_alpha(0.1)
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(1)
                    store_selected_images.append(event.inaxes)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick_select)
        fig.canvas.draw_idle()
        plt.show(block=True)

        # store the indices of the rejected images:
        # - indices for all slices together
        # - indices within each slice
        for ax in store_selected_images:
            c_bval = ax.values[0]
            c_dir = ax.values[1]
            c_idx = ax.values[2]

            # locate item in dataframe containing all images for this slice
            c_table = c_df[(c_df["diffusion_direction_original"] == c_dir)]
            c_table = c_table[(c_table["b_value_original"] == c_bval)]
            c_filename = c_table.iloc[c_idx]["file_name"]

            # store the index of the rejected image
            stored_indices_all_slices.append(c_table[(c_table["file_name"] == c_filename)]["index"].iloc[0])

    # the indices of the rejected frames
    rejected_indices = stored_indices_all_slices

    return rejected_indices


class SelectOutliers(ExtensionBase):
    def run(self):
        data = self.context["data"]
        info = self.context["info"]
        self.logger.info("Running Select Outliers")

        # check if there is already a column marking the images to be removed
        if "to_be_removed" not in data:
            data["to_be_removed"] = False

        # check if the info dictionary has the rejected_indices and n_images_rejected keys
        if "rejected_indices" not in info:
            info["rejected_indices"] = []
        if "n_images_rejected" not in info:
            info["n_images_rejected"] = 0

        session_file = pathlib.Path(self.settings["session"], "image_manual_removal_" + self.context["stage"] + ".zip")
        # check if this manual removal has been previously done
        if session_file.exists():
            self.logger.info("Manual image removal already done, loading information.")
            data_to_load = pd.read_pickle(session_file)
            # check if the column acquisition_date_time matches
            if not data_to_load["acquisition_date_time"].equals(data["acquisition_date_time"]):
                self.logger.error("Data in the session file does not match the current data.")
                self.logger.error("Please remove the session file and run the manual image removal again.")
                sys.exit()
            else:
                # toggle to True the indices to be removed
                for idx in data_to_load.loc[data_to_load["to_be_removed"] == True].index:
                    data.loc[idx, "to_be_removed"] = True
                # add attributes stored in data_to_load to data
                data.attrs["rejected_images"] = data_to_load.attrs["rejected_images"]

            rejected_indices = data.attrs["rejected_images"]
            info["n_images_rejected"] += len(rejected_indices)
            info["rejected_indices"].extend(rejected_indices)

        else:
            # Manual image removal
            rejected_indices = manual_image_removal(
                data,
                self.context["slices"],
                self.context["segmentation"],
                self.context["mask"],
                self.settings,
                self.context["stage"],
                info,
            )
            self.logger.info("Manual image removal done.")

            # toggle to True the indices to be removed
            for idx in rejected_indices:
                data.loc[idx, "to_be_removed"] = True
            data.attrs["rejected_images"] = rejected_indices
            # save a table with the acquisition_date_time and to_be_removed columns
            data_to_be_saved = data[["acquisition_date_time", "to_be_removed"]]
            data_to_be_saved.to_pickle(session_file, compression={"method": "zip", "compresslevel": 9})

            info["n_images_rejected"] += len(rejected_indices)
            info["rejected_indices"].extend(rejected_indices)

        self.logger.info(
            "Number of images rejected after " + self.context["stage"] + ": " + str(info["n_images_rejected"])
        )

        if self.context["stage"] == "post":
            # plot all remaining DWIs also add the segmentation curves
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "diffusion_direction_original",
                self.settings,
                info,
                self.context["slices"],
                "dwis_accepted",
                os.path.join(self.settings["results"], "results_b"),
                info["rejected_indices"],
                self.context["segmentation"],
                False,
            )
            if self.settings["complex_data"]:
                create_2d_montage_from_database(
                    data,
                    "b_value_original",
                    "diffusion_direction_original",
                    self.settings,
                    info,
                    self.context["slices"],
                    "dwis_accepted_phase",
                    self.settings["debug_folder"],
                    info["rejected_indices"],
                    self.context["segmentation"],
                    False,
                    "image_phase",
                )
        self.context["data"] = data
        self.context["info"] = info