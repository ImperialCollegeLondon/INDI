import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import zscore

from indi.extensions.extensions import get_window
from indi.extensions.read_data.read_and_pre_process_data import create_2d_montage_from_database


def manual_image_removal(
    data: pd.DataFrame,
    slices: NDArray,
    segmentation: dict,
    mask: NDArray,
    settings: dict,
    stage: str,
    info: dict,
    prelim_residuals: dict = {},
) -> tuple[pd.DataFrame, pd.DataFrame, dict, NDArray]:
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

        # find outliers from the residuals
        if prelim_residuals:
            zscores = zscore(prelim_residuals[slice_idx], axis=0, nan_policy="omit")
            outliers_idx = np.where(np.abs(zscores) > 3)[0]
            outliers_idx += 1

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

                    # if an outlier than change the color to red otherwise keep it yellow
                    if prelim_residuals and c_img_indices[key][idx2] in outliers_idx:
                        line_colour = "tab:red"
                    else:
                        line_colour = "tab:green"

                    axs[idx, idx2].scatter(
                        segmentation[slice_idx]["epicardium"][:, 0],
                        segmentation[slice_idx]["epicardium"][:, 1],
                        marker=".",
                        s=0.1,
                        color=line_colour,
                        alpha=0.7,
                    )
                    if segmentation[slice_idx]["endocardium"].size != 0:
                        axs[idx, idx2].scatter(
                            segmentation[slice_idx]["endocardium"][:, 0],
                            segmentation[slice_idx]["endocardium"][:, 1],
                            marker=".",
                            s=0.1,
                            color=line_colour,
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

        # set the background colour of the figure
        fig.patch.set_facecolor("0.05")

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

            # locate item in dataframe containing all images for this slice with the current b-value and direction
            c_table = c_df[(c_df["diffusion_direction_original"] == c_dir)]
            c_table = c_table[(c_table["b_value_original"] == c_bval)]

            # what is the index of the image in the dataframe with all slices?
            # Calculate the index of the start of the slice
            slice_delta_idx = data[data["slice_integer"] == slice_idx].index[0]

            # store the index of the rejected image
            stored_indices_all_slices.append(c_table.index[c_idx] + slice_delta_idx)

    # the indices of the rejected frames
    rejected_indices = stored_indices_all_slices

    return rejected_indices


def select_outliers(
    data: pd.DataFrame,
    slices: NDArray,
    registration_image_data: dict,
    settings: dict,
    info: dict,
    logger: logging,
    stage: str,
    segmentation: dict = {},
    mask: NDArray = np.array([]),
    prelim_residuals: dict = {},
) -> tuple[pd.DataFrame, dict, NDArray]:
    """
    Remove Outliers: remove outliers, and display all DWIs in a montage

    Parameters
    ----------
    data: dataframe with images and diffusion info
    slices: array with slice integers
    registration_image_data: dict with registration images and QC data
    settings
    info
    logger
    stage: string with stage pre or post segmentation
    mask: array with heart masks

    Returns
    -------
    data, info, slices

    """
    # check if the info dictionary has the rejected_indices and n_images_rejected keys
    if "rejected_indices" not in info:
        info["rejected_indices"] = []
    if "n_images_rejected" not in info:
        info["n_images_rejected"] = 0

    # ========================================================={
    # manual removal of images
    # =========================================================
    if settings["remove_outliers_manually"]:
        session_file = os.path.join(settings["session"], "image_manual_removal_" + stage + ".zip")
        # check if this manual removal has been previously done
        if os.path.exists(session_file):
            logger.info("Manual image removal already done, loading information.")
            # load initial database with rejected images included
            data_to_load = pd.read_pickle(session_file)
            # check if the column acquisition_date_time matches
            if not data_to_load["acquisition_date_time"].equals(data["acquisition_date_time"]):
                logger.error("Data in the session file does not match the current data.")
                logger.error("Please remove the session file and run the manual image removal again.")
                raise ValueError(
                    "Data in the session file does not match the current data. Please remove the session file and run the manual image removal again."
                )
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
            # logger.info("Starting manual image removal...")
            rejected_indices = manual_image_removal(
                data,
                slices,
                segmentation,
                mask,
                settings,
                stage,
                info,
                prelim_residuals,
            )
            logger.info("Manual image removal done.")

            # toggle to True the indices to be removed
            for idx in rejected_indices:
                data.loc[idx, "to_be_removed"] = True
            data.attrs["rejected_images"] = rejected_indices
            # save a table with the acquisition_date_time and to_be_removed columns
            data_to_be_saved = data[["acquisition_date_time", "to_be_removed"]]
            data_to_be_saved.to_pickle(session_file, compression={"method": "zip", "compresslevel": 9})

            info["n_images_rejected"] += len(rejected_indices)
            info["rejected_indices"].extend(rejected_indices)

        logger.info("Number of images rejected after " + stage + ": " + str(info["n_images_rejected"]))

        # simplify this list to be able to save it in the yaml file
        info["rejected_indices"] = [int(i) for i in info["rejected_indices"]]

        if stage == "post":
            # remove the rejected images from the registration_image_data
            for slice_idx in slices:
                c_indices = data[data.slice_integer == slice_idx].index.values
                all_rejected_indices = info["rejected_indices"]
                all_rejected_indices.sort(reverse=True)

                to_remove_idxs = c_indices[np.isin(c_indices, all_rejected_indices)]
                to_remove_idxs = [np.where(c_indices == x)[0][0] for x in to_remove_idxs]

                registration_image_data[slice_idx]["img_post_reg"] = np.delete(
                    registration_image_data[slice_idx]["img_post_reg"], to_remove_idxs, axis=0
                )
                registration_image_data[slice_idx]["img_pre_reg"] = np.delete(
                    registration_image_data[slice_idx]["img_pre_reg"], to_remove_idxs, axis=0
                )
                registration_image_data[slice_idx]["deformation_field"]["field"] = np.delete(
                    registration_image_data[slice_idx]["deformation_field"]["field"], to_remove_idxs, axis=0
                )
                registration_image_data[slice_idx]["deformation_field"]["grid"] = np.delete(
                    registration_image_data[slice_idx]["deformation_field"]["grid"], to_remove_idxs, axis=0
                )

        if stage == "post":
            highlight_list = data.index[data["to_be_removed"] == True].tolist()

            # plot all remaining DWIs also add the segmentation curves
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "diffusion_direction_original",
                settings,
                info,
                slices,
                "dwis_accepted",
                os.path.join(settings["results"], "results_b"),
                highlight_list,
                segmentation,
                False,
            )
            if settings["complex_data"]:
                create_2d_montage_from_database(
                    data,
                    "b_value_original",
                    "diffusion_direction_original",
                    settings,
                    info,
                    slices,
                    "dwis_accepted_phase",
                    settings["debug_folder"],
                    highlight_list,
                    segmentation,
                    False,
                    "image_phase",
                )

    else:
        # no image removal to be done
        logger.info("No image removal to be done.")

    # =========================================================
    # remove outliers with AI
    # =========================================================
    if settings["remove_outliers_with_ai"]:
        pass
        # data_new, rows_to_drop = remove_outliers_ai(data, info, settings, slices, logger, threshold=0.25)
        # logger.info("Removed outliers with AI")
        # if settings["debug"]:
        #     create_2d_montage_from_database(
        #         data,
        #         "b_value_original",
        #         "direction_original",
        #         info,
        #         settings,
        #         slices,
        #         "dwis_outliers_with_ai",
        #         settings["debug_folder"],
        #         rows_to_drop,
        #     )
        #     logger.info("2d montage of DWIs after outlier removal with AI")
        #
        # # copy dataframe with removed images to the data variable
        # data = data_new.reset_index(drop=True)
        # del data_new
        # # update number of dicom files
        # info["n_images"] = data.shape[0]
        # logger.debug("DWIs after outlier removal with AI: " + str(info["n_images"]))

    return data, info, slices
