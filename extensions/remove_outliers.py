import ast
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from extensions.dwis_classifier import dwis_classifier
from extensions.extensions import crop_pad_rotate_array
from extensions.read_and_pre_process_data import create_2d_montage_from_database


def remove_outliers_ai(
    data: pd.DataFrame,
    info: dict,
    settings: dict,
    slices: NDArray,
    logger: logging.Logger,
    threshold: float = 0.3,
) -> [pd.DataFrame, NDArray]:
    """Remove the bad frames from the dataframe using the AI classifier

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with diffusion info
    info : dict
        useful info
    settings : dict
        settings
    slices : NDArray
        array with slice positions as strings
    logger : logging.Logger
        logger
    threshold : float, optional
        threshold value to consider bad in [0, 1], by default 0.3

    Returns
    -------
    Tuple[pd.DataFrame, NDArray]
        [dataframe without bad frames, array with bad frames positions]
    """

    # gather images from dataframe
    dwis = np.empty([info["n_files"], info["img_size"][0], info["img_size"][1]])
    for i in range(info["n_files"]):
        # moving image
        dwis[i] = data.loc[i, "image"]

    # make sure image stack has the correct dimensions
    dwis = crop_pad_rotate_array(dwis, (info["n_files"], 256, 96), True)

    # use the AI classifier to determine which ones are bad
    frame_labels = dwis_classifier(dwis, threshold)

    # drop frames frames labeled as bad (0)
    rows_to_drop = np.where(frame_labels < 1)[0]
    data_new = data.drop(index=list(rows_to_drop))

    logger.debug("Number of images removed by AI: " + str(len(rows_to_drop)))

    return data_new, rows_to_drop


def manual_image_removal(
    data: pd.DataFrame, info: dict, settings: dict, slices: NDArray, logger: logging.Logger
) -> [pd.DataFrame, pd.DataFrame, dict, NDArray]:
    """
    Manual removal of images. A matplotlib window will open, and we can select images to be removed.

    Parameters
    ----------
    data: dataframe with all the dwi data
    info: dict
    settings: dict
    slices: array with slice positions
    logger: logger for console and file

    Returns
    -------
    dataframe with all the data
    dataframe with data without the rejected images
    info: dict
    array with indices of rejected images in the original dataframe

    """
    # this is an empty dataframe to store all rejected data
    rejected_images_data = pd.DataFrame(columns=list(data.columns))
    # store also original dataframe before any image removal
    data_original = data.copy()

    # loop over the slices
    for slice_idx in slices:
        # dataframe with current slice
        c_df = data.loc[data["slice_integer"] == slice_idx].copy()

        # initiate maximum number of images found for each b-val and dir combination
        max_number_of_images = 0

        # convert list of directions to a tuple
        c_df["direction_original"] = c_df["direction_original"].apply(tuple)

        # get unique b-values
        b_vals = c_df.b_value_original.unique()
        b_vals.sort()

        # initiate the stacks for the images and the highlight masks
        c_img_stack = {}
        c_img_indices = {}

        # loop over sorted b-values
        for b_val in b_vals:
            # dataframe with current slice and current b-value
            c_df_b = c_df.loc[c_df["b_value_original"] == b_val].copy()

            # unique directions
            dirs = c_df_b["direction_original"].unique()
            dirs.sort()

            # loop over directions
            for dir_idx, dir in enumerate(dirs):
                # dataframe with current slice, current b-value, and current direction images
                c_df_b_d = c_df_b.loc[c_df_b["direction_original"] == dir].copy()

                # for each b_val and each dir collect all images
                c_img_stack[b_val, dir] = np.stack(c_df_b_d.image.values, axis=0)
                c_img_indices[b_val, dir] = c_df_b_d.index.values

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
        fig, axs = plt.subplots(rows, cols, figsize=(1500 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        for idx, key in enumerate(c_img_stack):
            cc_img_stack = c_img_stack[key]
            for idx2, img in enumerate(cc_img_stack):
                axs[idx, idx2].imshow(img, cmap="gray")
                axs[idx, idx2].text(
                    5,
                    5,
                    str(key[0]),
                    fontsize=3,
                    color="tab:orange",
                    horizontalalignment="left",
                    verticalalignment="top",
                )

                # axs[idx, idx2].set_title(
                #     str(key[0]) + "_" + str(idx),
                #     fontsize=2,
                # )

                axs[idx, idx2].set_xticks([])
                axs[idx, idx2].set_yticks([])

                axs[idx, idx2].name = str(key + (idx2,))
        # Setting the values for all axes.
        plt.setp(axs, xticks=[], yticks=[])
        plt.tight_layout(pad=0.1)
        # remove axes with no image
        [p.set_axis_off() for p in [i for i in axs.flatten() if len(i.images) < 1]]

        def onclick_select(event):
            """function to record the axes of the selected images in subplots"""
            if event.inaxes is not None:
                print(event.inaxes.name)
                if ast.literal_eval(event.inaxes.name) in store_selected_images:
                    store_selected_images.remove(ast.literal_eval(event.inaxes.name))
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(1)
                else:
                    event.inaxes.set_alpha(0.1)
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(1)
                    store_selected_images.append(ast.literal_eval(event.inaxes.name))
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick_select)
        plt.show()

        # now we have to loop over all selected images
        # remove them from the dataframe data
        for item in store_selected_images:
            c_bval = item[0]
            c_dir = item[1]
            c_idx = item[2]

            # locate item in dataframe containing all images for this slice
            c_table = c_df[(c_df["direction_original"] == c_dir)]
            c_table = c_table[(c_table["b_value_original"] == c_bval)]
            c_filename = c_table.iloc[c_idx]["file_name"]

            # # not a good idea to do this
            # # move file to rejected folder
            # origin = os.path.join(settings["dicom_folder"], c_filename)
            # destination = os.path.join(settings["dicom_folder"], "rejected_images", c_filename)
            # shutil.move(origin, destination)

            # remove row from the dataframe
            data.drop(data[data.file_name == c_filename].index, inplace=True)
            rejected_images_data = pd.concat([rejected_images_data, c_table.iloc[c_idx].to_frame().T])

    # store original dataframe with all images and also as an attribute
    # the indices of the rejected frames
    # this dataframe will be used to plot all dwis with red labels on the rejected ones
    rejected_indices = rejected_images_data.index.values
    data_original.attrs["rejected_images"] = rejected_indices
    save_path = os.path.join(settings["session"], "image_manual_removal.zip")
    data_original.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})

    # display some info
    info["n_files"] = len(data)
    data.reset_index(drop=True, inplace=True)
    info["n_images_rejected"] = len(rejected_indices)
    logger.info("Number of images rejected: " + str(info["n_images_rejected"]))

    return data_original, data, info, rejected_indices


def remove_outliers(
    data: pd.DataFrame, slices: NDArray, settings: dict, info: dict, logger: logging
) -> [pd.DataFrame, dict, NDArray]:
    """
    Remove Outliers: remove outliers, and display all DWIs in a montage

    Parameters
    ----------
    data: dataframe with images and diffusion info
    slices: array with slice integers
    settings
    info
    logger

    Returns
    -------
    data, info, slices

    """

    # ========================================================={
    # manual removal of images
    # =========================================================
    if settings["remove_outliers_manually"]:
        # check if this manual removal has been previously done
        if os.path.exists(os.path.join(settings["session"], "image_manual_removal.zip")):
            logger.info("Manual image removal already done, loading information.")
            # load initial database with rejected images included
            data_original = pd.read_pickle(os.path.join(settings["session"], "image_manual_removal.zip"))
            rejected_indices = data_original.attrs["rejected_images"]
            info["n_images_rejected"] = len(rejected_indices)
            logger.debug("Number of images rejected: " + str(info["n_images_rejected"]))
        else:
            # Manual image removal
            logger.info("Starting manual image removal...")
            # if not os.path.exists(os.path.join(settings["dicom_folder"], "rejected_images")):
            #     os.makedirs(os.path.join(settings["dicom_folder"], "rejected_images"))
            data_original, data, info, rejected_indices = manual_image_removal(data, info, settings, slices, logger)
            logger.info("Manual image removal done.")
    else:
        # no image removal to be done
        logger.info("No image removal to be done.")
        data_original = data.copy()
        rejected_indices = []

    # =========================================================
    # remove outliers with AI
    # =========================================================
    if settings["remove_outliers_with_ai"]:
        data_new, rows_to_drop = remove_outliers_ai(data, info, settings, slices, logger, threshold=0.25)
        logger.info("Removed outliers with AI")
        if settings["debug"]:
            create_2d_montage_from_database(
                data,
                "b_value_original",
                "direction_original",
                info,
                settings,
                slices,
                "dwis_outliers_with_ai",
                settings["debug_folder"],
                rows_to_drop,
            )
            logger.info("2d montage of DWIs after outlier removal with AI")

        # copy dataframe with removed images to the data variable
        data = data_new.reset_index(drop=True)
        del data_new
        # update number of dicom files
        info["n_files"] = data.shape[0]
        logger.debug("DWIs after outlier removal with AI: " + str(info["n_files"]))

    # =========================================================
    # display all DWIs in a montage
    # =========================================================
    create_2d_montage_from_database(
        data_original,
        "b_value_original",
        "direction_original",
        info,
        settings,
        slices,
        "dwis_after_image_rejection",
        os.path.join(settings["results"], "results_b"),
        rejected_indices,
    )

    return data, info, slices