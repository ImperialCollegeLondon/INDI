import csv
import logging
import os
import pickle
import subprocess

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
import skimage.util
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from indi.extensions.export_vectors_tensors_vtk import export_vectors_tensors_vtk
from indi.extensions.extensions import get_window


def export_results(
    data: pd.DataFrame,
    dti: dict,
    info: dict,
    settings: dict,
    mask_3c: NDArray,
    slices: NDArray,
    average_images: NDArray,
    segmentation: dict,
    colormaps: dict,
    logger: logging.Logger,
):
    """Export results to disk (VTK, PNGs, HDF5, pickle, CSV, and ZIP).

    Args:
        data (pd.DataFrame): Diffusion images and metadata.
        dti (dict): DTI maps to export.
        info (dict): Metadata dictionary.
        settings (dict): Output and debug settings.
        mask_3c (NDArray): Heart segmentation mask.
        slices (NDArray): Slice indices.
        average_images (NDArray): Normalised average images.
        segmentation (dict): LV segmentation, contours, and insertion points.
        colormaps (dict): DTI-specific colour maps.
        logger (logging.Logger): Logger for export progress.
    """

    logger.info("Exporting results to disk...")

    # create a string with the folder names
    folder_id = settings["work_folder"]
    folder_id = os.path.normpath(folder_id)
    folders = folder_id.split(os.sep)
    folder_id = folders[-4:-1]
    folder_id = "_".join(folder_id)

    # plot eigenvectors and tensor in VTK format
    export_vectors_tensors_vtk(dti, info, settings, mask_3c, average_images)

    # plot results maps
    plot_results_maps(slices, mask_3c, average_images, dti, segmentation, colormaps, settings, folder_id)

    # save results to h5 file
    export_to_hdf5(dti, mask_3c, settings)

    # save results summary to a csv table
    export_summary_table(dti, settings, slices)

    # save to disk dti dictionary
    dti["info"] = info
    with open(os.path.join(settings["results"], "data", "DTI_data.dat"), "wb") as handle:
        pickle.dump(dti, handle)

    # save the final diffusion database
    # save the dataframe and the info dict
    data.attrs["mask"] = mask_3c
    save_path = os.path.join(settings["results"], "data", "database.zip")
    data.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})

    # get git commit hash
    def get_git_revision_hash():
        try:
            return (
                subprocess.check_output(
                    ["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))
                )
                .strip()
                .decode()
            )
        except:
            return "No Git repository found!"

    info["git_hash"] = get_git_revision_hash()

    # save to disk basic info in a csv file
    with open(os.path.join(settings["results"], "data", "DTI_data.csv"), "w", newline="") as handle1:
        w = csv.writer(handle1)
        w.writerows(info.items())

    # do final results montage
    make_results_montage(settings, slices, folder_id)


def make_results_montage(settings, slices, folder_id):
    for slice_idx in slices:
        slice_montage = []
        # list of images to include in the montage
        folder_a = os.path.join(settings["results"], "results_a")
        list_all_slice_imgs = [
            os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith(f"_{str(slice_idx).zfill(2)}.png")
        ]
        # select sublist where the filename starts with maps
        list_slice_maps = [f for f in list_all_slice_imgs if os.path.basename(f).startswith("maps")]
        list_slice_maps.sort()
        # same for histograms
        list_slice_histograms = [f for f in list_all_slice_imgs if os.path.basename(f).startswith("histograms")]
        list_slice_histograms.sort()

        # read all maps
        images = {}
        for map in list_slice_maps:
            map_name = os.path.basename(map)
            map_path = map
            images[map_name] = skimage.io.imread(map_path)
        montage = skimage.util.montage(
            list(images.values()),
            grid_shape=(1, len(images)),
            padding_width=0,
            channel_axis=3,
        )

        slice_montage.append(montage)

        # repeat for histograms
        images_histograms = {}
        for hist in list_slice_histograms:
            hist_name = os.path.basename(hist)
            hist_path = hist
            images_histograms[hist_name] = skimage.io.imread(hist_path)
        montage_histograms = skimage.util.montage(
            list(images_histograms.values()),
            grid_shape=(1, len(images_histograms)),
            padding_width=0,
            channel_axis=3,
        )

        slice_montage.append(montage_histograms)

        # now read results from folder b
        folder_b = os.path.join(settings["results"], "results_b")
        list_all_slice_imgs = [
            os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith(f"_{str(slice_idx).zfill(2)}.png")
        ]
        list_all_slice_imgs.sort()
        images = {}
        for image in list_all_slice_imgs:
            image_name = os.path.basename(image)
            image_path = image
            c_img = skimage.io.imread(image_path)
            # fit within 500x500 and pad if necessary
            if c_img.shape[0] != 500 or c_img.shape[1] != 500:
                scale = min(500 / c_img.shape[0], 500 / c_img.shape[1])
                resized_height = max(1, round(c_img.shape[0] * scale))
                resized_width = max(1, round(c_img.shape[1] * scale))
                resized_img = skimage.transform.resize(
                    c_img,
                    (resized_height, resized_width, *c_img.shape[2:]),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.uint8)

                c_img = np.full((500, 500, *c_img.shape[2:]), 255, dtype=np.uint8)
                top = (500 - resized_height) // 2
                left = (500 - resized_width) // 2
                c_img[top : top + resized_height, left : left + resized_width] = resized_img
            images[image_name] = c_img
        montage_b = skimage.util.montage(
            list(images.values()),
            grid_shape=(1, len(images)),
            padding_width=0,
            channel_axis=3,
        )

        slice_montage.append(montage_b)

        # make sure all images have the same size, if not pad the smaller ones
        max_height = max(img.shape[0] for img in slice_montage)
        max_width = max(img.shape[1] for img in slice_montage)
        for i in range(len(slice_montage)):
            img = slice_montage[i]
            if img.shape[0] != max_height or img.shape[1] != max_width:
                padded_img = np.full((max_height, max_width, *img.shape[2:]), 255, dtype=np.uint8)
                top = (max_height - img.shape[0]) // 2
                left = (max_width - img.shape[1]) // 2
                padded_img[top : top + img.shape[0], left : left + img.shape[1]] = img
                slice_montage[i] = padded_img

        # save the montage
        slice_montage = np.concatenate(slice_montage, axis=0)
        caption = np.full((30, *slice_montage.shape[1:]), 255, dtype=np.uint8)
        caption_image = Image.fromarray(caption)
        draw = ImageDraw.Draw(caption_image)
        label = str(folder_id)
        font = ImageFont.load_default(size=20)
        text_bounds = draw.textbbox((0, 0), label, font=font)
        text_x = (caption_image.width - (text_bounds[2] - text_bounds[0])) // 2 - text_bounds[0]
        text_y = (caption_image.height - (text_bounds[3] - text_bounds[1])) // 2 - text_bounds[1]
        draw.text((text_x, text_y), label, fill="black", font=font)
        slice_montage = np.concatenate((slice_montage, np.asarray(caption_image)), axis=0)
        montage_filename = f"results_montage_{folder_id}_slice_{str(slice_idx).zfill(2)}.png"
        montage_filename = os.path.join(os.path.abspath(settings["results"]), montage_filename)
        skimage.io.imsave(montage_filename, slice_montage.astype(np.uint8))


def export_summary_table(dti: dict, settings: dict, slices: NDArray):
    """Export summary statistics of DTI values to CSV.

    Args:
        dti (dict): DTI maps and derived metrics.
        settings (dict): Output settings containing the results path.
        slices (NDArray): Slice indices.
    """
    # get absolute E2A and TA
    dti["abs_e2a"] = np.abs(dti["e2a"])
    dti["abs_ta"] = np.abs(dti["ta"])
    dti["md_1e3"] = dti["md"] * 1000
    dti["frob_norm_1e3"] = dti["frob_norm"] * 1000
    dti["mag_anisotropy_1e3"] = dti["mag_anisotropy"] * 1000
    var_list = ["fa", "mode", "md_1e3", "frob_norm_1e3", "mag_anisotropy_1e3", "ha", "e2a", "ta", "abs_e2a", "abs_ta"]
    str_list = ["FA", "MODE", "MD", "NORM", "MAG_ANISOTROPY", "HA", "E2A", "TA", "|E2A|", "|TA|"]
    # global values
    table_global = []
    table_global.append(["Global", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    table_global.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    for idx, var in enumerate(var_list):
        table_global.append(
            [
                str_list[idx],
                np.nanmean(dti[var]),
                np.nanstd(dti[var]),
                np.nanmedian(dti[var]),
                np.nanquantile(dti[var], 0.25),
                np.nanquantile(dti[var], 0.75),
                np.nanmin(dti[var]),
                np.nanmax(dti[var]),
            ]
        )

    # also add HA line profile slopes and R^2 values
    ha_line_profiles_slope_vals = []
    ha_line_profiles_r_sq_vals = []
    ha_line_profiles_2_slope_vals = []
    ha_line_profiles_2_r_sq_vals = []
    for slice_idx in slices:
        ha_line_profiles_slope_vals.append(dti["ha_line_profiles"][slice_idx]["slope"])
        ha_line_profiles_r_sq_vals.append(dti["ha_line_profiles"][slice_idx]["r_sq"])
        ha_line_profiles_2_slope_vals.append(dti["ha_line_profiles_2"][slice_idx]["slope"])
        ha_line_profiles_2_r_sq_vals.append(dti["ha_line_profiles_2"][slice_idx]["r_sq"])
    table_global.append(
        [
            "HAT_1 slope",
            np.nanmean(ha_line_profiles_slope_vals),
            np.nanstd(ha_line_profiles_slope_vals),
            np.nanmedian(ha_line_profiles_slope_vals),
            np.nanquantile(ha_line_profiles_slope_vals, 0.25),
            np.nanquantile(ha_line_profiles_slope_vals, 0.75),
            np.nanmin(ha_line_profiles_slope_vals),
            np.nanmax(ha_line_profiles_slope_vals),
        ]
    )
    table_global.append(
        [
            "HAT_1 r-sq",
            np.nanmean(ha_line_profiles_r_sq_vals),
            np.nanstd(ha_line_profiles_r_sq_vals),
            np.nanmedian(ha_line_profiles_r_sq_vals),
            np.nanquantile(ha_line_profiles_r_sq_vals, 0.25),
            np.nanquantile(ha_line_profiles_r_sq_vals, 0.75),
            np.nanmin(ha_line_profiles_r_sq_vals),
            np.nanmax(ha_line_profiles_r_sq_vals),
        ]
    )
    table_global.append(
        [
            "HAT_2 slope",
            np.nanmean(ha_line_profiles_2_slope_vals),
            np.nanstd(ha_line_profiles_2_slope_vals),
            np.nanmedian(ha_line_profiles_2_slope_vals),
            np.nanquantile(ha_line_profiles_2_slope_vals, 0.25),
            np.nanquantile(ha_line_profiles_2_slope_vals, 0.75),
            np.nanmin(ha_line_profiles_2_slope_vals),
            np.nanmax(ha_line_profiles_2_slope_vals),
        ]
    )
    table_global.append(
        [
            "HAT_2 r-sq",
            np.nanmean(ha_line_profiles_2_r_sq_vals),
            np.nanstd(ha_line_profiles_2_r_sq_vals),
            np.nanmedian(ha_line_profiles_2_r_sq_vals),
            np.nanquantile(ha_line_profiles_2_r_sq_vals, 0.25),
            np.nanquantile(ha_line_profiles_2_r_sq_vals, 0.75),
            np.nanmin(ha_line_profiles_2_r_sq_vals),
            np.nanmax(ha_line_profiles_2_r_sq_vals),
        ]
    )

    # also calculate global values for SNR
    # gather all keys found in all slices
    keys = []
    for slice_idx in slices:
        keys.extend(dti["snr"][slice_idx].keys())
    keys = list(set(keys))
    # sort keys
    keys.sort()
    # loop over each key and collect all values from all slices
    for key in keys:
        values = []
        for slice_idx in slices:
            if key in dti["snr"][slice_idx]:
                c_vals = dti["snr"][slice_idx][key]
                c_vals = c_vals[~np.isnan(c_vals)]
                c_vals = c_vals[np.isfinite(c_vals)]
                values.extend(c_vals)
        table_global.append(
            [
                "SNR_" + key,
                np.nanmean(values),
                np.nanstd(values),
                np.nanmedian(values),
                np.nanquantile(values, 0.25),
                np.nanquantile(values, 0.75),
                np.nanmin(values),
                np.nanmax(values),
            ]
        )

    if len(slices) > 1:
        # per slice values
        table_per_slice = []
        table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        table_per_slice.append(["Per slice", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for idx, var in enumerate(var_list):
            table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            for slice_idx in slices:
                table_per_slice.append(
                    [
                        str_list[idx] + "_slice_" + str(slice_idx).zfill(2),
                        np.nanmean(dti[var][slice_idx]),
                        np.nanstd(dti[var][slice_idx]),
                        np.nanmedian(dti[var][slice_idx]),
                        np.nanquantile(dti[var][slice_idx], 0.25),
                        np.nanquantile(dti[var][slice_idx], 0.75),
                        np.nanmin(dti[var][slice_idx]),
                        np.nanmax(dti[var][slice_idx]),
                    ]
                )
        table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for slice_idx in slices:
            table_per_slice.append(
                [
                    "HAT_1 slope",
                    dti["ha_line_profiles"][slice_idx]["slope"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
            table_per_slice.append(
                [
                    "HAT_1 r-sq",
                    dti["ha_line_profiles"][slice_idx]["r_sq"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
        for slice_idx in slices:
            table_per_slice.append(
                [
                    "HAT_2 slope",
                    dti["ha_line_profiles_2"][slice_idx]["slope"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
            table_per_slice.append(
                [
                    "HAT_2 r-sq",
                    dti["ha_line_profiles_2"][slice_idx]["r_sq"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
        # do the same for SNR
        table_per_slice.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for slice_idx in slices:
            for key in dti["snr"][slice_idx]:
                c_vals = dti["snr"][slice_idx][key]
                c_vals = c_vals[~np.isnan(c_vals)]
                c_vals = c_vals[np.isfinite(c_vals)]
                # check if c_vals is not empty
                if len(c_vals) == 0:
                    continue
                table_per_slice.append(
                    [
                        "SNR_" + key + "_slice_" + str(slice_idx).zfill(2),
                        np.nanmean(c_vals),
                        np.nanstd(c_vals),
                        np.nanmedian(c_vals),
                        np.nanquantile(c_vals, 0.25),
                        np.nanquantile(c_vals, 0.75),
                        np.nanmin(c_vals),
                        np.nanmax(c_vals),
                    ]
                )

        # add to the global table
        table_global.extend(table_per_slice)

    # convert table to dataframe
    table_global = pd.DataFrame(
        table_global,
        columns=["Parameter", "Mean", "Std", "Median", "Q25", "Q75", "Min", "Max"],
    )
    table_global = table_global.round(decimals=2)
    # export dataframe csv
    table_global.to_csv(os.path.join(settings["results"], "results_table.csv"), index=False)


def export_to_hdf5(dti: dict, mask_3c: NDArray, settings: dict):
    """Export DTI maps and masks to an HDF5 file.

    Args:
        dti (dict): DTI maps to persist.
        mask_3c (NDArray): Segmentation mask to include.
        settings (dict): Output settings including the results path.
    """
    with h5py.File(os.path.join(settings["results"], "data", "DTI_maps" + ".h5"), "w") as hf:
        for name, key in dti.items():
            if isinstance(key, np.ndarray) or isinstance(key, float):
                hf.create_dataset(name, data=key)
            elif isinstance(key, dict):
                for subname, subkey in key.items():
                    if isinstance(subkey, np.ndarray) or isinstance(subkey, float):
                        hf.create_dataset(name + "_" + str(subname), data=subkey)
                    elif isinstance(key, dict):
                        for subsubname, subsubkey in subkey.items():
                            if isinstance(subsubkey, np.ndarray) or isinstance(subsubkey, float):
                                hf.create_dataset(name + "_" + str(subname) + "_" + subsubname, data=subsubkey)
        hf.create_dataset("mask", data=mask_3c)

    # # to read a map example
    # with h5py.File(os.path.join(settings["results"], "data", "DTI_maps" + ".h5"), "r") as hf:
    #     data = hf["ha"][:]


def plot_results_maps(
    slices: NDArray,
    mask_3c: NDArray,
    average_images: dict,
    dti: dict,
    segmentation: dict,
    colormaps: dict,
    settings: dict,
    folder_id: str,
):
    """Plot and export the main montage of DTI results.

    Args:
        slices (NDArray): Slice identifiers.
        mask_3c (NDArray): Heart segmentation mask.
        average_images (dict): Average image per slice.
        dti (dict): DTI parameters and derived maps.
        segmentation (dict): LV segmentation details (contours and insertion
            points).
        colormaps (dict): Custom colour maps for visualisation.
        settings (dict): Processing and output settings.
        folder_id (str): Identifier appended to exported filenames.
    """

    params = {}

    params["MD"] = {}
    params["MD"]["var_name"] = "md"
    params["MD"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["MD"]["cmap"] = colormaps["MD"]
    params["MD"]["title"] = "Mean diffusivity"
    params["MD"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MD"]["scale"] = 1000
    params["MD"]["abs"] = False

    params["MAG_ANISOTROPY"] = {}
    params["MAG_ANISOTROPY"]["var_name"] = "mag_anisotropy"
    params["MAG_ANISOTROPY"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["MAG_ANISOTROPY"]["cmap"] = "viridis"
    params["MAG_ANISOTROPY"]["title"] = "Mag anisotropy"
    params["MAG_ANISOTROPY"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MAG_ANISOTROPY"]["scale"] = 1000
    params["MAG_ANISOTROPY"]["abs"] = False

    params["MODE"] = {}
    params["MODE"]["var_name"] = "mode"
    params["MODE"]["vmin_max"] = [-1, 1]
    params["MODE"]["cmap"] = "plasma"
    params["MODE"]["title"] = "Tensor mode"
    params["MODE"]["units"] = "10^{-3} mm^2s^{-1}"
    params["MODE"]["scale"] = 1
    params["MODE"]["abs"] = False

    params["FA"] = {}
    params["FA"]["var_name"] = "fa"
    params["FA"]["vmin_max"] = [0, 1]
    params["FA"]["cmap"] = colormaps["FA"]
    params["FA"]["title"] = "Fractional anisotropy"
    params["FA"]["units"] = "[]"
    params["FA"]["scale"] = 1
    params["FA"]["abs"] = False

    params["FROB_NORM"] = {}
    params["FROB_NORM"]["var_name"] = "frob_norm"
    params["FROB_NORM"]["vmin_max"] = [settings["md_scale"][0], settings["md_scale"][1]]
    params["FROB_NORM"]["cmap"] = "inferno"
    params["FROB_NORM"]["title"] = "Frobenius norm"
    params["FROB_NORM"]["units"] = "10^{-3} mm^2s^{-1}"
    params["FROB_NORM"]["scale"] = 1000
    params["FROB_NORM"]["abs"] = False

    params["HA"] = {}
    params["HA"]["var_name"] = "ha"
    params["HA"]["vmin_max"] = [-90, 90]
    params["HA"]["cmap"] = colormaps["HA"]
    params["HA"]["title"] = "Helix angle"
    params["HA"]["units"] = "degrees"
    params["HA"]["scale"] = 1
    params["HA"]["abs"] = False

    params["TA"] = {}
    params["TA"]["var_name"] = "ta"
    params["TA"]["vmin_max"] = [-90, 90]
    params["TA"]["cmap"] = "twilight_shifted"
    params["TA"]["title"] = "Transverse angle"
    params["TA"]["units"] = "degrees"
    params["TA"]["scale"] = 1
    params["TA"]["abs"] = False

    params["E2A"] = {}
    params["E2A"]["var_name"] = "e2a"
    params["E2A"]["vmin_max"] = [-90, 90]
    params["E2A"]["cmap"] = "twilight_shifted"
    params["E2A"]["title"] = "Sheetlet angle"
    params["E2A"]["units"] = "degrees"
    params["E2A"]["scale"] = 1
    params["E2A"]["abs"] = False

    params["abs_E2A"] = {}
    params["abs_E2A"]["var_name"] = "e2a"
    params["abs_E2A"]["vmin_max"] = [0, 90]
    params["abs_E2A"]["cmap"] = colormaps["abs_E2A"]
    params["abs_E2A"]["title"] = "Absolute sheetlet angle"
    params["abs_E2A"]["units"] = "degrees"
    params["abs_E2A"]["scale"] = 1
    params["abs_E2A"]["abs"] = True

    # Also plots maps individually
    for slice_idx in slices:
        alphas_whole_heart = np.copy(mask_3c[slice_idx])
        alphas_whole_heart[alphas_whole_heart > 0.1] = 1

        alphas_myocardium = np.copy(mask_3c[slice_idx])
        alphas_myocardium[alphas_myocardium == 2] = 0
        alphas_myocardium[alphas_myocardium > 0.1] = 1

        for param in params:
            plt.figure(figsize=(5, 5))
            plt.imshow(average_images[slice_idx], cmap="Greys_r")
            if params[param]["abs"]:
                plt.imshow(
                    np.abs(dti[params[param]["var_name"]][slice_idx] * params[param]["scale"]),
                    alpha=alphas_whole_heart,
                    vmin=params[param]["vmin_max"][0],
                    vmax=params[param]["vmin_max"][1],
                    cmap=params[param]["cmap"],
                )
            else:
                plt.imshow(
                    dti[params[param]["var_name"]][slice_idx] * params[param]["scale"],
                    alpha=alphas_whole_heart,
                    vmin=params[param]["vmin_max"][0],
                    vmax=params[param]["vmin_max"][1],
                    cmap=params[param]["cmap"],
                )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout(pad=1.0)
            plt.axis("off")
            plt.title(params[param]["title"])
            plt.savefig(
                os.path.join(
                    settings["results"],
                    "results_a",
                    "maps_" + param + "_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

            plt.figure(figsize=(5, 5))
            if params[param]["abs"]:
                vals = abs(dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"])
            else:
                vals = dti[params[param]["var_name"]][slice_idx][alphas_myocardium > 0] * params[param]["scale"]
            bins = np.linspace(params[param]["vmin_max"][0], params[param]["vmin_max"][1], 40)
            weights = np.ones_like(vals) / len(vals)
            n, bins, patches = plt.hist(vals, bins=bins, weights=weights, rwidth=0.95)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            # scale values to interval [0,1]
            col = bin_centers - min(bin_centers)
            col /= max(col)
            cm = matplotlib.colormaps.get_cmap(params[param]["cmap"])
            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cm(c))
            plt.title(params[param]["title"])
            plt.savefig(
                os.path.join(
                    settings["results"],
                    "results_a",
                    "histograms_" + param + "_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=100,
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        # plot LV 12 sectors
        plt.figure(figsize=(5, 5))
        plt.imshow(average_images[slice_idx], cmap="Greys_r")
        plt.imshow(
            dti["lv_sectors"][slice_idx],
            alpha=alphas_whole_heart * 0.5,
            vmin=1,
            vmax=12,
            cmap=matplotlib.colors.ListedColormap(matplotlib.colormaps.get_cmap("tab20c").colors[0:12]),
        )
        if segmentation[slice_idx]["anterior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["anterior_ip"][0],
                segmentation[slice_idx]["anterior_ip"][1],
                "2",
                color="tab:orange",
                markersize=20,
                alpha=1.0,
            )
        if segmentation[slice_idx]["inferior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["inferior_ip"][0],
                segmentation[slice_idx]["inferior_ip"][1],
                "1",
                color="tab:orange",
                markersize=20,
                alpha=1.0,
            )
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        ticklabels = [str(i) for i in range(1, 12 + 1)]
        tickpos = np.linspace(1.5, 12 - 0.5, 12)
        cbar.set_ticks(tickpos, labels=ticklabels)
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.title("LV sectors")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "lv_segments_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()

        # plot S0 map
        plt.figure(figsize=(5, 5))
        # plt.imshow(average_images[slice_idx], cmap="Blues_r", vmin=0, vmax=1)
        vmin, vmax = get_window(dti["s0"][slice_idx], mask_3c[slice_idx])
        plt.imshow(dti["s0"][slice_idx], cmap="Greys_r", vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        if segmentation[slice_idx]["epicardium_true_border"].size != 0:
            plt.scatter(
                segmentation[slice_idx]["epicardium_true_border"][:, 0],
                segmentation[slice_idx]["epicardium_true_border"][:, 1],
                marker=".",
                s=10,
                color="tab:blue",
                alpha=0.8,
            )
        if segmentation[slice_idx]["endocardium_true_border"].size != 0:
            plt.scatter(
                segmentation[slice_idx]["endocardium_true_border"][:, 0],
                segmentation[slice_idx]["endocardium_true_border"][:, 1],
                marker=".",
                s=10,
                color="tab:red",
                alpha=0.8,
            )
        if segmentation[slice_idx]["anterior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["anterior_ip"][0],
                segmentation[slice_idx]["anterior_ip"][1],
                "2",
                color="tab:orange",
                markersize=20,
                alpha=0.8,
            )
        if segmentation[slice_idx]["inferior_ip"].size != 0:
            plt.plot(
                segmentation[slice_idx]["inferior_ip"][0],
                segmentation[slice_idx]["inferior_ip"][1],
                "1",
                color="tab:orange",
                markersize=20,
                alpha=0.8,
            )
        plt.tight_layout(pad=1.0)
        plt.axis("off")
        plt.title("S0")
        plt.savefig(
            os.path.join(
                settings["results"],
                "results_b",
                "maps_s0_slice_" + str(slice_idx).zfill(2) + ".png",
            ),
            dpi=100,
            pad_inches=0,
            transparent=False,
        )
        plt.close()
