import logging
import os
import sys
import time

import itk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def registration_loop(
    data: pd.DataFrame, ref_images: dict, slices: NDArray, info: dict, settings: dict, logger: logging.Logger
) -> tuple[pd.DataFrame, NDArray, NDArray]:
    """
    Registration image loop. It will use differents parts of the code depending
    the method of registration

    Parameters
    ----------
    data: data to be registered
    ref_images: dictionary with the reference images and some other info
    slices: array with slice positions
    info: useful info
    settings: useful info
    logger: logger

    Returns
    -------
    dataframe with registered images, arrays with images pre- and post-registration
    """

    # ============================================================
    # Elastix Registration config files
    # ============================================================
    if settings["registration"] == "elastix_rigid":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_rigid.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

    if settings["registration"] == "elastix_affine":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_rigid.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_affine.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

    if settings["registration"] == "elastix_non_rigid":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_rigid.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")
        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_affine.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")
        parameter_object.AddParameterFile(
            os.path.join(settings["code_path"], "extensions", "image_registration_recipes", "Elastix_bspline.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

        # # Export custom parameter maps to files
        # for index in range(parameter_object.GetNumberOfParameterMaps()):
        #     parameter_map = parameter_object.GetParameterMap(index)
        #     parameter_object.WriteParameterFile(
        #         parameter_map,
        #         os.path.join(
        #             settings["code_path"],
        #             "extensions",
        #             "image_registration_recipes",
        #             "current_parameters{0}.txt".format(index),
        #         ),
        #     )

    if settings["registration"] == "elastix_groupwise":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(
            os.path.join(
                settings["code_path"],
                "extensions",
                "image_registration_recipes",
                "Elastix_groupwise.txt",
            )
        )

        # parameter_object = itk.ParameterObject.New()
        # groupwise_parameter_map = parameter_object.GetDefaultParameterMap("groupwise")
        # parameter_object.AddParameterMap(groupwise_parameter_map)

        # # Export custom parameter maps to files
        # for index in range(parameter_object.GetNumberOfParameterMaps()):
        #     parameter_map = parameter_object.GetParameterMap(index)
        #     parameter_object.WriteParameterFile(
        #         parameter_map,
        #         os.path.join(
        #             settings["code_path"],
        #             "extensions",
        #             "image_registration_recipes",
        #             "current_parameters{0}.txt".format(index),
        #         ),
        #     )

    # loop over slices
    img_post_reg = {}
    img_pre_reg = {}
    for slice_idx in slices:
        # store images after registration
        img_post_reg[slice_idx] = np.empty(
            [ref_images[slice_idx]["n_images"], info["img_size"][0], info["img_size"][1]]
        )
        # store images before registration
        img_pre_reg[slice_idx] = np.empty(
            [ref_images[slice_idx]["n_images"], info["img_size"][0], info["img_size"][1]]
        )

        # reference image for this slice
        ref = ref_images[slice_idx]["image"]
        ref = np.asarray(ref, dtype=np.float32)
        if (
            settings["registration"] == "elastix_rigid"
            or settings["registration"] == "elastix_affine"
            or settings["registration"] == "elastix_non_rigid"
        ):
            ref = itk.GetImageFromArray(ref)

        # dataframe for this slice
        current_entries = data.loc[data["slice_integer"] == slice_idx]

        # images to be registered
        mov_all = np.transpose(np.dstack(current_entries["image"].values), (2, 0, 1))

        # create mask for the registration
        # this mask removes the edges along the readout direction
        mask = np.zeros([info["img_size"][0], info["img_size"][1]])
        if info["img_size"][0] > info["img_size"][1]:
            short_dim = info["img_size"][1]
            large_dim = info["img_size"][0]
            mask[int((large_dim - short_dim * 1.2) / 2) : int((large_dim + short_dim * 1.2) / 2), :] = 1
        else:
            short_dim = info["img_size"][0]
            large_dim = info["img_size"][1]
            mask[:, int((large_dim - short_dim * 1.2) / 2) : int((large_dim + short_dim * 1.2) / 2)] = 1
        mask_arr = np.asarray(mask, dtype=np.ubyte)
        if settings["registration"] == "elastix_groupwise":
            # stack the mask for every image
            mask_arr = np.repeat(mask_arr[np.newaxis, :, :], mov_all.shape[0], axis=0)

        mask = itk.GetImageFromArray(mask_arr)

        # Groupwise registration
        if settings["registration"] == "elastix_groupwise":
            # get all images
            mov_all = np.ascontiguousarray(np.array(mov_all, dtype=np.float32))

            # store images before registration
            img_pre_reg[slice_idx] = np.copy(mov_all)

            logger.info("Slice " + str(slice_idx).zfill(2) + ": Starting groupwise registration. Please hold...")

            # register all images groupwise
            img_reg, result_transform_parameters = itk.elastix_registration_method(
                mov_all,
                mov_all,
                parameter_object=parameter_object,
                fixed_mask=mask,
                log_to_console=False,
            )

            # format registered images
            img_reg = np.copy(np.asarray(img_reg, dtype=np.float32))
            img_reg[img_reg < 0] = 0
            img_reg[np.isnan(img_reg)] = 0

            # store images after registration
            img_post_reg[slice_idx] = np.copy(img_reg)

            # replace images in the dataframe
            for i in current_entries.index:
                data.at[i, "image"] = img_reg[i]

        else:
            # if not groupwise registration
            logger.info("Slice " + str(slice_idx).zfill(2) + ": Starting registration. Please hold...")
            # loop through all images and register one by one
            for i in tqdm(range(ref_images[slice_idx]["n_images"]), desc="Registering images"):
                if i == ref_images[slice_idx]["index"]:
                    # here the current image is the reference, so we do not perform registration
                    img_pre_reg[slice_idx][i] = np.asarray(mov_all[i], dtype=np.float32)
                    img_reg = img_pre_reg[slice_idx][i]

                else:
                    # moving image
                    mov = np.asarray(mov_all[i], dtype=np.float32)
                    img_pre_reg[slice_idx][i] = mov

                    # run registration
                    # elastix
                    if (
                        settings["registration"] == "elastix_rigid"
                        or settings["registration"] == "elastix_affine"
                        or settings["registration"] == "elastix_non_rigid"
                    ):
                        mov = itk.GetImageFromArray(mov)
                        img_reg, result_transform_parameters = itk.elastix_registration_method(
                            ref,
                            mov,
                            parameter_object=parameter_object,
                            fixed_mask=mask,
                            log_to_console=False,
                        )
                    # basic quick rigid
                    elif settings["registration"] == "quick_rigid":
                        shift, error, diffphase = phase_cross_correlation(
                            np.multiply(mask_arr, ref), np.multiply(mask_arr, mov), upsample_factor=100
                        )
                        img_reg = fourier_shift(np.fft.fftn(mov), shift)
                        img_reg = np.abs(np.fft.ifftn(img_reg))

                    # none
                    elif settings["registration"] == "none":
                        img_reg = mov
                    else:
                        logger.error("No method available fot registration: " + settings["registration"])
                        sys.exit()

                    # store registered image
                    # correct for registration small errors
                    img_reg = np.asarray(img_reg, dtype=np.float32)
                    img_reg[img_reg < 0] = 0

                # replace the images in the dataframe with all slices
                data.at[current_entries.index[i], "image"] = img_reg

                # store images post registration
                img_post_reg[slice_idx][i] = img_reg

    return data, img_pre_reg, img_post_reg


def get_ref_image(current_entries: pd.DataFrame, slice_idx: int, settings: dict, logger: logging.Logger) -> dict:
    """

    Get all the lower b-value images, and groupwise register them if more than one.
    Then get the mean. That will be our reference image for the subsequent registration.
    If the registration is groupwise, then we do not need to calculate a reference.

    Parameters
    ----------
    current_entries dataframe with one slice only
    slice_str slice position string
    settings
    logger

    Returns
    -------
    dict with reference image

    """

    ref_images = {}
    if settings["registration"] != "elastix_groupwise":
        # get unique b-values
        b_values = current_entries["b_value_original"].unique()
        # sort b-values
        b_values = np.sort(b_values)
        # get indices for the lowest b-value
        index_pos = current_entries.index[current_entries["b_value_original"] == b_values[0]].tolist()
        n_images = len(index_pos)

        if n_images < 2 or settings["registration_reference_method"] == "first":
            if n_images < 2:
                logger.info(
                    "Slice "
                    + str(slice_idx).zfill(2)
                    + ": only one image found for the lowest b-value, using that image as reference"
                )
            else:
                logger.info(
                    "Slice "
                    + str(slice_idx).zfill(2)
                    + ": using the first image as reference, registration_reference_method = first"
                )
            ref_images["image"] = current_entries.at[index_pos[0], "image"]
            ref_images["index"] = index_pos
            ref_images["n_images"] = len(current_entries)
            ref_images["groupwise_reg_info"] = {}
        else:
            logger.info(
                "Slice "
                + str(slice_idx).zfill(2)
                + ": "
                + str(n_images)
                + " images found for the lowest b-value, registering them groupwise for a reference. Please hold..."
            )
            # stack all images to be registered
            image_stack = np.stack(current_entries["image"][index_pos].values)

            # groupwise registration recipe
            parameter_object = itk.ParameterObject.New()
            parameter_object.AddParameterFile(
                os.path.join(
                    settings["code_path"],
                    "extensions",
                    "image_registration_recipes",
                    "Elastix_groupwise.txt",
                )
            )

            # parameter_object = itk.ParameterObject.New()
            # groupwise_parameter_map = parameter_object.GetDefaultParameterMap("groupwise")
            # parameter_object.AddParameterMap(groupwise_parameter_map)

            # modify type for itk
            image_stack = np.ascontiguousarray(np.array(image_stack, dtype=np.float32))
            # store images before registration
            img_pre = image_stack

            # create mask stack of the FOV central region
            mask = np.zeros([image_stack.shape[1], image_stack.shape[2]])
            if image_stack.shape[1] > image_stack.shape[2]:
                short_dim = image_stack.shape[2]
                large_dim = image_stack.shape[1]
                mask[int((large_dim - short_dim * 1.2) / 2) : int((large_dim + short_dim * 1.2) / 2), :] = 1
            else:
                short_dim = image_stack.shape[1]
                large_dim = image_stack.shape[2]
                mask[:, int((large_dim - short_dim * 1.2) / 2) : int((large_dim + short_dim * 1.2) / 2)] = 1
            mask_arr = np.asarray(mask, dtype=np.ubyte)
            mask_arr = np.repeat(mask_arr[np.newaxis, :, :], n_images, axis=0)
            mask = itk.GetImageFromArray(mask_arr)

            # register all images groupwise
            t0 = time.time()
            img_reg, result_transform_parameters = itk.elastix_registration_method(
                image_stack,
                image_stack,
                parameter_object=parameter_object,
                fixed_mask=mask,
                log_to_console=False,
            )
            t1 = time.time()
            total = t1 - t0
            logger.info(
                "Slice "
                + str(slice_idx).zfill(2)
                + ": Time for groupwise registration: "
                + str(int(total))
                + " seconds"
            )

            # format registered images
            img_reg = np.copy(np.asarray(img_reg, dtype=np.float32))
            img_reg[img_reg < 0] = 0
            img_reg[np.isnan(img_reg)] = 0

            # reference image is the mean of the registered images
            c_ref = np.mean(img_reg, axis=0)

            # save all the needed info to a dict
            ref_images["image"] = c_ref
            ref_images["index"] = index_pos
            ref_images["n_images"] = len(current_entries)
            ref_images["groupwise_reg_info"] = {}
            ref_images["groupwise_reg_info"]["pre"] = img_pre
            ref_images["groupwise_reg_info"]["post"] = img_reg
    else:
        logger.info(
            "Slice " + str(slice_idx).zfill(2) + ": Registration is elastix_groupwise, so reference image is None."
        )
        ref_images["image"] = None
        ref_images["index"] = 0
        ref_images["n_images"] = len(current_entries)
        ref_images["groupwise_reg_info"] = {}

    return ref_images


def plot_ref_images(ref_images: dict, slices: NDArray, settings: dict):
    """

    Parameters
    ----------
    ref_images dictionaary with all the info on the reference images used
    slices array with strings of slice positions
    settings

    Returns
    -------

    """
    if settings["debug"] and settings["registration"] != "elastix_groupwise":
        # plot reference images
        for slice_idx in slices:
            plt.figure(figsize=(5, 5))
            plt.imshow(ref_images[slice_idx]["image"], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    settings["debug_folder"],
                    "reference_images_for_registration_slice_" + str(slice_idx).zfill(2) + ".png",
                ),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        # plot results of groupwise registration
        for slice_idx in slices:
            if ref_images[slice_idx]["groupwise_reg_info"]:
                n_images = len(ref_images[slice_idx]["groupwise_reg_info"]["pre"])
                img_pre = ref_images[slice_idx]["groupwise_reg_info"]["pre"]
                img_reg = ref_images[slice_idx]["groupwise_reg_info"]["post"]
                c_ref = ref_images[slice_idx]["image"]

                plt.figure()
                for i in range(n_images):
                    plt.subplot(4, n_images, i + 1)
                    plt.imshow(img_pre[i])
                    plt.title(str(i) + "_pre")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + n_images + 1)
                    plt.imshow(img_reg[i])
                    plt.title(str(i) + "_reg")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + 2 * n_images + 1)
                    plt.imshow(abs(c_ref - img_pre[i]))
                    plt.title("diff_to_ref")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + 3 * n_images + 1)
                    plt.imshow(abs(c_ref - img_reg[i]))
                    plt.title("diff_to_ref")
                    plt.axis("off")
                plt.tight_layout(pad=1.0)
                plt.savefig(
                    os.path.join(
                        settings["debug_folder"],
                        "groupwise_registration_reference_slice_" + str(slice_idx).zfill(2) + ".png",
                    ),
                    dpi=200,
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()


def image_registration(
    data: pd.DataFrame, slices: NDArray, info: dict, settings: dict, logger: logging.Logger
) -> tuple[pd.DataFrame, NDArray, NDArray, dict]:
    """
    Image registration

    Parameters
    ----------
    data: data to be registered
    slices: array with slice integer
    info: dictionary with useful info
    settings: dict with settings
    logger: logger

    Returns
    -------
    registered dataframe, images pre- and post-registration, reference images

    """

    # check if registration has been done already
    if not os.path.exists(os.path.join(settings["session"], "image_registration_data.zip")) or not os.path.exists(
        os.path.join(settings["session"], "image_registration_extras.npz")
    ):
        logger.info("No saved registration found.")
        logger.info("Registration type: " + settings["registration"])

        # check if the reference images have been saved already
        if not os.path.exists(os.path.join(settings["session"], "image_registration_references.npz")):
            logger.info("No saved reference images found.")
            # get reference image, index position and number of images per slice
            ref_images = {}
            for slice_idx in slices:
                # dataframe for each slice
                current_entries = data.loc[data["slice_integer"] == slice_idx]
                # reference image will be the mean of all lower b-values
                ref_images[slice_idx] = get_ref_image(current_entries, slice_idx, settings, logger)

            # save reference images
            save_path = os.path.join(settings["session"], "image_registration_references.npz")
            np.savez_compressed(save_path, ref_images=ref_images)

        else:
            logger.info("Saved reference images found.")
            # load reference images
            save_path = os.path.join(settings["session"], "image_registration_references.npz")
            npzfile = np.load(save_path, allow_pickle=True)
            ref_images = npzfile["ref_images"].item()

        # plot reference images
        plot_ref_images(ref_images, slices, settings)

        # run the registration loop
        data, img_pre_reg, img_post_reg = registration_loop(data, ref_images, slices, info, settings, logger)

        logger.info("Image registration done")

        # saving registration data
        save_path = os.path.join(settings["session"], "image_registration_data.zip")
        # table with only filename, image, acquisition time and date
        data_basic = data[["file_name", "image", "acquisition_time", "acquisition_date"]]
        data_basic.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})
        # saving registration extras
        np.savez_compressed(
            os.path.join(settings["session"], "image_registration_extras.npz"),
            img_pre_reg=img_pre_reg,
            img_post_reg=img_post_reg,
            ref_images=ref_images,
        )

    else:
        logger.info("Saved registration found.")

        # loading registration data
        data_loaded_basic = pd.read_pickle(os.path.join(settings["session"], "image_registration_data.zip"))
        data_basic = data[["file_name", "image", "acquisition_time", "acquisition_date"]]
        # check if the original data basic table matches the loaded one (except the image column)
        if not data_basic.drop(columns=["image"]).equals(data_loaded_basic.drop(columns=["image"])):
            logger.error("Loaded Dataframe with registered images does not match pre-registered Dataframe!")
            logger.error("Registration saved data needs to be deleted as something changed!")
            sys.exit()

        logger.info("Passed data consistency check. Loading registered data.")
        # data matches, so now I need to replace the image column with the loaded one
        data["image"] = data_loaded_basic["image"]
        # also load the extra saved data
        npzfile = np.load(os.path.join(settings["session"], "image_registration_extras.npz"), allow_pickle=True)
        img_pre_reg = npzfile["img_pre_reg"].item()
        img_post_reg = npzfile["img_post_reg"].item()
        ref_images = npzfile["ref_images"].item()
        logger.info("Image registration loaded")

        # plot reference images
        plot_ref_images(ref_images, slices, settings)

    return data, img_pre_reg, img_post_reg, ref_images
