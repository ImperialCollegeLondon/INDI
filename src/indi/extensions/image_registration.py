import logging
import os
import time

import cv2 as cv
import itk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm


def get_grid_image(img_shape: NDArray, grid_step: int) -> NDArray:
    """
    Get an image with a regular grid. This is going to be deformed by the
    displacement field of the registration

    Args
        img_shape: shape of the image
        grid_step: step size for the grid

    Returns:
        grid_img: image with grid

    """
    grid_img = np.zeros(img_shape)
    for i in range(0, img_shape[0], grid_step):
        grid_img[i, :] = 1
    for j in range(0, img_shape[1], grid_step):
        grid_img[:, j] = 1
    return grid_img


def denoise_img_nlm(c_img: NDArray) -> NDArray:
    """
    Denoise image with non-local means

    Args
        c_img: image to denoise

    Returns:
        denoised_img: denoised image
    """
    # nlm config
    patch_kw = dict(
        patch_size=5,
        patch_distance=6,
        channel_axis=None,
    )
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(c_img, channel_axis=None))
    # fast algorithm, sigma provided
    denoised_img = denoise_nl_means(c_img, h=10 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)

    return denoised_img


def registration_loop(
    data: pd.DataFrame, ref_images: dict, mask: NDArray, info: dict, settings: dict, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    """
    Registration image loop. This is where we perform the registration of the DWIs.

    Args:
        data: data to be registered
        ref_images: dictionary with the reference images and some other info
        mask: registration mask
        info: useful info
        settings: useful info
        logger: logger

    Returns:
        data: dataframe with registered images, dict with registration info
        registration_image_data: registration image data
    """

    # ============================================================
    # Elastix Registration config files
    # ============================================================
    script_path = os.path.dirname(__file__)
    if settings["registration"] == "elastix_rigid":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(os.path.join(script_path, "image_registration_recipes", "Elastix_rigid.txt"))
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

    if settings["registration"] == "elastix_affine":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(os.path.join(script_path, "image_registration_recipes", "Elastix_rigid.txt"))
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

        parameter_object.AddParameterFile(
            os.path.join(script_path, "image_registration_recipes", "Elastix_affine.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

    if settings["registration"] == "elastix_non_rigid":
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(os.path.join(script_path, "image_registration_recipes", "Elastix_rigid.txt"))
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

        parameter_object.AddParameterFile(
            os.path.join(script_path, "image_registration_recipes", "Elastix_affine.txt")
        )
        if settings["registration_speed"] == "slow":
            parameter_object.SetParameter("MaximumNumberOfIterations", "2000")
            parameter_object.SetParameter("NumberOfResolutions", "4")

        parameter_object.AddParameterFile(
            os.path.join(script_path, "image_registration_recipes", "Elastix_bspline.txt")
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
                script_path,
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

    # dicts to store information about the registration
    registration_image_data = {}

    # initialise images after registration
    registration_image_data["img_post_reg"] = np.empty(
        [ref_images["n_images"], info["img_size"][0], info["img_size"][1]]
    )
    # initialise images before registration
    registration_image_data["img_pre_reg"] = np.empty(
        [ref_images["n_images"], info["img_size"][0], info["img_size"][1]]
    )
    # initialise deformation field and grid
    registration_image_data["deformation_field"] = {}
    registration_image_data["deformation_field"]["field"] = np.zeros(
        [ref_images["n_images"], info["img_size"][0], info["img_size"][1], 2]
    )
    registration_image_data["deformation_field"]["grid"] = np.zeros(
        [ref_images["n_images"], info["img_size"][0], info["img_size"][1]]
    )

    # reference image for this slice
    ref = ref_images["image"]
    ref = np.asarray(ref, dtype=np.float32)
    if (
        settings["registration"] == "elastix_rigid"
        or settings["registration"] == "elastix_affine"
        or settings["registration"] == "elastix_non_rigid"
    ):
        ref = itk.GetImageFromArray(ref)

    # images to be registered
    mov_all = np.transpose(np.dstack(data["image"].values), (2, 0, 1))
    if settings["complex_data"]:
        mov_all_phase = np.transpose(np.dstack(data["image_phase"].values), (2, 0, 1))

    # mask only regions of interest in the middle of the FOV.
    mask = itk.GetImageFromArray(mask)

    # Groupwise registration
    if settings["registration"] == "elastix_groupwise":
        if settings["complex_data"]:
            logger.error("Elastix groupwise registration not tested for complex data.")
            raise ValueError("Elastix groupwise registration not tested for complex data.")

        # get all images
        mov_all = np.ascontiguousarray(np.array(mov_all, dtype=np.float32))

        # store images before registration
        registration_image_data["img_pre_reg"] = np.copy(mov_all)

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
        registration_image_data["img_post_reg"] = np.copy(img_reg)

        # replace images in the dataframe
        for i in data.index:
            data.at[i, "image"] = img_reg[i]

    else:
        # if not groupwise registration
        # loop through all images and register one by one
        for i in tqdm(range(ref_images["n_images"]), desc="Registering images"):
            # moving image
            mov = np.asarray(mov_all[i], dtype=np.float32)
            registration_image_data["img_pre_reg"][i] = mov
            if settings["complex_data"]:
                mov_phase = np.asarray(mov_all_phase[i], dtype=np.float32)

            # run registration
            # elastix
            if (
                settings["registration"] == "elastix_rigid"
                or settings["registration"] == "elastix_affine"
                or settings["registration"] == "elastix_non_rigid"
            ):
                # apply the registration to a denoised version (helps with registration of low SNR images)
                mov_norm = (mov - np.min(mov)) / (np.max(mov) - np.min(mov))
                denoised_mov = denoise_img_nlm(mov_norm)

                denoised_mov = itk.GetImageFromArray(denoised_mov)

                img_reg, result_transform_parameters = itk.elastix_registration_method(
                    ref,
                    denoised_mov,
                    parameter_object=parameter_object,
                    fixed_mask=mask,
                    log_to_console=False,
                )

                # get the deformation field and apply it to the grid image
                def_field = itk.transformix_deformation_field(denoised_mov, result_transform_parameters)
                os.remove("deformationField.raw")
                os.remove("deformationField.mhd")
                def_field = np.asarray(def_field).astype(np.float32)
                registration_image_data["deformation_field"]["field"][i] = def_field
                # get the deformation grid
                grid_img = get_grid_image(info["img_size"], 6)
                grid_img_itk = itk.GetImageFromArray(grid_img)
                grid_img_transformed = itk.transformix_filter(grid_img_itk, result_transform_parameters)
                grid_img_transformed_np = itk.GetArrayFromImage(grid_img_transformed)
                # grid_img_transformed_np[grid_img_transformed_np < 0.5] = 0
                # grid_img_transformed_np[grid_img_transformed_np >= 0.5] = 1.0
                registration_image_data["deformation_field"]["grid"][i] = grid_img_transformed_np
                # finally apply the deformation field to the moving image (without denoising)
                if settings["complex_data"]:
                    # complex data registration
                    c_real = np.multiply(mov, np.cos(mov_phase))
                    c_imag = np.multiply(mov, np.sin(mov_phase))
                    c_real = itk.GetImageFromArray(c_real)
                    c_imag = itk.GetImageFromArray(c_imag)
                    img_reg_real = itk.transformix_filter(c_real, result_transform_parameters)
                    img_reg_imag = itk.transformix_filter(c_imag, result_transform_parameters)
                    img_reg_real = itk.GetArrayFromImage(img_reg_real)
                    img_reg_imag = itk.GetArrayFromImage(img_reg_imag)
                    img_reg = np.sqrt(np.square(img_reg_real) + np.square(img_reg_imag))
                    img_phase_reg = np.arctan2(img_reg_imag, img_reg_real)
                else:
                    # magnitude only data registration
                    mov = itk.GetImageFromArray(mov)
                    img_reg = itk.transformix_filter(mov, result_transform_parameters)
                    img_reg = itk.GetArrayFromImage(img_reg)

            # basic quick rigid
            elif settings["registration"] == "quick_rigid":
                shift, error, diffphase = phase_cross_correlation(
                    ref,
                    mov,
                    upsample_factor=100,
                    disambiguate=True,
                    reference_mask=mask,
                    moving_mask=mask,
                    overlap_ratio=0.5,
                )
                if settings["complex_data"]:
                    # complex data registration
                    c_real = np.multiply(mov, np.cos(mov_phase))
                    c_imag = np.multiply(mov, np.sin(mov_phase))
                    c_complex = c_real + 1j * c_imag
                    img_complex_reg = fourier_shift(np.fft.fftn(c_complex), shift)
                    img_complex_reg = np.fft.ifftn(img_complex_reg)
                    img_reg = np.abs(img_complex_reg)
                    img_phase_reg = np.arctan2(np.imag(img_complex_reg), np.real(img_complex_reg))
                else:
                    # magnitude only data registration
                    img_reg = fourier_shift(np.fft.fftn(mov), shift)
                    img_reg = np.abs(np.fft.ifftn(img_reg))

            # none
            elif settings["registration"] == "none":
                img_reg = mov
                if settings["complex_data"]:
                    img_phase_reg = mov_phase

            else:
                logger.error("No method available for registration: " + settings["registration"])
                raise ValueError("Registration method not available: " + settings["registration"])

            # store registered image
            # correct for registration small errors
            img_reg = np.asarray(img_reg, dtype=np.float32)
            img_reg[img_reg < 0] = 0
            if settings["complex_data"]:
                img_phase_reg = np.asarray(img_phase_reg, dtype=np.float32)
                img_phase_reg[img_phase_reg < -np.pi] = -np.pi
                img_phase_reg[img_phase_reg > np.pi] = np.pi

            # replace the images in the dataframe with all slices
            data.at[data.index[i], "image"] = img_reg
            if settings["complex_data"]:
                data.at[data.index[i], "image_phase"] = img_phase_reg

            # store images post registration
            registration_image_data["img_post_reg"][i] = img_reg

    return data, registration_image_data


def get_ref_image(current_entries: pd.DataFrame, slice_idx: int, settings: dict, logger: logging.Logger) -> dict:
    """
    Get all the lower b-value images, and groupwise register them if more than one.
    Then get the mean. That will be our reference image for the subsequent registration.
    If the registration is groupwise, then we do not need to calculate a reference.

    Args:
        current_entries: dataframe with the current entries
        slice_idx: index of the slice
        settings: settings dictionary
        logger: logger for logging

    Returns:
        ref_images: dict with reference image
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

        if (
            n_images < 2
            or settings["registration_reference_method"] == "best"
            or settings["registration_reference_method"] == "first"
        ):

            if n_images < 2:
                logger.info(
                    "Slice "
                    + str(slice_idx).zfill(2)
                    + ": only one image found for the lowest b-value, using that image as reference"
                )

            # stack all possible reference images
            image_stack = np.stack(current_entries["image"][index_pos].values)
            image_stack_sum = np.sum(image_stack, axis=(1, 2))

            if settings["registration_reference_method"] == "best":
                logger.info(
                    "Slice "
                    + str(slice_idx).zfill(2)
                    + ": using the brightest image as reference, registration_reference_method = best"
                )
                # get the image with the most signal
                c_img = current_entries.at[index_pos[np.argmax(image_stack_sum)], "image"]

            elif settings["registration_reference_method"] == "first":
                logger.info(
                    "Slice "
                    + str(slice_idx).zfill(2)
                    + ": using the first image as reference, registration_reference_method = first"
                )
                # get the first image
                c_img = current_entries.at[index_pos[0], "image"]

            # normalise 0 to 1
            c_img = (c_img - np.min(c_img)) / (np.max(c_img) - np.min(c_img))

            # denoise image
            denoised_img = denoise_img_nlm(c_img)

            ref_images["image"] = denoised_img
            ref_images["index"] = index_pos[np.argmax(image_stack_sum)]
            ref_images["n_images"] = len(current_entries)
            ref_images["groupwise_reg_info"] = {}

        elif settings["registration_reference_method"] == "groupwise":
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
                    os.path.dirname(os.path.dirname(__file__)),
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

            # denoise stack before masking
            for i in range(image_stack.shape[0]):
                image_stack[i] = denoise_img_nlm(image_stack[i])

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


def plot_ref_images(
    data: pd.DataFrame, ref_images: dict, mask: NDArray, contour: NDArray, slices: NDArray, settings: dict
):
    """
    Plot reference images and registration masks for debug purposes

    Args:
        data: dataframe with diffusion info
        ref_images: dictionary with all the info on the reference images used
        mask: registration mask
        contour: registration mask contours
        slices: array with strings of slice positions
        settings: settings dictionary
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
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            plt.close()

        # plot registration mask
        for slice_idx in slices:
            current_entries = data.loc[data["slice_integer"] == slice_idx]
            c_img_stack = np.stack(current_entries["image"].values)
            c_img_mean = np.mean(c_img_stack, axis=0)

            plt.figure(figsize=(5, 5))
            plt.imshow(c_img_mean, cmap="Greys_r")
            plt.plot(contour[:, 0], contour[:, 1], "r")
            plt.axis("off")
            plt.savefig(
                os.path.join(settings["debug_folder"], "registration_masks_slice_" + str(slice_idx).zfill(2) + ".png"),
                dpi=100,
                bbox_inches="tight",
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
                    plt.imshow(img_pre[i], vmin=np.min(img_pre[i]), vmax=np.max(img_pre[i]) * 0.3, cmap="Greys_r")
                    plt.title(str(i) + "_pre")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + n_images + 1)
                    plt.imshow(img_reg[i], vmin=np.min(img_reg[i]), vmax=np.max(img_reg[i]) * 0.3, cmap="Greys_r")
                    plt.title(str(i) + "_reg")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + 2 * n_images + 1)
                    plt.imshow(
                        abs(c_ref - img_pre[i]),
                        vmin=np.min(abs(c_ref - img_pre[i])),
                        vmax=np.max(abs(c_ref - img_pre[i])) * 0.3,
                        cmap="Greys_r",
                    )
                    plt.title("diff_to_ref")
                    plt.axis("off")
                    plt.subplot(4, n_images, i + 3 * n_images + 1)
                    plt.imshow(
                        abs(c_ref - img_pre[i]),
                        vmin=np.min(abs(c_ref - img_reg[i])),
                        vmax=np.max(abs(c_ref - img_reg[i])) * 0.3,
                        cmap="Greys_r",
                    )
                    plt.title("diff_to_ref")
                    plt.axis("off")
                plt.tight_layout(pad=1.0)
                plt.savefig(
                    os.path.join(
                        settings["debug_folder"],
                        "groupwise_registration_reference_slice_" + str(slice_idx).zfill(2) + ".png",
                    ),
                    dpi=100,
                    pad_inches=0,
                    transparent=False,
                )
                plt.close()


def get_registration_mask(info: dict, settings: dict) -> tuple[NDArray, NDArray]:
    """
    Define registration mask. A circular region from the centre of the FOV.
    Radius is scaled by settings["registration_mask_scale"]. A scale of 1 gives us
    a diameter equal to the shortest dimension of the image.

    Args:
        info: dictionary with image information
        settings: dictionary with settings

    Returns:
        mask: registration mask
        contour: registration mask contours

    """

    # create a circular mask for the registration
    shortest_dim = np.min(info["img_size"])
    img_centre = [int(info["img_size"][0] / 2) - 1, int(info["img_size"][1] / 2) - 1]
    Y, X = np.ogrid[: info["img_size"][0], : info["img_size"][1]]
    dist_from_center = np.sqrt((Y - img_centre[0]) ** 2 + (X - img_centre[1]) ** 2)
    border = shortest_dim * 0.5 * settings["registration_mask_scale"]
    mask = dist_from_center <= border
    mask = np.asarray(mask, dtype=np.ubyte)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = np.squeeze(contours[0])

    return mask, contour


def image_registration(
    data: pd.DataFrame, slices: NDArray, info: dict, settings: dict, logger: logging.Logger
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Image registration

    Args:
        data: data to be registered
        slices: array with slice integer
        info: dictionary with useful info
        settings: dict with settings
        logger: logger

    Returns:
        data: registered dataframe
        registration_image_data: images pre- and post-registration
        ref_images: reference images
        reg_mask: registration mask
    """

    # Registration is going to be a loop over each slice

    # reference images information dictionary
    ref_images = {}
    for slice_idx in slices:
        reg_file_reference = os.path.join(
            settings["session"], "image_registration_reference_slice_" + str(slice_idx).zfill(2) + ".npz"
        )

        # check if the reference images have been saved already
        if not os.path.exists(reg_file_reference):
            logger.info("No saved reference image found for slice " + str(slice_idx).zfill(2))

            # get reference image, index position and number of images per slice
            # dataframe for each slice
            current_entries = data.loc[data["slice_integer"] == slice_idx]
            # get the reference image
            ref_images[slice_idx] = get_ref_image(current_entries, slice_idx, settings, logger)

            # save reference images
            save_path = reg_file_reference
            np.savez_compressed(save_path, ref_images=ref_images[slice_idx])

        else:
            logger.info("Saved reference images found for slice " + str(slice_idx).zfill(2))
            # load reference images
            save_path = reg_file_reference
            npzfile = np.load(save_path, allow_pickle=True)
            ref_images[slice_idx] = npzfile["ref_images"].item()

    # get registration mask
    reg_mask, contour = get_registration_mask(info, settings)

    # plot reference images and registration mask
    plot_ref_images(data, ref_images, reg_mask, contour, slices, settings)

    # image registration based on the reference images
    registration_image_data = {}
    for slice_idx in slices:
        logger.info("Slice " + str(slice_idx).zfill(2) + ": Starting image registration")

        # check if registration has been done already
        reg_file = os.path.join(
            settings["session"], "image_registration_data_slice_" + str(slice_idx).zfill(2) + ".zip"
        )
        reg_file_extras = os.path.join(
            settings["session"], "image_registration_extras_slice_" + str(slice_idx).zfill(2) + ".npz"
        )

        if not os.path.exists(reg_file) or not os.path.exists(reg_file_extras):
            logger.info("No saved registration found for slice " + str(slice_idx).zfill(2))
            logger.info("Registration type: " + settings["registration"])

            # dataframe for this slice
            current_entries = data.loc[data["slice_integer"] == slice_idx]

            # run the registration loop
            current_entries, registration_image_data[slice_idx] = registration_loop(
                current_entries, ref_images[slice_idx], reg_mask, info, settings, logger
            )

            # table with current slice
            data.loc[data["slice_integer"] == slice_idx] = current_entries

            # saving registration data
            save_path = reg_file
            # table with only filename, image, acquisition time and date
            if settings["complex_data"]:
                data_basic = current_entries[["file_name", "image", "image_phase", "acquisition_date_time"]]
            else:
                data_basic = current_entries[["file_name", "image", "acquisition_date_time"]]
            data_basic.to_pickle(save_path, compression={"method": "zip", "compresslevel": 9})
            # saving registration extras
            np.savez_compressed(
                reg_file_extras,
                registration_image_data=registration_image_data[slice_idx],
            )

        else:
            logger.info("Saved registration found for slice " + str(slice_idx).zfill(2))

            # loading registration data
            data_loaded_basic = pd.read_pickle(reg_file)

            # current non registered database
            data_basic = data.loc[data["slice_integer"] == slice_idx]
            if settings["complex_data"]:
                data_basic = data_basic[["file_name", "image", "image_phase", "acquisition_date_time"]]
            else:
                data_basic = data_basic[["file_name", "image", "acquisition_date_time"]]

            # check if the original data basic table matches the loaded one (except the image column)
            if settings["complex_data"]:
                if not data_basic.drop(columns=["image", "image_phase"]).equals(
                    data_loaded_basic.drop(columns=["image", "image_phase"])
                ):
                    logger.error("Loaded Dataframe with registered images does not match pre-registered Dataframe!")
                    logger.error("Registration saved data needs to be deleted as something changed!")
                    raise ValueError(
                        "Loaded Dataframe with registered images does not match pre-registered Dataframe!"
                    )
            else:
                if not data_basic.drop(columns=["image"]).equals(data_loaded_basic.drop(columns=["image"])):
                    logger.error("Loaded Dataframe with registered images does not match pre-registered Dataframe!")
                    logger.error("Registration saved data needs to be deleted as something changed!")
                    raise ValueError(
                        "Loaded Dataframe with registered images does not match pre-registered Dataframe!"
                    )

            logger.info("Passed data consistency check. Loading registered data.")
            # data matches, so now I need to replace the image column with the loaded one
            data.loc[data["slice_integer"] == slice_idx, "image"] = data_loaded_basic["image"]
            if settings["complex_data"]:
                data.loc[data["slice_integer"] == slice_idx, "image_phase"] = data_loaded_basic["image_phase"]

            # also load the extra saved data
            npzfile = np.load(reg_file_extras, allow_pickle=True)
            registration_image_data[slice_idx] = npzfile["registration_image_data"].item()
            logger.info("Image registration loaded")

    return data, registration_image_data, ref_images, reg_mask
