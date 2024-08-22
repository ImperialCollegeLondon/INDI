import os
import pathlib

import itk
import numpy as np
from tqdm import tqdm

from extensions.extension_base import ExtensionBase


class RegistrationExVivo(ExtensionBase):
    def __init__(self, context, settings, logger):
        ExtensionBase.__init__(self, context, settings, logger)

        self.code_path = pathlib.Path(self.settings["code_path"])

    def run(self):
        data = self.context["data"]

        reg_images = {}
        for slice in data["slice_integer"].unique():
            reg_file_reference = os.path.join(
                self.settings["session"], "image_registration_reference_slice_" + str(slice).zfill(2) + ".npz"
            )
            # check if the reference images have been saved already
            if not os.path.exists(reg_file_reference):
                self.logger.info("No saved registration image found for slice " + str(slice).zfill(2))

            else:
                self.logger.info("Saved registration images found for slice " + str(slice).zfill(2))
                save_path = reg_file_reference
                npzfile = np.load(
                    save_path, allow_pickle=True
                )  # this is saved as a dictionary where it should be a numpy archive
                try:
                    reg_images[slice] = npzfile["img_post_reg"]
                    continue
                except KeyError:
                    self.logger.info("No registered images found for slice " + str(slice).zfill(2))

            images = np.asarray(data[data["slice_integer"] == slice]["image"])
            phase_images = None
            if self.settings["complex_data"]:
                phase_images = data[data["slice_integer"] == slice]["image_phase"].values

            # Get reference image
            ref_image = data.loc[
                data["b_value"] == np.sort(data[data["slice_integer"] == slice]["b_value"])[0], "image"
            ].values[0]
            if self.settings["complex_data"]:
                ref_image_phase = data.loc[
                    data["b_value"] == np.sort(data[data["slice_integer"] == slice]["b_value"])[0], "image_phase"
                ].values[0]
                ref_image = ref_image * np.exp(1j * ref_image_phase)

            indices = data[data["slice_integer"] == slice]["index"].values
            self.logger.info(f"Rigid registering slice {slice}")

            assert len(images) == len(indices)

            registered_images = self._register(
                ref_image,
                images,
                phase_images,
                indices,
                self.code_path / "extensions" / "image_registration_recipes" / "Elastix_rigid.txt",
            )

            # Averaging the repetitions
            average_image = []
            phase_images = []
            for index in np.unique(indices):
                if self.settings["complex_data"]:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_image_real = np.mean([img[0] for img in registered_images_index], axis=0)
                    average_image_imag = np.mean([img[1] for img in registered_images_index], axis=0)
                    average_image.append(np.sqrt(np.square(average_image_real) + np.square(average_image_imag)))
                    phase_images.append(np.arctan2(average_image_imag, average_image_real))
                else:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_image.append(np.mean(registered_images_index, axis=0))

            self.logger.info(f"BSpline registering slice {slice}")
            registered_images = self._register(
                ref_image,
                average_image,
                phase_images,
                indices,
                self.code_path / "extensions" / "image_registration_recipes" / "Elastix_bspline.txt",
            )

            average_image = []
            phase_images = []
            for index in np.unique(indices):
                if self.settings["complex_data"]:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_image_real = np.mean([img[0] for img in registered_images_index], axis=0)
                    average_image_imag = np.mean([img[1] for img in registered_images_index], axis=0)
                    average_image.append(np.sqrt(np.square(average_image_real) + np.square(average_image_imag)))
                    phase_images.append(np.arctan2(average_image_imag, average_image_real))
                else:
                    registered_images_index = [img for img, idx in registered_images if idx == index]
                    average_image.append(np.mean(registered_images_index, axis=0))

            reg_images[slice] = (
                registered_images * np.exp(1j * phase_images) if self.settings["complex_data"] else images
            )
            np.savez(
                reg_file_reference,
                img_post_reg=registered_images * np.exp(1j * phase_images)
                if self.settings["complex_data"]
                else images,
            )

    def _register(self, ref_image, images, phase_images, indices, recipe):
        ref_image = itk.GetImageFromArray(ref_image)

        # Denoise the images ?
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(recipe.as_posix())

        registered_images = []

        for i in tqdm(range(1, len(images)), desc="Registering images"):
            mov_image = images[i]

            if self.settings["complex_data"]:
                mov_image_real = mov_image * np.cos(phase_images[i])
                mov_image_imag = mov_image * np.sin(phase_images[i])

                img_reg_real, _ = itk.elastix_registration_method(
                    ref_image.real,
                    itk.GetImageFromArray(mov_image_real),
                    parameter_object=parameter_object,
                    log_to_console=False,
                )

                img_reg_imag, _ = itk.elastix_registration_method(
                    ref_image.imag,
                    itk.GetImageFromArray(mov_image_imag),
                    parameter_object=parameter_object,
                    log_to_console=False,
                )
                registered_images.append((img_reg_real, img_reg_imag, indices[i]))
            else:
                img_reg, _ = itk.elastix_registration_method(
                    ref_image,
                    itk.GetImageFromArray(mov_image),
                    parameter_object=parameter_object,
                    log_to_console=False,
                )
                registered_images.append((img_reg, indices[i]))

        return registered_images
