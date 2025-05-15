import os

import numpy as np
from denoise.nlm_tensor import nlm_tensor

from indi.extensions.tensor_fittings import plot_tensor_components


def tensor_denoising(dti, slices, average_images, mask_3c, logger, settings):

    logger.debug("Tensor denoising with NLM")

    tensor = dti["tensor"]

    np.save(os.path.join(settings["debug_folder"], "tensor.npy"), tensor)

    tensor_denoised = nlm_tensor(tensor, h=1)

    dti["tensor"] = tensor_denoised

    if settings["debug"]:
        plot_tensor_components(
            tensor_denoised,
            average_images,
            mask_3c,
            slices,
            settings,
            "tensor_denoising_",
        )

    return dti
