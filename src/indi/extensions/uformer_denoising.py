import numpy as np
from numpy.typing import NDArray

from indi.extensions.extensions import crop_pad_rotate_array, reshape_tensor_from_6_to_3x3
from indi.extensions.uformer_tensor_denoising.uformer_tensor_denoising import main as uformer_main


def denoise_tensor(D: NDArray, settings: dict) -> NDArray:
    """Denoise a diffusion tensor field using the Uformer ensemble model.

    Pads or crops the spatial dimensions to 128x128, runs the Uformer inference
    pipeline, and restores the original shape.

    Args:
        D (NDArray): Diffusion tensor array with shape
            ``(n_slices, rows, cols, n_dirs, n_components)``.
        settings (dict): Configuration dict; must include
            ``"uformer_breatholds"`` (number of breath-holds used for
            acquiring the data: ``1``, ``3``, or ``5``).

    Returns:
        D_denoised (NDArray): Denoised tensor array with the same shape as ``D``.
    """

    # Make the tensor H & W [128, 128]
    initial_shape = D.shape
    new_shape = (initial_shape[0], 128, 128, initial_shape[3], initial_shape[4])
    D_new = crop_pad_rotate_array(D, new_shape, False)

    # Reorder the dimensions of the tensor (N, C, H, W)
    D_new = np.transpose(D_new, (0, 3, 4, 1, 2))

    # run uformer denoising
    breath_holds = settings["uformer_breatholds"]
    D_denoised = uformer_main(breath_holds, D_new)

    # revert back tensor to the usual dim order
    D_denoised = np.transpose(D_denoised, (0, 2, 3, 1))

    # convert last dim from 6 to 3x3
    D_denoised = reshape_tensor_from_6_to_3x3(D_denoised)

    D_denoised = crop_pad_rotate_array(D_denoised, initial_shape, False)

    return D_denoised
