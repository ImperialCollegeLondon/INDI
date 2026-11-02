import math

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    def prange(*args):
        return range(*args)


def _validate_inputs(images, patch_radius, search_radius, h, center_weight, eps):
    images = np.asarray(images, dtype=np.float64)
    if images.ndim not in (3, 4):
        raise ValueError("images must have shape (n_images, H, W) or (n_images, H, W, D)")
    if patch_radius < 0:
        raise ValueError("patch_radius must be >= 0")
    # search_radius validation is deferred to _normalize_search_radius
    if h <= 0:
        raise ValueError("h must be > 0")
    if center_weight <= 0:
        raise ValueError("center_weight must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return images


def _to_scalar_float(value, name):
    # some callers (e.g. dipy's estimate_sigma) return a size-1 array instead of a plain float
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar (got array of size {array.size})")
    return float(array.reshape(-1)[0])


def _normalize_search_radius(search_radius, ndim):
    """Convert search_radius to a tuple of per-dimension radii.

    Parameters
    ----------
    search_radius : int or sequence of ints
        If int, applied to all spatial dimensions.
        If sequence, must have length equal to number of spatial dimensions.
    ndim : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    tuple of ints
        Per-dimension search radii.
    """
    if isinstance(search_radius, (int, np.integer)):
        return tuple([int(search_radius)] * ndim)
    else:
        search_radius = tuple(int(r) for r in search_radius)
        if len(search_radius) != ndim:
            raise ValueError(
                f"search_radius must be int or sequence of length {ndim} "
                f"(got sequence of length {len(search_radius)})"
            )
        return search_radius


def _pad_spatial(images, pad):
    """Pad image array.

    Parameters
    ----------
    images : ndarray
        Input images.
    pad : int or sequence of ints
        Padding per spatial dimension. If int, applied to all.

    Returns
    -------
    padded : ndarray
        Padded images.
    """
    if isinstance(pad, (int, np.integer)):
        pad_width = [(0, 0)] + [(pad, pad)] * (images.ndim - 1)
    else:
        pad_width = [(0, 0)] + [(p, p) for p in pad]
    return np.pad(images, pad_width, mode="reflect")


@njit(cache=True, inline="always")
def _patch_dist2_2d(padded, ref_img, cand_img, py, px, cy, cx, patch_radius):
    dist2 = 0.0
    count = 0
    for dy in range(-patch_radius, patch_radius + 1):
        for dx in range(-patch_radius, patch_radius + 1):
            a = padded[ref_img, py + dy, px + dx]
            b = padded[cand_img, cy + dy, cx + dx]
            d = a - b
            dist2 += d * d
            count += 1
    return dist2 / count


@njit(cache=True, inline="always")
def _patch_dist2_3d(padded, ref_img, cand_img, pz, py, px, cz, cy, cx, patch_radius):
    dist2 = 0.0
    count = 0
    for dz in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                a = padded[ref_img, pz + dz, py + dy, px + dx]
                b = padded[cand_img, cz + dz, cy + dy, cx + dx]
                d = a - b
                dist2 += d * d
                count += 1
    return dist2 / count


@njit(cache=True, parallel=True)
def _collaborative_nlm_2d_numba(
    padded,
    out,
    n_images,
    height,
    width,
    pad,
    patch_radius,
    search_radius_y,
    search_radius_x,
    h,
    center_weight,
    eps,
):
    h2 = h * h
    total = n_images * height * width

    for linear_idx in prange(total):
        ref_img = linear_idx // (height * width)
        rem = linear_idx % (height * width)
        y = rem // width
        x = rem % width

        py = y + pad[0]
        px = x + pad[1]

        weighted_sum = 0.0
        weight_total = 0.0

        for cand_img in range(n_images):
            for oy in range(-search_radius_y, search_radius_y + 1):
                cy = py + oy
                for ox in range(-search_radius_x, search_radius_x + 1):
                    cx = px + ox
                    dist2 = _patch_dist2_2d(padded, ref_img, cand_img, py, px, cy, cx, patch_radius)
                    w = math.exp(-dist2 / h2)

                    if cand_img == ref_img and oy == 0 and ox == 0:
                        w *= center_weight

                    weighted_sum += w * padded[cand_img, cy, cx]
                    weight_total += w

        out[ref_img, y, x] = weighted_sum / max(weight_total, eps)


@njit(cache=True, parallel=True)
def _collaborative_nlm_3d_numba(
    padded,
    out,
    n_images,
    depth,
    height,
    width,
    pad,
    patch_radius,
    search_radius_z,
    search_radius_y,
    search_radius_x,
    h,
    center_weight,
    eps,
):
    h2 = h * h
    total = n_images * depth * height * width

    for linear_idx in prange(total):
        ref_img = linear_idx // (depth * height * width)
        rem = linear_idx % (depth * height * width)
        z = rem // (height * width)
        rem2 = rem % (height * width)
        y = rem2 // width
        x = rem2 % width

        pz = z + pad[0]
        py = y + pad[1]
        px = x + pad[2]

        weighted_sum = 0.0
        weight_total = 0.0

        for cand_img in range(n_images):
            for oz in range(-search_radius_z, search_radius_z + 1):
                cz = pz + oz
                for oy in range(-search_radius_y, search_radius_y + 1):
                    cy = py + oy
                    for ox in range(-search_radius_x, search_radius_x + 1):
                        cx = px + ox
                        dist2 = _patch_dist2_3d(
                            padded,
                            ref_img,
                            cand_img,
                            pz,
                            py,
                            px,
                            cz,
                            cy,
                            cx,
                            patch_radius,
                        )
                        w = math.exp(-dist2 / h2)

                        if cand_img == ref_img and oz == 0 and oy == 0 and ox == 0:
                            w *= center_weight

                        weighted_sum += w * padded[cand_img, cz, cy, cx]
                        weight_total += w

        out[ref_img, z, y, x] = weighted_sum / max(weight_total, eps)


def _collaborative_nlm_2d_numpy(
    padded,
    out,
    n_images,
    height,
    width,
    pad,
    patch_radius,
    search_radius_y,
    search_radius_x,
    h,
    center_weight,
    eps,
):
    h2 = h * h
    pad_y, pad_x = pad
    for ref_img in range(n_images):
        for y in range(height):
            py = y + pad_y
            for x in range(width):
                px = x + pad_x
                ref_patch = padded[
                    ref_img,
                    py - patch_radius : py + patch_radius + 1,
                    px - patch_radius : px + patch_radius + 1,
                ]
                weighted_sum = 0.0
                weight_total = 0.0

                for cand_img in range(n_images):
                    for oy in range(-search_radius_y, search_radius_y + 1):
                        cy = py + oy
                        for ox in range(-search_radius_x, search_radius_x + 1):
                            cx = px + ox
                            cand_patch = padded[
                                cand_img,
                                cy - patch_radius : cy + patch_radius + 1,
                                cx - patch_radius : cx + patch_radius + 1,
                            ]
                            dist2 = np.mean((ref_patch - cand_patch) ** 2)
                            w = np.exp(-dist2 / h2)

                            if cand_img == ref_img and oy == 0 and ox == 0:
                                w *= center_weight

                            weighted_sum += w * padded[cand_img, cy, cx]
                            weight_total += w

                out[ref_img, y, x] = weighted_sum / max(weight_total, eps)


def _collaborative_nlm_3d_numpy(
    padded,
    out,
    n_images,
    depth,
    height,
    width,
    pad,
    patch_radius,
    search_radius_z,
    search_radius_y,
    search_radius_x,
    h,
    center_weight,
    eps,
):
    h2 = h * h
    pad_z, pad_y, pad_x = pad
    for ref_img in range(n_images):
        for z in range(depth):
            pz = z + pad_z
            for y in range(height):
                py = y + pad_y
                for x in range(width):
                    px = x + pad_x
                    ref_patch = padded[
                        ref_img,
                        pz - patch_radius : pz + patch_radius + 1,
                        py - patch_radius : py + patch_radius + 1,
                        px - patch_radius : px + patch_radius + 1,
                    ]
                    weighted_sum = 0.0
                    weight_total = 0.0

                    for cand_img in range(n_images):
                        for oz in range(-search_radius_z, search_radius_z + 1):
                            cz = pz + oz
                            for oy in range(-search_radius_y, search_radius_y + 1):
                                cy = py + oy
                                for ox in range(-search_radius_x, search_radius_x + 1):
                                    cx = px + ox
                                    cand_patch = padded[
                                        cand_img,
                                        cz - patch_radius : cz + patch_radius + 1,
                                        cy - patch_radius : cy + patch_radius + 1,
                                        cx - patch_radius : cx + patch_radius + 1,
                                    ]
                                    dist2 = np.mean((ref_patch - cand_patch) ** 2)
                                    w = np.exp(-dist2 / h2)

                                    if cand_img == ref_img and oz == 0 and oy == 0 and ox == 0:
                                        w *= center_weight

                                    weighted_sum += w * padded[cand_img, cz, cy, cx]
                                    weight_total += w

                    out[ref_img, z, y, x] = weighted_sum / max(weight_total, eps)


def collaborative_nlm(
    images,
    patch_radius=1,
    search_radius=2,
    h=10.0,
    center_weight=1.0,
    rician=False,
    sigma=0.0,
    eps=1e-12,
    use_numba=True,
):
    """
    Faster collaborative non-local means for aligned image stacks.

    Parameters
    ----------
    images : ndarray
        Shape (n_images, H, W) or (n_images, D, H, W).
        All images must be spatially aligned.
    patch_radius : int
        Radius of comparison patch. Patch size per dimension is (2r + 1).
    search_radius : int or sequence of ints
        Spatial search radius. If int, applied to all spatial dimensions.
        If sequence, must have length equal to number of spatial dimensions
        (2 for 3D images, 3 for 4D images).
    h : float
        Filtering parameter controlling weight decay.
    center_weight : float
        Extra multiplicative weight applied to the exact center candidate.
    rician : bool
        If True, apply simple Rician bias correction using sigma.
    sigma : float
        Noise standard deviation used only when rician=True.
    eps : float
        Small constant to avoid divide-by-zero.
    use_numba : bool
        If True and Numba is available, use JIT-compiled parallel kernels.

    Returns
    -------
    denoised : ndarray
        Denoised image stack with same shape as input.
    """
    h = _to_scalar_float(h, "h")
    sigma = _to_scalar_float(sigma, "sigma")
    images = _validate_inputs(images, patch_radius, search_radius, h, center_weight, eps)

    # Normalize search_radius to per-dimension values
    n_spatial_dims = images.ndim - 1
    search_radius_tuple = _normalize_search_radius(search_radius, n_spatial_dims)

    # Compute per-dimension padding
    pad = tuple(patch_radius + r for r in search_radius_tuple)
    padded = _pad_spatial(images, pad)
    out = np.zeros_like(images, dtype=np.float64)

    if images.ndim == 3:
        n_images, height, width = images.shape
        search_radius_y, search_radius_x = search_radius_tuple
        if use_numba and NUMBA_AVAILABLE:
            _collaborative_nlm_2d_numba(
                padded,
                out,
                n_images,
                height,
                width,
                pad,
                patch_radius,
                search_radius_y,
                search_radius_x,
                h,
                center_weight,
                eps,
            )
        else:
            _collaborative_nlm_2d_numpy(
                padded,
                out,
                n_images,
                height,
                width,
                pad,
                patch_radius,
                search_radius_y,
                search_radius_x,
                h,
                center_weight,
                eps,
            )
    else:
        n_images, depth, height, width = images.shape
        search_radius_z, search_radius_y, search_radius_x = search_radius_tuple
        if use_numba and NUMBA_AVAILABLE:
            _collaborative_nlm_3d_numba(
                padded,
                out,
                n_images,
                depth,
                height,
                width,
                pad,
                patch_radius,
                search_radius_z,
                search_radius_y,
                search_radius_x,
                h,
                center_weight,
                eps,
            )
        else:
            _collaborative_nlm_3d_numpy(
                padded,
                out,
                n_images,
                depth,
                height,
                width,
                pad,
                patch_radius,
                search_radius_z,
                search_radius_y,
                search_radius_x,
                h,
                center_weight,
                eps,
            )

    if rician:
        if sigma < 0:
            raise ValueError("sigma must be >= 0 when rician=True")
        out = np.sqrt(np.maximum(out * out - 2.0 * sigma * sigma, 0.0))

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_images, height, width = 4, 32, 32
    clean = np.zeros((n_images, height, width), dtype=np.float64)
    clean[:, 8:24, 10:22] = 100.0
    noisy = clean + rng.normal(0.0, 15.0, size=clean.shape)

    denoised = collaborative_nlm(
        noisy,
        patch_radius=1,
        search_radius=2,
        h=20.0,
        center_weight=1.2,
        use_numba=True,
    )

    print("Numba available:", NUMBA_AVAILABLE)
    print("Input shape:", noisy.shape)
    print("Output shape:", denoised.shape)
    print("Noisy mean/std:", noisy.mean(), noisy.std())
    print("Denoised mean/std:", denoised.mean(), denoised.std())
