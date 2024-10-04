import logging

import numpy as np
from numpy.typing import NDArray


def get_ha_e2a_maps(
    mask: NDArray, local_cardiac_coordinates: dict, eigenvectors: NDArray, ventricle
) -> tuple[NDArray, NDArray]:
    """
    Calculate HA and E2A maps

    Parameters
    ----------
    mask: segmentation masks
    local_cardiac_coordinates: local cardiac coordinates
    (longitudinal, radial, circumferential)
    eigenvectors: array with eigenvectors

    Returns
    -------
    HA, E2A maps
    """

    ev1 = eigenvectors[:, :, :, :, 2]
    ev2 = eigenvectors[:, :, :, :, 1]

    if ventricle == "LV":
        coords = np.where(mask == 1)
    elif ventricle == "RV":
        coords = np.where(mask == 2)
    circ_vecs = local_cardiac_coordinates["circ"][coords]
    long_vecs = local_cardiac_coordinates["long"][coords]
    radi_vecs = local_cardiac_coordinates["radi"][coords]

    ha_map = np.zeros(mask.shape)
    ta_map = np.zeros(mask.shape)
    e2a_map = np.zeros(mask.shape)

    # HA calculation
    vector_to_project_ha = ev1[coords]

    # HA
    plane_ha = np.asarray([circ_vecs, long_vecs])
    plane_ha = np.transpose(plane_ha, (1, 2, 0))
    a = np.transpose(plane_ha.conj(), (0, 2, 1))
    ev1_proj = np.squeeze(plane_ha @ np.linalg.inv(a @ plane_ha) @ a @ vector_to_project_ha[..., np.newaxis])
    ev1_proj = np.divide(ev1_proj, np.linalg.norm(ev1_proj, axis=1)[:, np.newaxis])

    test_pos = np.sum(ev1_proj * circ_vecs, axis=1) < 0
    ev1_proj[test_pos] *= -1

    a = ev1_proj.conj()
    b = circ_vecs
    values = np.divide(((a * b).sum(1)), np.linalg.norm(a, axis=1))
    values = np.around(values, 10)
    helix_angle = np.rad2deg(np.arccos(values))

    # adjust HA to [-90 90]
    adjusted_ha = helix_angle.copy()
    adjusted_ha[(ev1_proj[:, 2] >= 0) & (adjusted_ha >= 90)] = -1 * (
        180 - adjusted_ha[(ev1_proj[:, 2] >= 0) & (adjusted_ha >= 90)]
    )
    adjusted_ha[(ev1_proj[:, 2] < 0) & (adjusted_ha >= 90)] = (
        180 - adjusted_ha[(ev1_proj[:, 2] < 0) & (adjusted_ha >= 90)]
    )
    adjusted_ha[(ev1_proj[:, 2] < 0) & (adjusted_ha < 90)] = -(adjusted_ha[(ev1_proj[:, 2] < 0) & (adjusted_ha < 90)])
    helix_angle = adjusted_ha
    ha_map[coords] = adjusted_ha

    # TA calculation
    # Project E1 in the circ radial plane and measure angle with the circ direction
    vector_to_project_ta = ev1[coords]
    plane_ta = np.asarray([circ_vecs, radi_vecs])
    plane_ta = np.transpose(plane_ta, (1, 2, 0))
    a = np.transpose(plane_ta.conj(), (0, 2, 1))
    ev1_proj_b = np.squeeze(plane_ta @ np.linalg.inv(a @ plane_ta) @ a @ vector_to_project_ta[..., np.newaxis])
    ev1_proj_b = np.divide(ev1_proj_b, np.linalg.norm(ev1_proj_b, axis=1)[:, np.newaxis])

    # align the projected vector with circ direction
    test_pos = np.sum(ev1_proj_b * circ_vecs, axis=1) < 0
    ev1_proj_b[test_pos] *= -1

    # angle between circ and ev1_proj_b
    a = ev1_proj_b.conj()
    b = circ_vecs
    values = np.divide(((a * b).sum(1)), np.linalg.norm(a, axis=1))
    values = np.around(values, 10)
    transverse_angle = np.rad2deg(np.arccos(values))

    # TA polarity
    # When along circ if also along radial is positive, if not along radial then it is negative
    # we know the projected vector is aligned with circ. Now we will find out if it is also aligned with radial
    # if it is not then we will flip the sign of the angle to negative
    test_pos = np.sum(ev1_proj_b * radi_vecs, axis=1) < 0
    transverse_angle[test_pos] *= -1

    ta_map[coords] = transverse_angle

    # E2A
    # cross fiber vector is defined as the vector perpendicular
    # to v1_proj and the radial vector
    cross_fibre_vector = np.cross(radi_vecs, ev1_proj)
    cross_fibre_vector = np.divide(cross_fibre_vector, np.linalg.norm(cross_fibre_vector, axis=1)[:, np.newaxis])

    # make the cross_fibre_vector always point towards the base
    test_pos = cross_fibre_vector[:, 2] < 0
    cross_fibre_vector[test_pos] *= -1

    # I have the two vectors needed to define the plane cross-fibre and radial
    # so now calculate the projection of the second eigenvector to this plane
    vector_to_project_e2a = ev2[coords]
    plane_e2a = np.asarray([radi_vecs, cross_fibre_vector])
    plane_e2a = np.transpose(plane_e2a, (1, 2, 0))
    a = np.transpose(plane_e2a.conj(), (0, 2, 1))

    ev2_proj = np.squeeze(plane_e2a @ np.linalg.inv(a @ plane_e2a) @ a @ vector_to_project_e2a[..., np.newaxis])
    ev2_proj = np.divide(ev2_proj, np.linalg.norm(ev2_proj, axis=1)[:, np.newaxis])

    # calculate the angle of ev2_proj to the radial component
    a = ev2_proj.conj()
    b = radi_vecs
    values = np.divide(((a * b).sum(1)), np.linalg.norm(a, axis=1))
    values = np.around(values, 10)
    e2a = np.rad2deg(np.arccos(values))

    # wrap E2 angles to [-90 90]
    adjusted_e2a = e2a.copy()

    adjusted_e2a[(ev2_proj[:, 2] >= 0) & (adjusted_e2a >= 90)] = -1 * (
        180 - adjusted_e2a[(ev2_proj[:, 2] >= 0) & (adjusted_e2a >= 90)]
    )
    adjusted_e2a[(ev2_proj[:, 2] < 0) & (adjusted_e2a < 90)] = -1 * (
        adjusted_e2a[(ev2_proj[:, 2] < 0) & (adjusted_e2a < 90)]
    )
    adjusted_e2a[(ev2_proj[:, 2] < 0) & (adjusted_e2a >= 90)] = (
        180 - adjusted_e2a[(ev2_proj[:, 2] < 0) & (adjusted_e2a >= 90)]
    )

    # We are defining the angle in relation to the wall
    adjusted_e2a[adjusted_e2a >= 0] = 90 - adjusted_e2a[adjusted_e2a >= 0]
    adjusted_e2a[adjusted_e2a < 0] = -90 - adjusted_e2a[adjusted_e2a < 0]

    e2a = adjusted_e2a
    e2a_map[coords] = e2a

    return ha_map, ta_map, e2a_map


def get_tensor_orientation_maps(
    slices: NDArray,
    mask_3c: NDArray,
    local_cardiac_coordinates: dict,
    dti: dict,
    settings: dict,
    logger: logging.Logger,
    ventricle="LV",
) -> tuple[NDArray, NDArray, dict]:
    """_summary_

    Parameters
    ----------
    slices : NDArray
        _description_
    mask_3c : NDArray
        _description_
    local_cardiac_coordinates : dict
        _description_
    dti : dict
        _description_
    settings : dict
        _description_
    info : dict
        _description_

    Returns
    -------
    [NDArray, NDArray, dict]
        _description_
    """

    ha, ta, e2a = get_ha_e2a_maps(mask_3c, local_cardiac_coordinates, dti["eigenvectors"], ventricle)

    if ventricle == "LV":
        # the orientation maps from above should be nan outside the LV re
        ha[mask_3c != 1] = np.nan
        ta[mask_3c != 1] = np.nan
        e2a[mask_3c != 1] = np.nan
    elif ventricle == "RV":
        ha[mask_3c != 2] = np.nan
        ta[mask_3c != 2] = np.nan
        e2a[mask_3c != 2] = np.nan

    # var_names = ["HA"]
    # for var in var_names:
    #     for i, slice_idx in enumerate(slices):
    #         vals = eval(var.lower())[i][mask_3c[i] == 1]

    if ventricle == "LV":
        var_names = ["E2A"]
        for var in var_names:
            for i, slice_idx in enumerate(slices):
                vals = np.abs(eval(var.lower())[i][mask_3c[i] == 1])
                logger.debug(
                    "Median "
                    + var
                    + " for slice "
                    + str(slice_idx).zfill(2)
                    + ": "
                    + "%.2f" % np.nanmedian(vals)
                    + " ["
                    + "%.2f" % np.nanquantile(vals, 0.25)
                    + ", "
                    + "%.2f" % np.nanquantile(vals, 0.75)
                    + "]"
                )

    return ha, ta, e2a
