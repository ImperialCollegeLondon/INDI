"""
Read H5 file
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path_1 = "/Users/pf/WORK/WORK_2/CIMA_scans/initial_test_scans/test_elastix_randomseed/scan_1/Python_post_processing/results/data/DTI_maps.h5"
file_path_2 = "/Users/pf/WORK/WORK_2/CIMA_scans/initial_test_scans/test_elastix_randomseed/scan_2/Python_post_processing/results/data/DTI_maps.h5"
file_path_3 = "/Users/pf/WORK/WORK_2/CIMA_scans/initial_test_scans/test_elastix_randomseed/scan_3/Python_post_processing/results/data/DTI_maps.h5"


# read the h5 file
with h5py.File(file_path_1, "r") as h5:
    # List all groups
    # groups = list(h5.keys())
    # print("Groups:", groups)

    fa_1 = h5["fa"][()]
    ha_1 = h5["ha"][()]
    md_1 = h5["md"][()]


with h5py.File(file_path_2, "r") as h5:
    fa_2 = h5["fa"][()]
    ha_2 = h5["ha"][()]
    md_2 = h5["md"][()]

with h5py.File(file_path_3, "r") as h5:
    fa_3 = h5["fa"][()]
    ha_3 = h5["ha"][()]
    md_3 = h5["md"][()]

delta_fa_1 = fa_1 - fa_2
delta_fa_2 = fa_1 - fa_3
delta_fa_3 = fa_2 - fa_3

print("Delta FA (1 vs 2) mean and max:", np.nanmean(delta_fa_1), np.nanmax(delta_fa_1))
print("Delta FA (1 vs 3) mean and max:", np.nanmean(delta_fa_2), np.nanmax(delta_fa_2))
print("Delta FA (2 vs 3) mean and max:", np.nanmean(delta_fa_3), np.nanmax(delta_fa_3))

# plot the difference
plt.figure()
plt.subplot(131)
plt.imshow(delta_fa_1[0], cmap="hot")
plt.title("Delta FA 1 vs 2")
plt.colorbar()
plt.subplot(132)
plt.imshow(delta_fa_1[0], cmap="hot")
plt.title("Delta FA 1 vs 3")
plt.colorbar()
plt.subplot(133)
plt.imshow(delta_fa_1[0], cmap="hot")
plt.title("Delta FA 2 vs 3")
plt.colorbar()
plt.show()
