"""
Read H5 file
"""

import h5py
import matplotlib.pyplot as plt

file_path = "/Users/pf/WORK/WORK_1/dtcmr_dicom_data_examples/numerical_examples/5_slices/Python_post_processing/results/data/DTI_maps.h5"

# read the h5 file
with h5py.File(file_path, "r") as h5:
    # List all groups
    groups = list(h5.keys())
    print("Groups:", groups)

    fa = h5["fa"][()]
    e2a = h5["e2a"][()]
    ha = h5["ha"][()]
    md = h5["md"][()]
    ha_lp = h5["ha_line_profiles_1_lp_matrix"][()]
    lv_sectors = h5["lv_sectors"][()]


# plot some maps
plt.figure()
plt.subplot(221)
plt.imshow(fa[2, :, :], cmap="viridis")
plt.title("FA")
plt.colorbar()
plt.subplot(222)
plt.imshow(e2a[1, :, :], cmap="viridis")
plt.title("E2A")
plt.colorbar()
plt.subplot(223)
plt.imshow(ha[1, :, :], cmap="viridis")
plt.title("HA")
plt.colorbar()
plt.subplot(224)
plt.imshow(md[1, :, :], cmap="viridis")
plt.title("MD")
plt.colorbar()
plt.show()

plt.figure()
plt.plot(ha_lp.T)
plt.show()
