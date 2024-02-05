# YAML settings

The yaml file controls the workflow of the pipeline.
There are three main workflows with this tool. This is controlled by the `workflow_mode` parameter as explained in the section below.

## WORKFLOWS

`start_folder = PATH`: path to where any subfolder with the name `diffusion_images` will be processed.

`workflow_mode = anon | reg | main`:

- `anon`: No post-processing is done. Reads the DICOM files and converts the relevant data to a pandas dataframe. No patient identifying information is stored. After this **all DICOM files are deleted**!

>[!WARNING]
> When using the `anon` mode, all DICOM files are deleted automatically. So make sure there is a backup of the data.

- `reg`: Runs image registration only. This is useful when batch processing many datasets, we can pre-register all the data and then run the full post-processing later with the option `main`.
- `main`: Runs the full post-processing, may require manual input if processing for the first time.

`sequence_type = steam | se`: define the type of sequence. If using STEAM, the b0 values are adjusted to a >0 value (defined by the sequence spoilers), and then all b-values are adjusted for RR variations from the assumed value set in the protocol.

---

## DEBUGGING

- `debug = True | False`: If True, more images are saved during the post-processing for debugging / quality control.
- `registration_extra_debug = True | False`: If True, extra registration debug images are saved.

---

## THRESHOLDING

- `threshold_strength = [0, 1.0]`: Threshold strength for the thresholding step. 1.0 is the maximum thresholding, 0.0 is no thresholding. 0.5 has been found to work well for in-vivo data.

---

## IMAGE REGISTRATION

Method of registration to be used:

`registration = none | quick_rigid | elastix_rigid | elastix_affine | elastix_non_rigid | elastix_groupwise`:

- `none`: No registration is done
- `quick_rigid`: Rigid registration similar to MATLAB's `imregister` function, although no cropping is done
- `elastix_rigid`: Rigid registration using the `elastix` method from `itk`.
- `elastix_affine`: Affine registration using the `elastix` method from `itk`.
- `elastix_non_rigid`: Non-rigid registration using the `elastix` method from `itk`.
- `elastix_groupwise`: Groupwise registration using the `elastix` method from `itk`. **Experimental, not working correctly at the moment.**.

Elastix non rigid is the slowest, but potentially the best. Registration needs to be checked for unwanted distortions. If present, elastix affine may be the safest alternative.

### IMAGE SEGMENTATION

- **manual_segmentation**: If True, user is prompted to do a manual segmentation with spline curves.

### TENSOR FITTING

- **tensor_fit_method**: Method used for tensor fitting. This is a library imported from DiPy. Options are:
  - **LS**: Least squares
  - **WLS**: Weighted least squares
  - **NLLS**: Non-linear least squares
  - **RESTORE**: RESTORE method

NLLS should be the default. RESTORE is an alternative that may be more robust to noise, outliers, but it is slower and may not work well if data quality is too low.

### IMAGE REMOVAL

- **remove_outliers_manually**: If True, user is prompted to assess and remove images manually.
- **remove_outliers_with_ai**: If True, images are removed automatically using a trained AI model. **Not working well at the moment**.

### AI MODELS

#### SEGMENTATION

- **u_net_segmentation**: If True, U-Net segmentation is performed automatically. If manual_segmentation is also True, the U-Net segmentation is used as a starting point for the manual segmentation.
- **n_ensemble**: Number of U-Net models to use for the segmentation. Options are 1 to 5.

#### TENSOR DENOISING

- **uformer_denoise**: If True, U-Former denoising is performed on the tensor data. This is a deep learning method that denoises the tensor data. It may improve the results when the data quality is low. The models have been trained on datasets with a fixed number of breatholds:
  - **uformer_breathholds**: Number of breathholds to use for the U-Former denoising. Options are 1, 3, or 5.
