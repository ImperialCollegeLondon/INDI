# YAML settings

>[!WARNING]
> This page in under development.

The yaml file controls the workflow of the pipeline.
There are three main workflows with this tool.
This is controlled by the `workflow_mode` parameter as explained in the section below.

## WORKFLOWS

`start_folder = PATH`: path to where any subfolder with the name `diffusion_images` will be processed.
We can alternatively define this path when calling the python script in the terminal.

`workflow_mode = anon | reg | main`:

- `anon`: No post-processing is done. Reads the DICOM files and converts the relevant data to a pandas dataframe with the protocol information, and an h5 file with the pixel values. No patient identifying information is stored. After this **all DICOM files are archived** in a 7zip file encrypted with a password!
The password is defined in the `.env` file.

>[!WARNING]
> The DICOM archive may be useful when developing new features. For safety and privacy, make sure there is a backup of the DICOM files somewhere safe and remove the DICOM archive if sharing data.

- `reg`: Runs image registration only. Image registration takes a considerable amount of time, and this option may be useful when batch processing many datasets, we can pre-register all the data and then run the remaining post-processing later with the option `main`.
  
- `main`: Runs the full post-processing, will require manual input if processing for the first time.

`sequence_type = steam | se`: define the type of sequence. If using STEAM, the b0 values are adjusted to a >0 value (defined by the sequence spoilers), and then all b-values are adjusted for RR variations from the assumed value set in the protocol.

---

## DEBUGGING

- `debug = True | False`: If True, more images are saved during the post-processing for debugging / quality control.
- `registration_extra_debug = True | False`: If True, extra registration debug images are saved.

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

Elastix non rigid is the slowest, but potentially the best. Registration needs to be checked for unwanted distortions.
If present, elastix affine may be the safest alternative.

`registration_speed = slow | fast`
When using an Elastix registration method (`elastix_rigid`, `elastix_affine` or `elastix_non_rigid`),
the speed of the registration can be controlled with this parameter.
The slow option is more accurate, but takes longer to run:

- slow: iterations = 2000; resolutions = 4
- fast: iterations = 256; resolutions = 2

When using the following registration methods:(`quick_rigid`, `elastix_rigid`, `elastix_affine` or `elastix_non_rigid`),
we need to define a reference image for the registration. There are two methods available:

- `first`: the first image with the lowest b-value is used as the reference.
- `groupwise`: a groupwise registration is performed with all lowest b-value images,
and the average image is used as the reference.

---

## TENSOR FITTING

Tensor fitting algorithm. This is a library imported from DiPy.

`tensor_fit_method: LS | WLS | NLLS | RESTORE`:

- `LS`: Least squares
- `WLS`: Weighted least squares
- `NLLS`: Non-linear least squares
- `RESTORE`: [RESTORE method](https://onlinelibrary.wiley.com/doi/10.1002/mrm.20426)

NLLS should be the default.
RESTORE is an alternative that may be more robust to noise or outliers,
but it is slower and may not work well if data quality is too low.

---

## IMAGE REMOVAL

Some diffusion weighted images are corrupted with motion induced signal loss.
These images need to be removed before tensor fitting.
This can be done manually or automatically with AI (not working at the moment)

- `remove_outliers_manually = True | False`: If True, user is prompted to assess and remove images manually.
- `remove_outliers_with_ai = True | False`: If True, images are removed automatically using a trained AI model. **Not working well at the moment**.

---

## MISC

`assumed_rr_interval = float` Define manually the assumed rr interval (msec). The default value is 1000 msec.
This value is only used if no value was found in the header `image_comments` and only useful for the STEAM sequence.

`calculated_real_b0 = float` True b-value of the b0 images. Similar to the parameter above,
this value is only used if no value was found in the header `image_comments` and only useful for the STEAM sequence.

---

## AI MODELS

Settings for the AI models used in the pipeline.

### LV SEGMENTATION

`u_net_segmentation = True | False`: If True, U-Net segmentation is performed automatically.
If manual_segmentation is also True, the U-Net segmentation is used as a starting point for the manual segmentation.

`n_ensemble = int [1, 5]`: Number of U-Net models to use for the segmentation using a naive ensemble approach.

### TENSOR DENOISING

`uformer_denoise = True | False`: If True, [U-Former denoising](https://link.springer.com/chapter/10.1007/978-3-031-12053-4_8)
is performed on the tensor data.
This is a deep learning method that denoises the tensor data.
It may improve the results when the data quality is low.
The models have been trained on STEAM datasets with a fixed number of breatholds:

`uformer_breathholds = 1 | 3 | 5`: Number of breathholds to use for the U-Former denoising.
