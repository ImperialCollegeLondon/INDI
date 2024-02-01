# INDI (in-vivo diffusion)

Python pipeline to post-process in-vivo cardiac diffusion data.

## Requirements

### Download AI models

Download the U-Net and Tranformer models from the following link:

[One drive link](https://imperiallondon-my.sharepoint.com/:f:/g/personal/pferreir_ic_ac_uk/EtbqXB1XJY9JmBJ8kFcT40sBq9qHJrVZPwrzgEcW12VwUQ?e=qqDY8C)

U-Net models need to be copied to the following path:
```/usr/local/dtcmr/unet_ensemble/```

Tranformer models need to be copied to the following path:
```/usr/local/dtcmr/transformer_tensor_denoising/```

### Installation

Software has been tested on macOS Sonoma with python 3.10.

#### Installation in macOS (Intel and Apple silicon)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

You also need ImageMagick installed. You can install it with [Homebrew](https://brew.sh/):

```bash
brew install imagemagick
```

For development, also install the git hook scripts:

```bash
pre-commit install
```

Now pre-commit will run automatically on git commit. You can also run it manually with:

```bash
pre-commit run --all-files
```


## Run

Configure the `settings.yaml` file with the correct paths and parameters.

Then run:

```python main_script.py```

## Documentation

## YAML settings
The yaml file controls the workflow.

### WORKFLOW
- **start_folder**: path to where any subfolder with the name `diffusion_images` will be processed.  
- **workflow_mode**:
  - **anon**: TO BE DONE. Converts the DICOM files to a dataframe file. No post-processing is done. Reads the DICOM files, imports the relevant data to a dataframe, saves the dataframe to a file with no identifying data, and then **deletes the DICOM files**. No subsequent processing is done.
  - **reg**: Runs image registration only. This is useful when batch processing.
  - **main**: Runs the full post-processing, may require manual input if processing for the first time.

### DEBUGGING
- **debug**: If True, more images are saved for debugging / quality control
- **registration_extra_debug**: If True, extra registration debug images are saved

### THRESHOLDING
- **threshold_strength**: Threshold strength for the thresholding step. 1.0 is the maximum thresholding, 0.0 is no thresholding. 0.5 has been found to work well for in-vivo data.

### IMAGE REGISTRATION
- **registration**: Type of registration to be used. Options are:
  - **none**: No registration is done
  - **quick_rigid**: Rigid registration using the `quick_rigid` method from `pystackreg`. This is a fast method, but not very accurate.
  - **elastix_rigid**: Rigid registration using the `elastix` method from `itk`.
  - **elastix_affine**: Affine registration using the `elastix` method from `itk`.
  - **elastix_non_rigid**: Non-rigid registration using the `elastix` method from `itk`.
  - **elastix_groupwise**: Groupwise registration using the `elastix` method from `itk`. **Not working at the moment**.

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




