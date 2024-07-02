# INDI (in-vivo diffusion)

<p align="center">
<img src="assets/images/sa_e1_small.png">
</p>

<p align="center">
In-vivo diffusion analysis (INDI)<br>
Post-processing pipeline for in-vivo cardiac diffusion tensor imaging.
</p>

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Run](#run)

## Introduction

This software is a post-processing pipeline designed to post-process in-vivo cardiac diffusion tensor imaging.
It currently accepts Siemens and Philips diffusion weighted DICOM data, as well as NIFTI data. It also supports both STEAM and SE sequences.

After the data is loaded, the pipeline performs the following steps:

- Image registration
- Image curation
- Tensor fitting
- Diffusion tensor fitting
- Results export

![alt text](assets/images/summary_figure.png)

INDI runs from the command line, and when processing a dataset for the first time, INDI will require user input (matplotlib windows) which will be saved for future runs.

For more details:

See [Pipeline documentation](docs/Pipeline.md) for details on the post-processing pipeline.

See [YAML settings](docs/YAML_settings.md) for run configuration.

## Installation

Software has been tested on macOS Sonoma with python 3.10.

### Installation in macOS (Intel and Apple silicon)

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

#### Development

For development, also install the git hook scripts:

```bash
pre-commit install
```

Now pre-commit will run automatically on git commit. You can also run it manually with:

```bash
pre-commit run --all-files
```

This is required to ensure code quality and style before committing code changes.

## Run

Configure the `settings.yaml` file with the correct paths and parameters.
See [YAML settings](docs/YAML_settings.md) for more information.

Then run:

```python main_script.py <data_path>```

Where `<data_path>` is a folder that must contain at least a subfolder named `diffusion_images` with all the diffusion weighted images inside. Multiple subfolders can be processed at once. For more detailed information see [documentation](docs/documentation.md).
