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
- [How to](#how-to)

## Introduction

This software is a post-processing pipeline designed for in-vivo cardiac diffusion tensor imaging.
It currently accepts Siemens and Philips diffusion weighted DICOM data, as well as [anonymised NIFTI data](https://github.com/ImperialCollegeLondon/cdti_data_export).
It also supports both STEAM and spin-echo data.

After the data is loaded, the pipeline performs the following steps:

- Image registration
- Image curation
- Tensor fitting
- Segmentation
- Export results

![workflow](assets/images/summary_figure.png)

INDI runs from the command line, and when processing a dataset for the first time,
INDI will require user input (pop-up matplotlib windows) which will be saved for future runs.

For more details:

See [documentation](docs/documentation.md) for details on the post-processing pipeline.

See [YAML settings](docs/YAML_settings.md) for run configuration details.

## Installation

Software has been tested on:

- macOS 15.0 with python 3.12
- Ubuntu 22.04 with python 3.12
- Windows 11 with python 3.12

### Clone the repository

[Install git for your OS](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Then clone the repository.

```bash
git clone https://github.com/ImperialCollegeLondon/INDI.git
```

#### Installation in macOS (Intel and Apple silicon) with pip

You may need to instal Xcode and Xcode’s Command Line Tools package as well, with this command:

```bash
xcode-select --install
```

Then install [homebrew](https://brew.sh/).

With homebrew install python 3.12:

```bash
brew install python@3.12
```

Also install imagemagick:

```bash
brew install imagemagick
```

Install the python environment in the INDI root directory:

```bash
python@3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

#### Installation in Ubuntu 22.04 with pip

Install Python 3.12

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12
sudo apt install python3.12-venv
```

Also install imagemagick:

```bash
brew install imagemagick
```

Install the python environment in the INDI root directory:

```bash
python@3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pip-tools
pip install -r requirements.txt
```

#### Installation Windows 11 with conda

Install miniforge:
[Miniforge](https://github.com/conda-forge/miniforge)

Install the python environment with conda:

```bash
conda env create -f environment-cpu.yml
```

Or alternatively, if you have a CUDA compatible GPU for Win or Linux:

```bash
conda env create -f environment-gpu.yml
```

Install [imagemagick](https://imagemagick.org/).

### Development

For development, also install the git hook scripts:

```bash
pre-commit install
```

Now pre-commit will run automatically on git commit. You can also run it manually with:

```bash
pre-commit run --all-files
```

This is required to ensure code quality and style before committing code changes.

## Basic usage example

We are going to post-process a synthetic phantom dataset with non-rigid distortions. Please unzip [the phantom data](docs/test_phantom_cdti_dicoms.zip).

The `test_phantom_cdti_dicoms` folder contains a subfolder named `diffusion_images` with the cdti simulated DICOMs. The DICOM files contain noisy diffusion weighted images with periodic non-rigid distortions, simulating a typical in-vivo scan.

INDI always looks recursively for subfolders named `diffusion_images`. The DICOM files must be inside this folder.

Before running we should have a look at the `settings.yaml` file, and check if the parameters makes sense. See [YAML settings](docs/YAML_settings.md) for more information. For this phantom example, the default settings should be fine.

Then run in the INDI python environment:

```bash
python main_script.py <data_path>
```

Where `<data_path>` is a folder that must contain at least a subfolder named `diffusion_images` with all the
DICOM files.

In the videio tutorial below we show how to run INDI with the phantom data:

[![Watch the video tutorial](assets/images/indi_tutorial_movie_screenshot.png)](https://1drv.ms/v/s!Ah-7Qw9tn52siW8SQZYX0RjRPdKG?e=Pwq85B)
