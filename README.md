# INDI (in-vivo diffusion)

<p align="center">
<img src="assets/images/sa_e1_small.png">
</p>

<p align="center">
In-vivo diffusion analysis (INDI)<br>
Post-processing pipeline for in-vivo cardiac diffusion tensor imaging.
</p>


## Table of Contents

- [Requirements](#Requirements)
- [Installation](#Installation)
- [Run](#Run)
- [Documentation](#Documentation)

## Requirements

### Download AI models

Download the U-Net and Tranformer models from the following link:

[One drive link](https://imperiallondon-my.sharepoint.com/:f:/g/personal/pferreir_ic_ac_uk/EtbqXB1XJY9JmBJ8kFcT40sBq9qHJrVZPwrzgEcW12VwUQ?e=qqDY8C)

U-Net models need to be copied to the following path:
```/usr/local/dtcmr/unet_ensemble/```

Tranformer models need to be copied to the following path:
```/usr/local/dtcmr/transformer_tensor_denoising/```

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

For development, also install the git hook scripts:

```bash
pre-commit install
```

Now pre-commit will run automatically on git commit. You can also run it manually with:

```bash
pre-commit run --all-files
```

You will also need to create a file in the root directory called `.env` with the following content:

```bash
ARCHIVE_PASS=your_password
``` 
This password will be used to encrypt DICOM files when using the option `workflow_mode: anon`.

## Run

Configure the `settings.yaml` file with the correct paths and parameters. 
See [YAML settings](docs/YAML_settings.md) for more information.

Then run:

```python main_script.py```

## Documentation

See [Pipeline](docs/Pipeline.md) for details on the post-processing pipeline.

See [YAML settings](docs/YAML_settings.md) for more information.








