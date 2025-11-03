
# INDI (In-Vivo Diffusion)

<p align="center">
<img src="assets/images/indi_logo.png">
</p>

<p align="center">
In-vivo diffusion analysis (INDI)<br>
A post-processing pipeline for in-vivo cardiac diffusion tensor imaging.
</p>

INDI is a Python-based post-processing pipeline designed for in-vivo cardiac diffusion tensor imaging (cDTI) data.
It supports Siemens and Philips diffusion-weighted DICOMs, as well as [anonymised NIFTI data](https://github.com/ImperialCollegeLondon/cdti_data_export). Both STEAM and spin-echo sequences are supported.

After loading your data, INDI performs the following steps:

- Image registration
- Image curation
- Tensor fitting
- Segmentation
- Results export

![workflow](assets/images/summary_figure.png)

INDI is run from the command line. When processing a dataset for the first time, user input may be required (via pop-up matplotlib windows); these selections are saved for future runs.

For more information:

- See the [documentation](documentation.md) for details on the post-processing pipeline (under development 🚧).
- See [YAML settings](YAML_settings.md) for configuration details (under development 🚧).

For installation instructions, usage examples, license information, and acknowledgements, please refer to the [README.md](https://github.com/ImperialCollegeLondon/INDI) file.

If you use this software, please credit this website and "**The CMR Unit, Royal Brompton Hospital**".

## Acknowledgements

- Royal Brompton Hospital (Guy's and St Thomas' NHS Foundation Trust), London, UK
- Imperial College London, UK
- Supported by the British Heart Foundation RG/19/1/34160 and RG/F/23/110115
- Chan Zuckerberg Initiative DAF, an advised fund of the Silicon Valley Community Foundation: 2024-337787
- EPSRC Healthcare Technologies EP/X014010/1

![funding](assets/images/grant_logos.png)
