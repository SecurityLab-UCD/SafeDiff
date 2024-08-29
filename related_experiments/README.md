# Related Experiments

This directory contains related experiments conducted for the SafeDiff project,
namely
* Safe Latent Diffusion (SLD)
* Erasing Concepts from Diffusion Models (ERASING)

## Environments

There are two files that configures the environment:
* `sld_requirements.txt` contains requirements for running SLD (specifically
  `run_sld.py`). A pip requirement file is used since SLD is uploaded to PyPi as
  a `diffusers` pipeline.
* `erasing_env.yml` is the conda environment for running ERASING (which
  happens in their own repository).

## Scripts

* `run_sld.py` use SLD to  generate images from a dataset
* `format_input.py` adds the `case_number` and `evaluation_seed` columns needed
  for ERASING
