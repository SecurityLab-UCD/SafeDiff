# Related Experiments

This directory contains related experiments conducted for the SafeDiff project,
namely
* Safe Latent Diffusion (sld)
* Erasing Concepts from Diffusion Models (erasing)

## Environments

There are two files that configures the environment:
* `sld_requirements.txt` contains requirements for running `sld` (specifically
  `run_sld.py`). A pip requirement file is used since SLD is uploaded to PyPi as
  a `diffusers` pipeline.
* `erasing_env.yml` is the conda environment for running `erasing` (which
  happens in their own repository).
