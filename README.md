
# MuFoRa – A Multimodal Dataset of Traffic Elements Under Controllable and Measured Adverse Weather Conditions of Fog and Rain

[Download dataset](https://doi.org/10.5281/zenodo.18988561) | [Dowload paper](https://link.springer.com/content/pdf/10.1007/s42979-026-04976-9.pdf)

This repository provides code and resources to reproduce the experiments presented in the paper:

**MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather on Camera and LiDAR**

It is designed to help researchers quickly get started with dataset usage, benchmarking, and evaluation pipelines.

## Getting Started

To ensure full reproducibility of all experiments, we provide a complete, containerized development setup. To reproduce the results, you mus execute the following steps:

1. Download the datasets
2. Configure the `PATHS` variable so it points to the donwload/storage-location of the MuFoRa dataset
3. Re-open the reporitory inside the pre-configured devcontainers (see `.devcontainer/` folder)
4. Execute the scripts for the resuluts you want to reproduce

## Cite This Paper

```bibtex
@article{behret2026mufora,
  author    = {Behret, Valentino and Kushtanova, R. and Weber, S. and others},
  title     = {MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather on Camera and LiDAR},
  journal   = {SN Computer Science},
  volume    = {7},
  number    = {442},
  year      = {2026},
  doi       = {10.1007/s42979-026-04976-9}
}
```
