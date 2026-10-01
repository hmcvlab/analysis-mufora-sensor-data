# MuFoRa – A Multimodal Dataset of Traffic Elements Under Controllable and Measured Adverse Weather Conditions of Fog and Rain

[![Paper](https://img.shields.io/badge/Paper-10.1007%2Fs42979--026--04976--9-blue)](https://link.springer.com/article/10.1007/s42979-026-04976-9)
[![Dataset](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14175611-blue)](https://doi.org/10.5281/zenodo.14175611)

This repository contains the code to reproduce the sensor degradation
analysis presented in:

> **MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather on Camera
> and LiDAR**\
> Valentino Behret, Regina Kushtanova, Simon Weber, Julian Wiesner, Thomas
> Helmer, Frank Palme\
> *SN Computer Science*, 7, 442 (2026)\
> https://doi.org/10.1007/s42979-026-04976-9

It covers the camera/LiDAR weather-impact evaluation (Figures 7–11, A1 and
Tables A1–A3 of the paper). The circle detection benchmark (Table 6) lives in
the companion repository
[hmcvlab/model-ball-detection](https://github.com/hmcvlab/model-ball-detection).

## Dataset

The MuFoRa dataset is published on
[Zenodo](https://doi.org/10.5281/zenodo.14175611) (restricted access –
request access via the Zenodo page). It consists of three archives:

| Archive | Contents | Needed here? |
| --- | --- | --- |
| `MuFoRa_dataset.zip` | Camera images (`*.png`) and LiDAR point clouds (`*.pcd`) with per-folder `calib.yaml` | Yes – extract so that a `rawdata/` directory is created |
| `extrinsics_v2.zip` | Per-day camera–LiDAR extrinsics | Already mirrored in `data/calib/` |
| `avaerage_visibility_measurements.7z` | Visibility time series of the fog sensor | Already mirrored in `data/fog/` |

The 2D and 3D ground truth annotations of the spherical target are bundled
with this repository in `data/annotations/`.

Expected layout after extraction:

```
<dataset-root>/
└── rawdata/
    └── <YYYY-MM-DD>_<HH-MM-SS>_raw_{dry_light|dry_dark_with_car_light|fog|rain}_<dist>m[_<int>mm]/
        ├── <timestamp_ns>_left_image.png
        ├── <timestamp_ns>_right_image.png
        ├── <timestamp_ns>_qb2_0.pcd      # LiDAR Ls (sparse)
        ├── <timestamp_ns>_qb2_1.pcd      # LiDAR Ld (dense)
        └── calib.yaml
```

Point the `MUFORA_ROOT` environment variable at `<dataset-root>` (the
directory that contains `rawdata/`):

```bash
export MUFORA_ROOT=/path/to/mufora
```

Without `MUFORA_ROOT`, the fallback search paths in `mufora/data.py` are used.
Run `python scripts/check_setup.py` (or `make check`) to verify your setup.

## Installation

Python >= 3.10 is required. Two options:

1. **Dev container (recommended):** re-open the repository in one of the
   pre-configured dev containers (`.devcontainer/cpu` or
   `.devcontainer/gpu`, based on `hmcvlab/computer-vision:3.2.7`). The
   package is installed automatically via `postCreateCommand`.
2. **Plain pip:** `pip install -e .` (add `.[dev]` for the test dependency).

## Reproduce the paper results

In short:

```bash
make check       # verify dataset + annotation setup
make reproduce   # run the full analysis pipeline and all figure/table scripts
```

Each script can also be run individually and supports a `--debug` flag for a
quick smoke test on a small subset. The full mapping between scripts and
paper figures/tables is documented in [REPRODUCING.md](REPRODUCING.md).

## Repository structure

- `mufora/`: library code (sphere/circle detection, metrics, calibration
  handling, table utilities)
- `scripts/analysis/`: ground truth, metadata and evaluation scripts that
  produce the CSV files consumed by the figure scripts
- `scripts/figures/`: figure scripts (Figures 7–11, A1)
- `scripts/table/`: correlation summary used for the metric discussion
- `scripts/check_setup.py`: dataset/annotation sanity check
- `data/calib/`, `data/fog/`, `data/annotations/`: extrinsics, visibility
  measurements and ball annotations bundled for reproduction
- `tests/`: unit tests (`pytest`)

## Development

```bash
make format   # code formatting (docker)
make lint     # linting (docker)
make test     # run unit tests (docker)
```

## Cite This Paper

```bibtex
@article{behret2026mufora,
  author    = {Behret, Valentino and Kushtanova, Regina and Weber, Simon and
               Wiesner, Julian and Helmer, Thomas and Palme, Frank},
  title     = {MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather
               on Camera and LiDAR},
  journal   = {SN Computer Science},
  volume    = {7},
  number    = {442},
  year      = {2026},
  doi       = {10.1007/s42979-026-04976-9}
}
```

## License

This project is released under the [MIT License](LICENSE).
