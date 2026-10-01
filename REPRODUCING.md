# Reproducing the paper results

This document maps every script to the figure/table it produces in

> **MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather on Camera
> and LiDAR** (SN Computer Science, 7, 442, 2026)

and describes the order in which to run them.

## Prerequisites

1. The MuFoRa dataset extracted so that `data/rawdata/` or
   `data/MuFoRa_dataset/` exists (see [README.md](README.md#dataset)) – or
   `MUFORA_ROOT` pointing at the directory containing it.
2. The package installed (`pip install -e .` or the dev container).
3. Run the sanity check:

   ```bash
   python scripts/check_setup.py   # or: make check
   ```

   It verifies the dataset root, the rawdata folder structure, per-folder
   `calib.yaml` files, and the bundled annotation/visibility/extrinsic files
   under `data/`.

## Pipeline overview

The analysis scripts produce CSV files under `<dataset-root>/analysis/`.
They depend on each other as follows (top to bottom):

```
data/fog/*.csv                 ──► scripts/analysis/fog.py         ──► fog.csv
data/annotations/2d/…/coco.json ──► scripts/analysis/gt_2d.py      ──► gt_2d.csv
data/annotations/3d/*.json      ──► scripts/analysis/gt_3d.py      ──► gt_3d.csv
rawdata/ + gt_2d.csv + fog.csv  ──► scripts/analysis/metadata_2d.py ──► metadata_2d.csv
rawdata/ + gt_3d.csv + fog.csv  ──► scripts/analysis/metadata_3d.py ──► metadata_3d.csv
rawdata/ + metadata_2d.csv      ──► scripts/analysis/eval_2d.py    ──► eval_2d.csv
rawdata/ + metadata_3d.csv      ──► scripts/analysis/eval_3d.py    ──► eval_3d.csv
```

`fog.csv`, `gt_2d.csv` and `gt_3d.csv` only need the files bundled in this
repository and can be generated without the dataset. The `metadata_*` and
`eval_*` scripts scan `rawdata/`; `eval_3d.py` is the most expensive step
(RANSAC sphere fitting for every point cloud).

Run everything in one go:

```bash
make reproduce
```

## Script → paper artifact mapping

| Paper artifact | Command | Inputs | Output |
| --- | --- | --- | --- |
| Fig. 7 – daily reprojection errors (Ls/Ld, labelled vs. refined) | `python scripts/figures/reprojection_error.py` | `eval_2d.csv`, `eval_3d.csv`, `data/calib/*.json` | `tmp/reprojection_error.pdf` |
| Fig. 8 – fog visibility + sample frames | `python scripts/figures/fog.py` | `fog.csv`, `metadata_2d.csv`, raw images | `tmp/fog.png` |
| Fig. 9 – camera normalised entropy Ŝ | `python scripts/figures/weather_impact.py` | `eval_2d.csv`, `eval_3d.csv` | `tmp/cam_distance.pdf` |
| Fig. 10 – inlier ratio qI, LiDAR Ls | `python scripts/figures/weather_impact.py` | `eval_3d.csv` | `tmp/qb2_0_distance.pdf` |
| Fig. 11 – inlier ratio qI, LiDAR Ld | `python scripts/figures/weather_impact.py` | `eval_3d.csv` | `tmp/qb2_1_distance.pdf` |
| Fig. A1 – sample image grid | `python scripts/figures/samples.py` | `metadata_2d.csv`, raw images | `tmp/samples.pdf` |
| Tables A1–A2 – file counts | log output of `metadata_2d.py` / `metadata_3d.py` (`aux.summary_count`) | metadata CSVs | stdout |
| Table A3 – measurement dates | log output of `table/correlation_metrics.py` (`aux.summary_sensor_date`) | `eval_*.csv` | stdout |
| Metric justification (§5.3: entropy vs. GLCM) | `python scripts/table/correlation_metrics.py` | `eval_2d.csv`, `eval_3d.csv` | stdout |
| Table 6 – detection benchmark | companion repo [model-ball-detection](https://github.com/hmcvlab/model-ball-detection) | Accurate Balls Detection dataset | TeX table |

Figures are written to `tmp/` relative to the working directory, so run the
figure scripts from the repository root.

## Step by step

```bash
# 1. Ground truth and weather data (only needs the bundled files)
python scripts/analysis/fog.py
python scripts/analysis/gt_2d.py
python scripts/analysis/gt_3d.py

# 2. Metadata index over rawdata/ (joins GT + fog visibility)
python scripts/analysis/metadata_2d.py
python scripts/analysis/metadata_3d.py

# 3. Per-sample metrics
python scripts/analysis/eval_2d.py    # normalised entropy + GLCM features
python scripts/analysis/eval_3d.py    # RANSAC sphere fit, inlier ratio

# 4. Figures and tables
python scripts/figures/reprojection_error.py
python scripts/figures/fog.py
python scripts/figures/weather_impact.py
python scripts/figures/samples.py
python scripts/table/correlation_metrics.py
```

## Smoke test / partial reproduction

Every analysis script accepts `--debug`, which processes only a small random
subset (10 samples) and skips writing the output CSV – useful to verify that
the dataset is wired up correctly before the full run:

```bash
python scripts/analysis/eval_3d.py --debug
```

`eval_2d.py`/`eval_3d.py` also accept `--sensor` to evaluate only one sensor
(e.g. `--sensor qb2_0`), and all scripts expose `--file-*`/`--dir-*` arguments
to override input/output paths (see `--help`).

## Determinism

- `eval_3d.py` seeds NumPy's RNG (`--seed`, default `0`). The paper results
  were produced without a fixed seed, so reproduced values may deviate
  slightly in individual samples while the aggregated statistics (medians,
  box plots) should match.
- `table.save()` merges new rows into existing CSVs instead of overwriting
  them. To regenerate a table from scratch, delete the corresponding file in
  `data/analysis/` (or `$MUFORA_ROOT/analysis/` if the variable is set)
  first.

## Expected eval_3d output

Some rows in `eval_3d.csv` end up with `inlier_ratio = NaN`. This is
expected, not a bug:

- In dense fog or at long distances the target produces fewer than the four
  points required for a sphere fit.
- The 3D ground truth is joined by `(date, distance, weather, sensor)` and
  stems from separate reference recordings. For some folders the reference
  position differs by more than the 0.35 m cone radius around the expected
  ball position, so no fit is attempted.

NaN rows are excluded from the figures; the original evaluation behaved the
same way.

## Published archive vs. original results

`MuFoRa_dataset.zip` is a subsample of the raw recordings used for the paper
(roughly 5× fewer files for rain, ~15× for fog/dry, and a few measurement
sessions are omitted entirely). Because `metadata_*.py` samples frame
*indices* (`linspace`), the reproduced CSVs contain different frames than the
original `eval_*.csv` — per-row values cannot match exactly.

The aggregated distributions do, however. Comparing medians per
(sensor, weather, distance) cell against the original `eval_*.csv`:

- `inlier_ratio` (qI): median absolute deviation ≈ 0.01; 84% of cells within
  0.05
- normalized entropy: median absolute deviation ≈ 0.005; 92% of cells within
  0.05 (the largest deviations are fog cells, where visibility changes
  during a session, so a different frame subset shifts the median)

Note: ~150 `.pcd` files in the archive are empty (0 B) and are skipped by
the eval.

## Terminology

| Code | Paper |
| --- | --- |
| `qb2_0` | LiDAR Ls (sparse, 80 scanlines) |
| `qb2_1` | LiDAR Ld (dense, 200 scanlines) |
| `cam_l` / `cam_r` | left/right camera of the stereo setup |
| `metric` (eval_2d) | normalised entropy Ŝ |
| `inlier_ratio` (eval_3d) | inlier ratio qI |
| `intensity` | rain rate in mm/h, or fog visibility in m |
