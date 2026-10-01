"""
Created on Tue Oct 08 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger as log

from mufora import settings

WHITELIST = [
    "raw_dry_dark_with_car_light",
    "raw_dry_light",
    "raw_fog",
    "raw_rain",
]
ENV_ROOT = "MUFORA_ROOT"
REPO_DATA = Path(__file__).resolve().parent.parent / "data"
PATHS = [
    Path("/mnt/data/"),
    Path("/mnt/labor/"),
    Path("/mnt/labor/storage1/"),
    Path("/media/T9/"),
]


RAWDATA_DIRNAMES = ["rawdata", "MuFoRa_dataset"]


def _rawdata_dir(path: Path) -> bool:
    """Check if path contains a directory with the measurement folders."""
    return any(path.joinpath(name).is_dir() for name in RAWDATA_DIRNAMES)


def root() -> Path:
    """Return the MuFoRa dataset root.

    Resolution order:

    1. The ``MUFORA_ROOT`` environment variable, if set.
    2. ``data/`` inside this repository, i.e. the dataset (or a symlink to it)
       extracted to ``data/rawdata``.
    3. The first directory in ``PATHS`` containing ``rawdata/``.

    A root is valid if it contains a ``rawdata/`` directory or the
    ``MuFoRa_dataset/`` top-level directory of the Zenodo archive.
    """
    if env := os.environ.get(ENV_ROOT):
        root_path = Path(env)
        if not _rawdata_dir(root_path):
            raise FileNotFoundError(
                f"{ENV_ROOT}={env} contains neither a 'rawdata' nor a "
                "'MuFoRa_dataset' directory."
            )
        return root_path

    for path in [REPO_DATA] + PATHS:
        if _rawdata_dir(path):
            return path

    raise FileNotFoundError(
        "MuFoRa dataset root not found. Download the dataset "
        "(https://doi.org/10.5281/zenodo.15016480) and extract it so that "
        f"'{REPO_DATA}/rawdata' exists (a symlink works too), or point "
        f"{ENV_ROOT} to it, e.g. 'export {ENV_ROOT}=/path/to/mufora'."
    )


def rawdata() -> Path:
    """Return the directory holding the measurement folders."""
    base = root()
    for name in RAWDATA_DIRNAMES:
        if base.joinpath(name).is_dir():
            return base.joinpath(name)
    raise FileNotFoundError(f"No rawdata directory found in {base}")


def relevant_raw_folders() -> list[Path]:
    """Find relevant folders for the project."""
    folders = sorted(rawdata().rglob("*"))
    log.info(f"Found {len(folders)} folders")
    folders = list(filter(lambda x: x.is_dir(), folders))
    folders = list(filter(lambda x: any(fol in x.name for fol in WHITELIST), folders))
    log.info(f"Found {len(folders)} relevant folders")
    return folders


def _filename2metadata(filename: Path) -> dict:
    """Extract metadata from filename."""
    file_parts = filename.stem.split("_")
    date = datetime.fromtimestamp(float(file_parts[0]) / 1e9, tz=timezone.utc)
    return {
        "datetime": date,
        "folder": str(filename.parent),
        "file": str(filename.stem),
        "type": filename.suffix,
        "sensor": "_".join(file_parts[1:]),
    }


def collect(path: Path, config: settings.Calibration) -> pd.DataFrame:
    """Collect data recursively in the given path and converts the filenames into
    metadata."""
    whitelist = config.whitelist

    # Extract all .pcd and png files
    data = []
    for file in filter(
        lambda x: any(sensor in x.stem for sensor in whitelist), path.rglob("*.pcd")
    ):
        data.append(_filename2metadata(file))

    for file in filter(
        lambda x: any(sensor in x.stem for sensor in whitelist), path.rglob("*.png")
    ):
        data.append(_filename2metadata(file))

    # Extract the metadata
    return pd.DataFrame(data).set_index("datetime")


def find_closest(df: pd.DataFrame) -> pd.DataFrame:
    """Find for each entry of the sensor the closest datetimes of the other sensors."""
    all_sensors = df["sensor"].unique()
    anchor_sensor = all_sensors[0]
    log.info(f"Anchor sensor: {anchor_sensor}")

    # Match sensors
    df_matches = pd.DataFrame(columns=all_sensors)
    df_main_sensor = df[df["sensor"] == anchor_sensor]
    df_matches[anchor_sensor] = df_main_sensor.index
    df_matches = df_matches.astype("datetime64[ns]").set_index(anchor_sensor)

    for idx in df_main_sensor.index:
        for sensor in filter(lambda x: x != anchor_sensor, all_sensors):
            tmp_dt = df[df["sensor"] == sensor].index
            df_matches.at[idx, sensor] = tmp_dt[np.argmin(abs(tmp_dt - idx))]

    # Drop duplicates where the delta is greater
    for sensor in df_matches.columns:
        df_matches[f"delta_{sensor}"] = abs(df_matches[sensor] - df_matches.index)
        df_matches[f"seconds_{sensor}"] = (
            df_matches[sensor].dt.second + 60 * df_matches[sensor].dt.minute
        )
        df_matches = (
            df_matches.sort_values(by=f"delta_{sensor}", ascending=True)
            .drop_duplicates(subset=sensor)
            .drop_duplicates(subset=f"seconds_{sensor}")
        )
        df_matches = df_matches.drop([f"delta_{sensor}", f"seconds_{sensor}"], axis=1)

    return df_matches
