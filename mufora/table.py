"""
Created on Tue Nov 26 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from loguru import logger as log


def save(
    df: pd.DataFrame,
    filename: Path,
    overwrite=True,
):
    """First load old table if exists then merge tables and replace online."""
    if not overwrite and filename.exists():
        df_old = pd.read_csv(filename)
        df = df.reset_index()
        df = pd.concat([df_old, df]).drop_duplicates(
            keep="last", subset=["datetime", "sensor"]
        )
        log.info(f"Updating {filename} by adding {len(df)-len(df_old)} new rows.")
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Save table {filename}.")

    log.info(f"Table info:\n{df.dtypes}")

    df.to_csv(filename, index=False)


def folder2weather(filename: str) -> str:
    """Extract weather from filename."""
    weather = None
    if "raw_dry_dark_with_car_light" in filename:
        weather = "night"
    if "raw_fog" in filename:
        weather = "fog"
    if "raw_dry_light" in filename:
        weather = "light"
    if "raw_rain" in filename:
        weather = "rain"
    return weather


def folder2intensity(filename: str) -> int:
    """Match fog time to visibility."""
    intensity = -1
    if "rain" in filename:
        # Return rain_{2-3}[0-9]mm use regex
        intensity = re.search(r"[0-9]{2,3}mm", filename).group(0).replace("mm", "")
    return int(intensity)


def file2datetime(file: Path) -> datetime:
    """Extract timestamp from filename."""

    # Data was recorded in CET
    tz_cet = pytz.timezone("Europe/Berlin")
    datetime_cet = datetime.fromtimestamp(
        float(file.name.split("_")[0]) / 1e9, tz=tz_cet
    )
    datetime_utc = datetime_cet.astimezone(pytz.utc)
    return datetime_utc


def file2sensor(file: Path):
    """Extract sensor type from filename."""
    if "left_image" in file.name:
        sensor = "cam_l"
    elif "right_image" in file.name:
        sensor = "cam_r"
    elif "qb2_0" in file.name:
        sensor = "qb2_0"
    elif "qb2_1" in file.name:
        sensor = "qb2_1"
    else:
        raise RuntimeError(f"No valid sensor from file: {file}")
    return sensor


def folder2sensor(filename: str):
    """Extract sensor type from filename."""
    if "cam_l" in filename:
        sensor = "cam_l"
    elif "cam_r" in filename:
        sensor = "cam_r"
    elif "qb2_0" in filename:
        sensor = "qb2_0"
    elif "qb2_1" in filename:
        sensor = "qb2_1"
    else:
        raise RuntimeError(f"No valid sensor from filename: {filename}")
    return sensor


def folder2distance(filename: str) -> int:
    """Extract distance from filename."""
    filename += "_"
    try:
        sub_str = re.search(r"_[0-9]{1,2}m_", filename).group(0)
    except AttributeError as err:
        raise RuntimeError(
            f"No valid distance from filename: {filename}\n{err}"
        ) from err
    return int(sub_str[1:-2])


def folder2date(filename: str) -> str:
    """Extract date from filename."""
    return re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", filename).group(0)


def folder2time(filename: str) -> str:
    """Extract time from filename."""
    str_time = re.search(r"_[0-9]{2}-[0-9]{2}-[0-9]{2}_", filename).group(0)
    return str_time[1:-1]


def add_fog_intensity(df_meta: pd.DataFrame, df_fog: pd.DataFrame):
    """Add fog intensity column."""
    df_meta["datetime"] = pd.to_datetime(df_meta["datetime"], utc=True)
    df_meta = df_meta.sort_values("datetime")

    # Convert fog datetime to UTC and rename idx into idx_file_fog
    df_fog["datetime"] = pd.to_datetime(df_fog["datetime"], utc=True, format="ISO8601")
    df_fog = df_fog.sort_values("datetime")

    # Merge fog visibility
    df_tmp = pd.merge_asof(
        df_meta[["datetime"]],
        df_fog,
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1s"),
    )
    df_tmp = df_tmp.set_index("datetime")
    df_meta = df_meta.set_index("datetime")
    df_meta.update(df_tmp)
    return df_meta.reset_index()
