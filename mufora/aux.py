"""
Created on Wed Jul 09 2025
Copyright (c) 2025 Munich University of Applied Sciences

Module for auxiliary functions
"""

import pandas as pd
from loguru import logger as log


def times_summary(list_of_times: list, title: str) -> str:
    """Summarize timestamps"""
    t_deltas = [
        list_of_times[i + 1] - list_of_times[i] for i in range(len(list_of_times) - 1)
    ]
    t_deltas_ms = [round(t.total_seconds() * 1000) for t in t_deltas]
    return f"{title}: {t_deltas_ms}={sum(t_deltas_ms)} ms"


def summary_max(df: pd.DataFrame, col: str, title: str):
    """Generate a summary of a specific column"""
    df_max = (
        df[["weather", "distance", col]]
        .groupby(["weather", "distance"], as_index=False)
        .max()
        .pivot(index="weather", columns="distance")
        .fillna(0)
        .round(3)
        # .astype(int)
    )
    log.info(f"Max {col} for {title}:\n{df_max}")
    log.info(
        "\n"
        + title.center(80, "=")
        + f"\nMax metric for each combination:\n{df_max}\n"
        + "=" * 80
    )


def summary_count(df: pd.DataFrame, title: str = ""):
    """Check all combinations of weather, intensity and distance"""

    df["intensity_int"] = (df["intensity"] / 10).round().astype(int) * 10

    df_grouped = df[["weather", "distance", "intensity_int"]].pivot_table(
        index=["weather", "intensity_int"],
        columns="distance",
        aggfunc="size",
        fill_value=0,
    )

    log.info(
        "\n"
        + title.center(80, "=")
        + f"\nNumber of datapoints for each combination:\n{df_grouped}\n"
        + "=" * 80
    )


def summary_sensor_date(df: pd.DataFrame):
    """Check all combinations of sensor and date"""
    df_grouped = df[["sensor", "date"]].pivot_table(
        index="date",
        columns="sensor",
        aggfunc="size",
        fill_value=0,
    )

    log.info(
        "\n"
        + "Evaluation".center(80, "=")
        + f"\nNumber of datapoints for each combination:\n{df_grouped}\n"
        + "=" * 80
    )
