"""
Created on Sat Nov 16 2024
Copyright (c) 2024 Munich University of Applied Sciences

Script to generate plots from csv files.
"""

import argparse
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log

from mufora import aux, data, draw

MIN_DATETIME = datetime(
    year=2024, month=2, day=28, hour=12, minute=10, tzinfo=timezone.utc
)
MAX_DATETIME = datetime(
    year=2024, month=5, day=29, hour=10, minute=15, tzinfo=timezone.utc
)
COLOR_MAP = {k: next(draw.COLORS) for k in ["rain", "fog", "light", "", "night"]}


@dataclass
class PlotData:
    df: pd.DataFrame
    name: str
    title: str
    x_label: str
    x_min: float = 0
    x_max: float = 1e3

    def __post_init__(self):
        if self.name == "cam":
            self.x_min = np.floor(self.df["metric"].min())
            self.x_max = np.ceil(self.df["metric"].max())


def _format_label(label: str):
    """Define style for all subplots"""
    if "+" not in label:
        return ""
    weather, intensity = label.split("+")
    x_label = {
        "light": "Clear",
        "night": "Night",
        "rain": f"Rain ({intensity} " + r"$\frac{\mathrm{mm}}{\mathrm{h}}$)",
        "fog": f"Fog ({intensity} m)",
    }[weather]
    return x_label


def _get_labels(df_weather: pd.DataFrame) -> list:
    """Get labels for y-axis of box plot"""
    df_combo = (
        df_weather[["weather", "intensity"]]
        .drop_duplicates()
        .sort_values(["weather", "intensity"])
    )
    labels = []
    for _, row in df_combo.iterrows():
        labels.append((row.weather, row.intensity))
    return labels


def _sort_df(sub_df: pd.DataFrame, all_indexes) -> pd.DataFrame:
    """Sort dataframe:
    1. Clear
    2. Night
    3. Rain from min intensity to max
    4. Fog from max visibility to min
    """

    # Add rows for all_indexes if don't exist
    for weather, intensity in all_indexes:
        if (weather, intensity) not in sub_df.index:
            new_row = pd.DataFrame(
                index=pd.MultiIndex.from_tuples([(weather, intensity)]),
                data={"metric": [np.nan], "intensity_int": [np.nan]},
            )
            sub_df = pd.concat([sub_df, new_row])

    # Sort by weather and intensity
    sub_df = sub_df.sort_index()
    sub_df["pos"] = np.nan
    sub_df.loc["light", "pos"] = 1
    sub_df.loc["night", "pos"] = 2
    sub_df.loc["rain", "pos"] = (
        1 + sub_df["pos"].max() + np.arange(len(sub_df.loc["rain"]))
    )
    sub_df.loc["fog", "pos"] = (
        1 + sub_df["pos"].max() + np.arange(len(sub_df.loc["fog"]))[::-1]
    )

    # Reverse order of dataframe
    sub_df["pos"] = sub_df["pos"].max() - sub_df["pos"] + 1

    return sub_df


def _max_delta(sub_df: pd.DataFrame) -> dict:
    """Draw max deltas into each subplot"""
    sub_df["median"] = sub_df["metric"].apply(np.median)
    baseline = float(sub_df.loc["light", "median"].iloc[0])
    sub_df["delta"] = (baseline - sub_df["median"]) / baseline * 100

    pos_max = sub_df["pos"].max()

    max_deltas = {}
    for weather in ["light", "rain", "fog"]:
        subsub_df = sub_df.loc[weather]
        if subsub_df["delta"].isna().all():
            continue
        intensity = subsub_df["delta"].idxmax()
        row = subsub_df.loc[intensity]

        max_deltas[(weather, intensity)] = pd.Series(
            {
                "q0": float(row["median"]),
                "q1": baseline,
                "pos0": pos_max - subsub_df["pos"].min(),
                "pos1": pos_max - subsub_df["pos"].max(),
                "percent": row["delta"],
            }
        )

    return max_deltas


def group_plot(plt_data: PlotData):
    """Save grouped plot"""

    if plt_data.df.empty:
        raise ValueError("Dataframe is empty")

    # Filter data
    df_eval = plt_data.df[["weather", "distance", "metric", "intensity"]]

    # Create bins depending on weather
    unique_distances = sorted(df_eval["distance"].unique())
    unique_weathers = ["rain", "fog", "dark", "light"]
    df_weather = df_eval[df_eval["weather"].isin(unique_weathers)]

    # Divide fog intensities into bins but only for fog
    df_fog = df_eval[df_eval["weather"] == "fog"].copy()
    df_non_fog = df_eval[df_eval["weather"] != "fog"].copy()
    fog_bins = [0, 10, 20, 40, 80, 160]
    df_fog["intensity"] = pd.cut(df_fog["intensity"], fog_bins, labels=fog_bins[1:])
    df_weather = pd.concat([df_non_fog, df_fog])
    df_weather = df_weather.dropna(subset="intensity")
    df_weather["intensity"] = df_weather["intensity"].astype(int)

    aux.summary_count(df_weather, title=plt_data.name)

    # All combinations
    all_indexes = _get_labels(df_weather)

    # Generate box plots for each combination of weather distance and intensity
    df_grouped = df_weather.groupby(
        ["weather", "distance", "intensity"], observed=True
    ).agg(list)

    n_other = len(unique_distances) // 2
    fig, ax = plt.subplots(
        ncols=2,
        nrows=n_other,
        figsize=(8, 1 + n_other * 1.9),
        # sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    indexes = itertools.product(range(n_other), range(2))
    idx_dist_map = zip(indexes, unique_distances)
    for idx, distance in idx_dist_map:
        sub_df = df_grouped.xs(distance, level="distance")
        sub_df = _sort_df(sub_df, all_indexes)

        max_deltas = _max_delta(sub_df)
        for (weather, intensity), sub_sub_df in sub_df.iterrows():
            color = COLOR_MAP[weather]

            # Median
            max_delta = max_deltas.get((weather, intensity), None)
            if weather == "light" and max_delta is not None:
                ax[idx].axhline(
                    max_delta.q0,
                    color=COLOR_MAP["light"],
                    linewidth=1,
                    linestyle="--",
                )
            elif max_delta is not None and max_delta.percent > 0:
                ax[idx].plot(
                    [max_delta.pos0 + 0.5, max_delta.pos1 - 0.5],
                    [max_delta.q0, max_delta.q0],
                    color=color,
                    linewidth=1,
                    linestyle="--",
                )
                # Draw arrow
                pos_c = (max_delta.pos0 + max_delta.pos1) / 2 + 0.5
                ax[idx].annotate(
                    "",
                    xytext=(pos_c, max_delta.q0 - 0.02),
                    xy=(pos_c, max_delta.q1 + 0.02),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color=color,
                        linewidth=1,
                    ),
                )
                ax[idx].text(
                    pos_c,
                    max_delta.q0 - 0.15 if max_delta.q1 > 0.6 else max_delta.q1 + 0.1,
                    f"-{max_delta['percent']:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                )

            # Boxplot
            x_values = sub_sub_df.metric
            label = _format_label(f"{weather}+{intensity}")  # if idx[0] == 0 else ""
            pos = sub_df["pos"].max() - sub_sub_df.pos
            bplot = ax[idx].boxplot(
                [x_values],
                tick_labels=[label] if idx[0] == n_other - 1 else [""],
                patch_artist=True,
                positions=[pos],
                widths=0.5,
                medianprops={"color": "black"},
            )

            # Change colors
            bplot["boxes"][0].set_facecolor(color)
            ax[idx].axvspan(
                pos - 0.5, pos + 0.5, alpha=0.1, facecolor=color, edgecolor="none"
            )

        ax[idx].set_title(f"{distance} m")
        ax[idx].grid(True)
        ax[idx].set_ylim(0, 1)
        ax[idx].set_yticks([0, 0.5, 1])
        ax[idx].set_yticklabels([0, 0.5, 1])
        ax[idx].tick_params(axis="x", labelrotation=60)
        ax[idx].set_xticklabels(ax[idx].get_xticklabels(), ha="right")

    fig.supylabel(plt_data.x_label)
    fig.tight_layout()
    fig.savefig(f"./tmp/{plt_data.name}_distance.pdf", bbox_inches="tight")


def _filter(df: pd.DataFrame, is_valid: pd.Series, name: str):
    """Apply filter and log how many points got filtered"""
    log.info(f"{name.title()} removed ({sum(~is_valid)}/{len(df)}) outliers")
    return df[is_valid]


def figure_metrics_3d(df_eval: pd.DataFrame):
    """Plot metrics with grouping by distance and weather"""

    # Preprocess: compute norm, remove units and convert delta_t to fog distance
    df_eval["norm"] = df_eval[["x_m", "y_m", "z_m"]].apply(np.linalg.norm, axis=1)

    # Remove outliers
    is_norm_valid = (df_eval["norm"] - df_eval["distance"]).abs() / df_eval[
        "distance"
    ] < 0.2
    is_in_fog_time = df_eval["datetime"].between(MIN_DATETIME, MAX_DATETIME) | (
        df_eval["weather"] != "fog"
    )
    df_eval = df_eval.drop("norm", axis=1)
    df_eval = _filter(df_eval, is_norm_valid, "norm")
    df_eval = _filter(df_eval, is_in_fog_time, "fog time")

    # Split data by sensor
    df_qb2_0 = df_eval[df_eval["sensor"] == "qb2_0"]
    df_qb2_1 = df_eval[df_eval["sensor"] == "qb2_1"]
    aux.summary_count(df_qb2_0, title="qb2_0")
    aux.summary_count(df_qb2_1, title="qb2_1")

    if df_eval["metric"].max() >= 1:
        aux.summary_max(df_qb2_0, "metric", "qb2_0")
        aux.summary_max(df_qb2_1, "metric", "qb2_1")

    # Plot
    qb2_0 = PlotData(
        df=df_qb2_0,
        name="qb2_0",
        title=r"LiDAR $L_0$",
        x_label=r"Inlier ratio $q_{I}$",
    )
    qb2_1 = PlotData(
        df=df_qb2_1,
        name="qb2_1",
        title=r"LiDAR $L_1$",
        x_label=r"Inlier ratio $q_{I}$",
    )
    group_plot(qb2_0)
    group_plot(qb2_1)


def figure_metrics_2d(df_eval: pd.DataFrame):
    """Plot metrics with grouping by distance and weather"""

    # Remove outliers
    is_in_fog_time = df_eval["datetime"].between(MIN_DATETIME, MAX_DATETIME) | (
        df_eval["weather"] != "fog"
    )
    df_eval = _filter(df_eval, is_in_fog_time, "fog time")

    # Split data by sensor
    aux.summary_count(df_eval, title="cam_l")

    if df_eval["metric"].max() > 1.0:
        aux.summary_max(df_eval, "metric", "cam_l")

    # Plot
    cam = PlotData(
        df=df_eval,
        name="cam",
        title=r"Cameras",
        x_label=r"Normalised entropy $\hat{S}$",
    )
    group_plot(cam)


def main(args: argparse.Namespace):
    """In thew main function all relevant table are loaded for plotting."""

    # Read SQL data
    df_eval_2d = pd.read_csv(args.file_eval_2d)
    df_eval_3d = pd.read_csv(args.file_eval_3d)

    # Choose metric
    # df_eval_2d["metric"] = 1 - df_eval_2d["glcm_homogeneity"]
    df_eval_3d["metric"] = df_eval_3d["quality"]

    # Preprocess tables
    df_eval_2d["datetime"] = pd.to_datetime(df_eval_2d["datetime"], utc=True)
    df_eval_3d["datetime"] = pd.to_datetime(df_eval_3d["datetime"], utc=True)

    # Create figures
    figure_metrics_3d(df_eval_3d)
    figure_metrics_2d(df_eval_2d)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file-eval-2d", type=Path, default=data.root() / "analysis/eval_2d.csv"
    )
    argparser.add_argument(
        "--file-eval-3d", type=Path, default=data.root() / "analysis/eval_3d.csv"
    )
    main(argparser.parse_args())
