"""
Created on Sat Nov 16 2024
Copyright (c) 2024 Munich University of Applied Sciences

Script to generate plots from csv files.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log

from mufora import data, draw

MIN_DATETIME = datetime(
    year=2024, month=2, day=28, hour=12, minute=10, tzinfo=timezone.utc
)
MAX_DATETIME = datetime(
    year=2024, month=5, day=29, hour=10, minute=15, tzinfo=timezone.utc
)
FILENAME = Path(__file__).stem
DIR_IMAGES = data.root() / "rawdata"


def _extract_images_same_visibility(df, time_delta=pd.Timedelta(minutes=1)):
    """Extract from dataframe 4 rows with same visibility and max timestamp distance."""

    entries = []
    fog_bins = np.arange(0, 200, 2)
    df = df[["datetime", "distance", "intensity", "day", "filename"]].copy()
    df["intensity_bin"] = pd.cut(df["intensity"], fog_bins, labels=fog_bins[1:])
    for (day, distance, _), sub_df in df.groupby(["day", "distance", "intensity_bin"]):
        if len(sub_df) < 4:
            continue
        indexes = sub_df.index
        idx0 = indexes[0]
        keep = [idx0]
        for i in range(1, len(indexes)):
            idx1 = indexes[i]
            t0 = sub_df["datetime"].loc[idx0]
            t1 = sub_df["datetime"].loc[idx1]
            delta = t1 - t0
            if delta > time_delta:
                keep.append(idx1)
                idx0 = idx1
        entries.append(
            {
                "day": str(day),
                "distance": int(distance),
                "datetime": sub_df.loc[keep, "datetime"].to_list(),
                "intensity": sub_df.loc[keep, "intensity"].to_list(),
                "filenames": sub_df.loc[keep, "filename"].to_list(),
            }
        )

    # Sort entries by number of images
    entries = sorted(entries, key=lambda x: len(x["filenames"]), reverse=True)
    best_entry = entries[0]

    # Get the 4 rows with max distance from each other
    return pd.Series(best_entry)


def figure_fog_matching(df_fog: pd.DataFrame, df_meta: pd.DataFrame):
    """Plot for visibility for each recording."""
    # Set day for syncing
    # Plot
    fig = plt.figure(figsize=(9.4, 6.8))
    gs = fig.add_gridspec(2, 1, hspace=0.2, height_ratios=[2, 1])
    gs0 = gs[1].subgridspec(1, 4, wspace=0.0)
    gs1 = gs[0].subgridspec(2, 1, hspace=0.75)
    ax = [
        fig.add_subplot(gs1[0]),
        fig.add_subplot(gs1[1]),
        fig.add_subplot(gs0[0]),
        fig.add_subplot(gs0[1]),
        fig.add_subplot(gs0[2]),
        fig.add_subplot(gs0[3]),
    ]

    # Custom vertical space between first second and third row
    # first and second very low and second and third high

    # Filter by weather
    df_meta = df_meta[df_meta["weather"] == "fog"]
    df_fog = df_fog[df_fog["datetime"] > MIN_DATETIME]
    df_meta = df_meta[df_meta["datetime"] > MIN_DATETIME]
    df_fog = df_fog.reset_index()
    df_meta = df_meta.reset_index()

    # Fog per day
    dates = ["2024-02-28", "2024-05-29"]
    df_fog_per_day = df_fog.groupby("day").agg(list)
    for day, sub_df in df_fog_per_day.iterrows():
        x = sub_df["datetime"]
        y = sub_df["intensity"]
        day = day.strftime("%Y-%m-%d")
        idx = dates.index(day)
        label = "All data" if idx == 0 else None
        ax[idx].scatter(x, y, label=label, marker=".", color="black", s=0.5)

    # Fog per distance
    df_grouped = df_meta.groupby(["distance", "day"]).agg(list)
    for (distance, day), sub_df in df_grouped.iterrows():
        title = day.strftime("%B %d, %Y")
        day = day.strftime("%Y-%m-%d")
        idx = dates.index(day)
        x = sub_df["datetime"]
        y = sub_df["intensity"]
        color = draw.COLOR_MAP[distance]
        marker = draw.MARKER_MAP[distance]
        ax[idx].plot(
            x,
            y,
            label=f"{distance:.0f} m",
            marker=marker,
            color=color,
            markevery=200,
            linewidth=0.75,
        )
        ax[idx].set_title(title)
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("Visibility in m")
        ax[idx].set_yscale("log")
        ax[idx].set_ylim(6, 1e3)
        ax[idx].grid(True)
        # ax[idx].tick_params(axis="x", labelrotation=45)
        ax[idx].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))

    # Get all  entries with visibility 50 and distance 35 m
    best_entry = None
    time_delta = pd.Timedelta(minutes=2)
    while best_entry is None:
        log.info(f"Trying time delta: {time_delta}")
        best_entry = _extract_images_same_visibility(df_meta, time_delta=time_delta)
        best_entry = None if len(best_entry["filenames"]) < 4 else best_entry
        time_delta -= pd.Timedelta(seconds=20)
    print(best_entry)

    # Load images into subplots
    letters = ["a", "b", "c", "d"]
    y_pos = [400, 400, 400, 400]
    for idx, (intensity, filename, dtime) in enumerate(
        zip(best_entry["intensity"], best_entry["filenames"], best_entry["datetime"])
    ):
        file = DIR_IMAGES / filename
        letter = letters[idx]
        ax[idx + 2].imshow(mpimg.imread(file))
        ax[idx + 2].axis("off")
        ax[idx + 2].set_title(f"({letter}) {intensity:.1f} m")

        # Plot letter corresponding to subplot
        if "2024-02" in best_entry["day"]:
            ax_idx = 0
        elif "2024-05" in best_entry["day"]:
            ax_idx = 1
        else:
            log.error(f"Unknown day {best_entry['day']}")
            break
        log.info(f"Plotting {letter} in {ax_idx} @ {dtime}")

        x0 = dtime
        y0 = y_pos[idx]
        y1 = intensity
        ax[ax_idx].plot([x0, x0], [y0, y1], color="black", linewidth=0.5)
        ax[ax_idx].text(
            x0,
            y0,
            letter,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

        if idx == len(ax):
            break

    # Sort legend labels
    handles = []
    labels = []
    for axis in ax:
        handle, label = axis.get_legend_handles_labels()
        handles += handle
        labels += label
    labels = map(lambda x: "0" + x if len(x) < 4 else x, labels)
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    labels = map(lambda x: x if x[0] != "0" else x[1:], labels)

    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncols=6)
    file_img = Path("tmp") / f"{FILENAME}.png"
    file_img.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving figure to {file_img}")
    fig.savefig(file_img, bbox_inches="tight")


def main(args: argparse.Namespace):
    """In thew main function all relevant table are loaded for plotting."""

    # Read SQL data
    df_fog = pd.read_csv(args.file_fog)
    df_meta = pd.read_csv(args.file_meta_2d)

    df_fog["datetime"] = pd.to_datetime(df_fog["datetime"], utc=True, format="ISO8601")
    df_meta["datetime"] = pd.to_datetime(df_meta["datetime"], utc=True)
    df_fog["day"] = df_fog["datetime"].dt.date
    df_meta["day"] = df_meta["datetime"].dt.date

    # Create figures
    figure_fog_matching(df_fog, df_meta)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument(
        "--file-fog", type=Path, default=data.root() / "analysis/fog.csv"
    )
    argparser.add_argument(
        "--file-meta-2d", type=Path, default=data.root() / "analysis/metadata_2d.csv"
    )
    main(argparser.parse_args())
