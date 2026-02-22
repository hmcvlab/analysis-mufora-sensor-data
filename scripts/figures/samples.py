"""
Created on Sat Nov 16 2024
Copyright (c) 2024 Munich University of Applied Sciences

Script to generate plots from csv files.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger as log

from mufora import data, sql

MIN_DATETIME = datetime(
    year=2024, month=2, day=28, hour=12, minute=10, tzinfo=timezone.utc
)
MAX_DATETIME = datetime(
    year=2024, month=5, day=29, hour=10, minute=15, tzinfo=timezone.utc
)
FILENAME = Path(__file__).stem
DIR_IMAGES = data.root() / "rawdata"


def _merge(row: pd.Series) -> str:
    """Merge inetensity and weather"""
    return f"{row.weather}+{row.intensity}"


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


def _sort_weathers(unique_weathers) -> list:
    """Sort dataframe:
    1. Clear
    2. Night
    3. Rain from min intensity to max
    4. Fog from max visibility to min
    """

    # Add rows for all_indexes if don't exist
    new_list = []
    for weather in ["light", "night", "rain", "fog"]:
        sub_weather = [x for x in unique_weathers if weather in x]
        new_list += sub_weather
    return new_list


def figure_samples(df_meta: pd.DataFrame):
    """Plot for visibility for each recording."""

    # Only use very close images at 10 m
    df_meta = df_meta[df_meta["distance"] == 5]
    df_meta = df_meta[df_meta["datetime"] > MIN_DATETIME]
    df_meta = df_meta[df_meta["datetime"] < MAX_DATETIME]

    # Divide fog intensities into bins but only for fog
    df_fog = df_meta[df_meta["weather"] == "fog"].copy()
    df_non_fog = df_meta[df_meta["weather"] != "fog"].copy()
    fog_bins = [0, 10, 20, 40, 80, 160]
    df_fog["intensity"] = pd.cut(df_fog["intensity"], fog_bins, labels=fog_bins[1:])
    df_meta = pd.concat([df_non_fog, df_fog])
    df_meta = df_meta.dropna(subset="intensity")
    df_meta["intensity"] = df_meta["intensity"].astype(int)
    df_meta["iweather"] = df_meta[["weather", "intensity"]].apply(_merge, axis=1)

    unique_weathers = list(df_meta["iweather"].unique())
    unique_weathers = _sort_weathers(unique_weathers)

    # Plot
    ncols = 8
    nrows = len(unique_weathers)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 1.5), squeeze=False
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.01)

    # Iterate over all combinations of weather and intensity
    for i, iweather in enumerate(unique_weathers):

        #
        df_sub = df_meta[df_meta["iweather"] == iweather]
        df_sub = df_sub[df_sub["day"] == df_sub["day"].max()]
        df_sub = df_sub.sort_values("datetime")

        offset = 0
        limit = 3
        for j in range(ncols):
            # Take the first three and the last 1
            if j > limit:
                row = df_sub.iloc[-(j + offset - limit)]
            else:
                row = df_sub.iloc[j + offset]

            file = DIR_IMAGES / row.filename
            img = mpimg.imread(file)

            ax[i, j].imshow(img)
            ax[i, j].set_title(row["datetime"].strftime("%d.%m %H:%M:%S"), fontsize=9)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

            if j == 0:
                label = _format_label(row["iweather"])
                ax[i, j].set_ylabel(label)

    # fig.tight_layout()

    # Save figure
    file_img = Path("tmp") / f"{FILENAME}.pdf"
    file_img.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving figure to {file_img}")
    fig.savefig(file_img, bbox_inches="tight")


def main():
    """In thew main function all relevant table are loaded for plotting."""

    # Read SQL data
    engine = sql.engine(database="weather")
    df_meta = sql.query2df("SELECT * FROM metadata_2d", engine)

    df_meta["datetime"] = pd.to_datetime(df_meta["datetime"], utc=True)
    df_meta["day"] = df_meta["datetime"].dt.date

    # Create figures
    figure_samples(df_meta)


if __name__ == "__main__":
    main()
