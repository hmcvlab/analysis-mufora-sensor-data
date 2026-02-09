"""
Created on Wed Oct 30 2024
Copyright (c) 2024 Munich University of Applied Sciences

Script to generate a ground truth csv file from coco annotations
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger as log
from rich.progress import Progress

from mufora import aux, data, table


def main():
    """Main function."""

    dir_labels = data.root() / "annotate/ball/3d/labels"

    data_processed = []
    for sensor in ["qb2_0", "qb2_1"]:
        files = list(dir_labels.glob(f"*_{sensor}.json"))
        with Progress() as progress:
            for file_gt in progress.track(
                files, description="Processing", total=len(files)
            ):
                with open(file_gt, "r", encoding="utf-8") as f:
                    label = json.load(f)

                if "objects" not in label or len(label["objects"]) == 0:
                    progress.console.print(f"No object in {file_gt}!")
                    continue

                folder = label["filename"]

                if "calib" in folder:
                    progress.console.print(f"Skipping calib {folder}")
                    continue

                weather = table.folder2weather(folder)
                if weather is None:
                    progress.console.print(f"No valid weather: {folder}")
                    continue

                obj = label["objects"][0]
                data_processed.append(
                    {
                        "sensor": sensor,
                        "date": table.folder2date(folder),
                        "time": table.folder2time(folder),
                        "weather": weather,
                        "distance": table.folder2distance(folder),
                        "intensity": table.folder2intensity(folder),
                        "x_m": obj["centroid"]["x"],
                        "y_m": obj["centroid"]["y"],
                        "z_m": obj["centroid"]["z"],
                        "radius_m": obj["dimensions"]["length"] / 2,
                    }
                )

    df = (
        pd.DataFrame(data_processed)
        .sort_values(["weather", "distance", "intensity", "date", "time"])
        .reset_index(drop=True)
    )
    df["distance"] = df["distance"].astype(int)
    df["intensity"] = df["intensity"].astype(float)
    log.info(df)
    log.info(df.dtypes)

    # Remove duplicates and only keep the last
    relevant_columns = ["weather", "distance", "intensity", "date", "sensor"]
    indexes = df[relevant_columns].drop_duplicates(keep="last").index
    df_keep = df.loc[indexes]
    df_remove = df[~df.index.isin(indexes)]
    log.info(f"Final dataframe:\n{df_keep.to_string()}")
    log.warning(f"Removed data:\n{df_remove.to_string()}")
    aux.summary_count(df_keep[df_keep["sensor"] == "qb2_0"], title="qb2_0")
    aux.summary_count(df_keep[df_keep["sensor"] == "qb2_1"], title="qb2_1")

    # Export
    engine = table.engine(database="weather")
    table.save(df_keep, table="gt_3d", sql_engine=engine)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument(
        "--file-output", type=Path, default=data.root() / "analysis/gt_2d.csv"
    )
    argparser.add_argument(
        "--file-input",
        type=Path,
        default=data.root()
        / "annotate/ball/2d/carissma-indoor-multi-cam.v2i.coco/_annotations.coco.json",
    )
    main(argparser.parse_args())
