"""
Created on Thu Oct 24 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger as log
from rich.progress import Progress

from mufora import aux, data, table

N_SAMPLES = 1000
FOG_SMOOTH_TIME = pd.Timedelta(seconds=1)


def collect_by_pattern(
    pattern: str,
    progress: Progress,
    folders: list,
    df_2d: pd.DataFrame,
) -> list:
    """Collect data only by using a pattern recursively."""
    metadata = []
    for folder in progress.track(folders, description=f"Collecting {pattern} data"):
        # Find all left camera images
        dirname = folder.name
        files = sorted(list(folder.glob(pattern)))
        progress.console.print(f"Found {len(files)=} in {dirname} for {pattern=}")

        if len(files) == 0:
            progress.console.print(f"Skipping {dirname}")
            continue

        weather = table.folder2weather(dirname)
        if weather is None:
            progress.console.print(f"Skipping {dirname}")
            continue

        # Collect metadata
        distance = table.folder2distance(dirname)
        indexes = np.unique(np.linspace(0, len(files) - 1, N_SAMPLES).astype(int))
        date = table.folder2date(dirname)

        # Create sub dataframes
        df_2d_tmp = df_2d[
            (df_2d["date"] == date)
            & (df_2d["distance"] == distance)
            & (df_2d["weather"] == weather)
        ]

        if df_2d_tmp.empty:
            progress.console.print(f"Missing data 2D {dirname}")
            continue

        t_start = table.file2datetime(files[0])
        for idx in indexes:
            file = files[idx]
            t_tmp = table.file2datetime(file)
            duration = t_tmp - t_start
            sensor = table.file2sensor(file)

            # Get ground truth
            idx_gt_2d = df_2d_tmp[df_2d_tmp["sensor"] == sensor].index
            if len(idx_gt_2d) != 1 and weather not in ["fog", "rain"]:
                raise RuntimeError(f"No unique ground truth:\n{df_2d.loc[idx_gt_2d]}")

            tmp_metadata = {
                "datetime": t_tmp,
                "sensor": sensor,
                "weather": weather,
                "distance": distance,
                "intensity": table.folder2intensity(dirname),
                "duration": duration.seconds,
                "r_gt_px": float(df_2d.loc[idx_gt_2d, "radius_px"].mean()),
                "x_gt_px": float(df_2d.loc[idx_gt_2d, "x_px"].mean()),
                "y_gt_px": float(df_2d.loc[idx_gt_2d, "y_px"].mean()),
                "filename": f"{dirname}/{file.name}",
            }
            metadata.append(tmp_metadata)

    return metadata


def main(args: argparse.Namespace):
    """
    Main function
    """
    log.info("Starting to generate metadata...")

    # Load additional data
    engine = table.engine(database="weather")
    df_2d = table.query2df("SELECT * FROM gt_2d", engine)

    log.info(f"Loading gt_2d: {len(df_2d)}")
    log.info(f"Describe gt_2d: {df_2d.dtypes}")
    aux.summary_count(df_2d)

    # Merge data by updating it with the new
    folders = data.relevant_raw_folders()

    if args.debug:
        folders = list(filter(lambda x: "2024-02-28" in x.name, folders))[:2]

    # Iterate over folders
    metadata = []
    with Progress() as progress:
        metadata += collect_by_pattern("*left*.png", progress, folders, df_2d)
        metadata += collect_by_pattern("*right*.png", progress, folders, df_2d)
    df_meta = pd.DataFrame(metadata).sort_values("datetime")

    # Add fog data
    df_meta = table.add_fog_intensity(df_meta)

    # Check if all combinations of weather and distance are present
    aux.summary_count(df_meta[df_meta["sensor"] == "cam_l"], title="cam_l")
    aux.summary_count(df_meta[df_meta["sensor"] == "cam_r"], title="cam_r")

    # Export
    if not args.debug:
        table.save(df_meta, Path(__file__).stem, engine, overwrite=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    main(argparser.parse_args())
