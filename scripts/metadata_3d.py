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
    df_3d: pd.DataFrame,
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
        df_3d_tmp = df_3d[
            (df_3d["date"] == date)
            & (df_3d["distance"] == distance)
            & (df_3d["weather"] == weather)
        ]

        if df_3d_tmp.empty:
            progress.console.print(f"Missing data 3D {dirname}")
            continue

        t_start = table.file2datetime(files[0])
        for idx in indexes:
            file = files[idx]
            t_tmp = table.file2datetime(file)
            duration = t_tmp - t_start
            sensor = table.file2sensor(file)

            # Get ground truth
            idx_gt_3d = df_3d_tmp[df_3d_tmp["sensor"] == sensor].index
            if len(idx_gt_3d) != 1 and weather not in ["fog", "rain"]:
                raise RuntimeError(f"No unique ground truth:\n{df_3d.loc[idx_gt_3d]}")

            tmp_metadata = {
                "datetime": t_tmp,
                "sensor": sensor,
                "weather": weather,
                "distance": distance,
                "intensity": table.folder2intensity(dirname),
                "duration": duration.seconds,
                "r_gt_m": float(df_3d.loc[idx_gt_3d, "radius_m"].mean()),
                "x_gt_m": float(df_3d.loc[idx_gt_3d, "x_m"].mean()),
                "y_gt_m": float(df_3d.loc[idx_gt_3d, "y_m"].mean()),
                "z_gt_m": float(df_3d.loc[idx_gt_3d, "z_m"].mean()),
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
    df_3d = table.query2df("SELECT * FROM gt_3d", engine)

    log.info(f"Loading gt_3d: {len(df_3d)}")
    log.info(f"Describe gt_3d: {df_3d.dtypes}")
    aux.summary_count(df_3d)

    # Merge data by updating it with the new
    folders = data.relevant_raw_folders()

    if args.debug:
        folders = list(filter(lambda x: "2024-02-28" in x.name, folders))[:2]

    # Iterate over folders
    metadata = []
    with Progress() as progress:
        metadata += collect_by_pattern("*qb2_0*.pcd", progress, folders, df_3d)
        metadata += collect_by_pattern("*qb2_1*.pcd", progress, folders, df_3d)
    df_meta = pd.DataFrame(metadata).sort_values("datetime")

    # Add fog data
    df_meta = table.add_fog_intensity(df_meta)

    # Check if all combinations of weather and distance are present
    aux.summary_count(df_meta[df_meta["sensor"] == "qb2_0"], title="qb2_0")
    aux.summary_count(df_meta[df_meta["sensor"] == "qb2_1"], title="qb2_1")

    # Export
    if not args.debug:
        table.save(df_meta, Path(__file__).stem, engine)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    main(argparser.parse_args())
