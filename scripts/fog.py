"""
Created on Thu Nov 07 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from loguru import logger as log

from mufora import data, table

ROOT = Path(__file__).parent.parent.parent


def main(args: argparse.Namespace):
    """Main function."""
    # File to import/export
    files_fog = sorted(list(args.dir_input.glob("*_fog.csv")))

    # Load additional data
    df_fog = pd.DataFrame()
    for file in sorted(files_fog):
        tmp = pd.read_csv(file)
        tmp["file_fog"] = file.stem
        df_fog = pd.concat([df_fog, tmp])
    df_fog["datetime"] = df_fog["datetime"].apply(
        lambda x: re.sub(r"[A-Z]|\(|\)", "", x)
    )
    log.info(f"Finished loading fog data: {len(df_fog)} entries")

    # Convert timestamps recorded in CET to UTC
    df_fog["datetime"] = pd.to_datetime(df_fog["datetime"], utc=True)
    df_fog = df_fog.rename(columns={"t_delta": "fog_duration_ms"})

    # Smooth fog data over +/- 30 seconds
    timedelta = pd.Timedelta(seconds=60)
    df_fog["visibility"] = (
        df_fog.set_index("datetime")["visibility"].rolling(timedelta).mean()
    ).to_list()
    log.info(f"Finished smoothing fog data over {timedelta}")

    df_fog = df_fog.rename(columns={"visibility": "intensity"})
    df_fog = df_fog.sort_values("datetime")
    df_fog = df_fog.drop_duplicates()

    # Export
    log.info(f"Exporting {table=} to {args.file_output}...")
    table.save(df_fog, args.file_output)
    log.info("Done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir-input", type=Path, default=ROOT / "data/fog")
    argparser.add_argument(
        "--file-output", type=Path, default=data.root() / "weather/fog.csv"
    )
    main(argparser.parse_args())
