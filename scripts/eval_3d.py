"""
Created on Thu Oct 24 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from loguru import logger as log
from rich.progress import Progress

from aux import data, detect, draw, filters, settings, summary, table

DATA_ROOT = data.root() / "rawdata"
DB_NAME = Path(__file__).stem


def _row2filename(row: pd.Series, suffix: str) -> str:
    """Convert row to filename in the tmp/ dir for debugging."""
    intensity = np.clip(row.intensity, 0, 150).round().astype(int)
    name = f"{row.sensor}_{row.weather}_{row.distance}m_{intensity}_{row.duration}s"
    filename = Path(f"tmp/{DB_NAME}/{name}.{suffix}")
    filename.parent.mkdir(parents=True, exist_ok=True)
    return str(filename)


def _eval_point_cloud(filename: Path, row: pd.Series, args: argparse.Namespace):
    """Process a point cloud."""
    t0 = datetime.now()
    file_calib = filename.parent / "calib.yaml"
    config = settings.from_files(file_calib)

    t1 = datetime.now()
    pcd = np.array(o3d.io.read_point_cloud(str(filename)).points)

    t2 = datetime.now()
    gt = row.rename(
        {
            "r_gt_m": "radius",
            "x_gt_m": "x",
            "y_gt_m": "y",
            "z_gt_m": "z",
        }
    )
    center_m = gt[["x", "y", "z"]].astype(float).to_numpy()
    res = filters.pcd_by_cone(
        pcd, config.radius_m + config.inlier_threshold_m, center_m
    )
    t3 = datetime.now()
    try:
        sphere = detect.sphere(res["pcd"], config, gt_center=center_m)
    except RuntimeWarning as e:
        log.warning(f"Skipping {filename}: {e}")
        t4 = datetime.now()
        log.debug(summary.times_summary([t0, t1, t2, t3, t4], "Time with exception"))
        return {
            "inlier_ratio": np.nan,
            "radius_m": np.nan,
            "pos_m": np.nan,
            "n_total": pcd.shape[0],
        }
    t4 = datetime.now()
    log.debug(summary.times_summary([t0, t1, t2, t3, t4], "Time for sphere detection"))

    if sphere.inlier_ratio > 1.0:
        log.error(f"Sphere inlier ratio > 1.0:\n{sphere}")

    # Save sample point cloud in debug mode
    if args.debug:
        center = center_m if np.isnan(sphere.center).any() else sphere.center
        radius = config.radius_m if np.isnan(sphere.radius) else sphere.radius
        pcd_o3d = draw.sphere(pcd, center, radius)

        o3d.io.write_point_cloud(_row2filename(row, "pcd"), pcd_o3d)

    return {
        "inlier_ratio": sphere.inlier_ratio,
        "radius_m": sphere.radius,
        "pos_m": sphere.center,
        "n_total": pcd.shape[0],
    }


def _process_df(df_eval: pd.DataFrame, sensor: str, args: argparse.Namespace, engine):
    """Run evaluation on a dataframe and return the dataframe."""
    df = df_eval[df_eval["sensor"] == sensor]

    # If debug take 100 random rows
    if args.debug:
        summary.summary_count(df)
        df = df.sample(10)

    log.debug(f"Processing:\n{df.to_string()}")

    with Progress() as progress:
        for idx, row in progress.track(
            df.iterrows(),
            total=len(df),
            description=f"{sensor}...",
        ):
            # Load settings
            filename = (DATA_ROOT / f"{row.filename}").resolve()
            res = _eval_point_cloud(filename, row, args)
            df.loc[idx, ["datetime"]] = table.file2datetime(filename)
            df.loc[idx, ["metric"]] = res["inlier_ratio"]
            df.loc[idx, ["x_m", "y_m", "z_m"]] = res["pos_m"]
            df.loc[idx, ["radius_m"]] = res["radius_m"]
            df.loc[idx, ["n_total"]] = res["n_total"]

    # Adjust data types
    df[["metric", "x_m", "y_m", "z_m", "radius_m"]] = df[
        ["metric", "x_m", "y_m", "z_m", "radius_m"]
    ].astype(float)
    df[["n_total"]] = df[["n_total"]].astype(int)

    log.debug(f"Results:\n{df.to_string()}")

    if not args.debug:
        table.update(df, DB_NAME, engine, overwrite=False)


def main(args: argparse.Namespace):
    """
    Main function
    """
    # Set log level
    log.remove()
    log.add(sys.stderr, level="DEBUG" if args.debug else "WARNING")

    log.info("Starting script...")
    engine = table.engine(database="weather")

    # Collect metadata
    df_meta = table.query2df("SELECT * FROM metadata_3d", engine)

    # Run evaluation by detecting circles/spheres
    log.info(f"Evaluating {len(df_meta)} samples...")

    # Split dataframe into 10 parts
    if args.sensor == "all" or args.sensor == "qb2_0":
        _process_df(df_meta, "qb2_0", args, engine)
    if args.sensor == "all" or args.sensor == "qb2_1":
        _process_df(df_meta, "qb2_1", args, engine)

    log.info("Done.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--sensor", choices=["all", "qb2_0", "qb2_1"], default="all")
    main(argparser.parse_args())
