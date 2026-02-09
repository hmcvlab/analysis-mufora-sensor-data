"""
Created on Thu Oct 24 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger as log
from rich.progress import Progress

from mufora import aux, data, detect, draw, pose, settings, table

DATA_ROOT = data.root() / "rawdata"
DB_NAME = Path(__file__).stem


def _row2filename(row: pd.Series, suffix: str) -> str:
    """Convert row to filename in the tmp/ dir for debugging."""
    intensity = np.clip(row.intensity, 0, 150).round().astype(int)
    name = f"{row.sensor}_{row.weather}_{row.distance}m_{intensity}_{row.duration}s"
    filename = Path(f"tmp/{DB_NAME}/{name}.{suffix}")
    filename.parent.mkdir(parents=True, exist_ok=True)
    return str(filename)


def _eval_image(filename: Path, row: pd.Series, args: argparse.Namespace):
    """Process an image."""
    file_calib = filename.parent / "calib.yaml"

    # Load settings and image
    config = settings.from_files(file_calib)
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)

    # Detect circle in 3D with GT 2D annotations
    gt = row.rename(
        {
            "r_gt_px": "radius",
            "x_gt_px": "x",
            "y_gt_px": "y",
        }
    )
    entropy = detect.normalized_pixel_entropy(image, gt)
    gt["entropy"] = entropy

    try:
        # Detect circles and store results
        circles = detect.deep_circles(image)
        circle = circles.iloc[0]
    except RuntimeWarning as e:
        log.warning(f"Skipping {filename}: {e}")
        return {
            "entropy": entropy,
            "pos_m": np.nan,
            "pos_px": np.nan,
            "radius_px": np.nan,
            "n_total": image.shape[0] * image.shape[1],
        }

    # Save sample image in debug mode
    if args.debug:
        image = draw.circles(image, circles)
        cv2.imwrite(_row2filename(row, "png"), image)

    return {
        "entropy": entropy,
        "pos_m": pose.from_circle(gt, config),
        "pos_px": circle[["x", "y"]].tolist(),
        "radius_px": circle["radius"],
        "n_total": image.shape[0] * image.shape[1],
    }


def _process_df(df_eval: pd.DataFrame, sensor: str, args: argparse.Namespace, engine):
    """Run evaluation on a dataframe and return the dataframe."""
    df = df_eval[df_eval["sensor"] == sensor]

    # If debug take 100 random rows
    if args.debug:
        aux.summary_count(df)
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
            df.loc[idx, ["datetime"]] = table.file2datetime(filename)
            res = _eval_image(filename, row, args)
            df.loc[idx, ["metric"]] = res["entropy"]
            df.loc[idx, ["x_m", "y_m", "z_m"]] = res["pos_m"]
            df.loc[idx, ["x_px", "y_px"]] = res["pos_px"]
            df.loc[idx, ["radius_px"]] = res["radius_px"]
            df.loc[idx, ["n_total"]] = res["n_total"]

    # Adjust data types
    df[["metric", "x_m", "y_m", "z_m", "x_px", "y_px", "radius_px"]] = df[
        ["metric", "x_m", "y_m", "z_m", "x_px", "y_px", "radius_px"]
    ].astype(float)
    df[["n_total"]] = df[["n_total"]].astype(int)

    log.debug(f"Results:\n{df.to_string()}")
    aux.summary_count(df[df["sensor"] == sensor], title=sensor)

    if not args.debug:
        table.save(df, DB_NAME, engine, overwrite=False)


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
    df_meta = table.query2df("SELECT * FROM metadata_2d", engine)
    aux.summary_count(df_meta)

    # Run evaluation by detecting circles/spheres
    log.info(f"Evaluating {len(df_meta)} samples...")

    # Split dataframe into 10 parts
    if args.sensor in ["all", "cam_l"]:
        _process_df(df_meta, "cam_l", args, engine)
    if args.sensor in ["all", "cam_r"]:
        _process_df(df_meta, "cam_r", args, engine)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--sensor", choices=["all", "cam_l", "cam_r"], default="all")
    main(argparser.parse_args())
