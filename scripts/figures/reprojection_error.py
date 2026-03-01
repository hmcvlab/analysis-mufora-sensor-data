"""
Created on Wed Jan 08 2025
Copyright (c) 2025 Munich University of Applied Sciences
"""

import argparse
import json
import pathlib
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log
from matplotlib.lines import Line2D
from rich.progress import Progress

from mufora import data, draw

ROOT = pathlib.Path(__file__).parent.parent.parent
NAME = Path(__file__).stem


def _reprojection_errors(df_2d, df_3d, calib_data, cols_3d_lidar) -> dict:
    """Compute reprojection errors based on detections from database."""
    # Relevant columns
    cols_2d_lidar = ["u_qb2", "v_qb2"]
    cols_2d_cam = ["x_gt_px", "y_gt_px"]

    results = {}
    with Progress() as progress:
        for date, value in progress.track(calib_data.items()):
            progress.console.print(f"{date}: {value.keys()}")

            intrinsics = np.array(value["intrinsics"]["K"])
            distortion = np.array(value["intrinsics"]["dist"])
            progress.console.print(f"Focal length: {intrinsics[0][0]}")
            progress.console.print(f"Distortion: {distortion}")

            extrinsics = {}
            extrinsics["qb2_0"] = np.array(value["lidar0_to_cam0"]).reshape(4, 4)
            extrinsics["qb2_1"] = np.array(value["lidar1_to_cam0"]).reshape(4, 4)
            progress.console.print(f"t(qb2_0): {extrinsics['qb2_0'][:3, 3]}")
            progress.console.print(f"t(qb2_1): {extrinsics['qb2_1'][:3, 3]}")

            # Get pixel coordinates
            # Keep only relevant columns
            df_cam = df_2d[(df_2d["sensor"] == "cam_l") & (df_2d["date"] == date)][
                ["datetime"] + cols_2d_cam
            ]

            results[date] = {}
            df_sub = df_3d[df_3d["date"] == date]
            for sensor in ["qb2_0", "qb2_1"]:
                df_lidar = df_sub[(df_sub["sensor"] == sensor)]
                df_lidar = df_lidar[["datetime"] + cols_3d_lidar]

                # Check that dataframe is not empty
                if df_lidar.empty:
                    df_debug = df_3d[["sensor", "date"]].pivot_table(
                        index="date",
                        columns="sensor",
                        aggfunc="size",
                        fill_value=0,
                    )
                    raise ValueError(f"Empty dataframe for {sensor}:\n{df_debug}")

                # Deproject points into image frame
                xyz = df_lidar[cols_3d_lidar].to_numpy()
                rvec = cv2.Rodrigues(extrinsics[sensor][:3, :3])[0]
                tvec = extrinsics[sensor][:3, 3]

                progress.console.print(
                    f"{sensor}: rvec={rvec.shape}, tvec={tvec.shape}, xyz={xyz.shape}"
                )
                uv_lidar = cv2.projectPoints(
                    xyz,
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=intrinsics,
                    distCoeffs=distortion,
                )
                df_lidar[cols_2d_lidar] = uv_lidar[0].reshape(-1, 2)

                # Merge within time limit
                df_merged = pd.merge_asof(
                    df_cam,
                    df_lidar[["datetime"] + cols_2d_lidar],
                    on="datetime",
                    direction="nearest",
                    tolerance=pd.Timedelta("10s"),
                ).dropna()

                reprojection_error = np.linalg.norm(
                    df_merged[cols_2d_lidar].to_numpy()
                    - df_merged[cols_2d_cam].to_numpy(),
                    axis=1,
                )
                reprojection_error = reprojection_error[~np.isnan(reprojection_error)]
                results[date][sensor] = list(reprojection_error)

    return results


def _box_info(x_vals: np.ndarray) -> str:
    """Log information for box plot."""
    x_median = np.median(x_vals)
    q1 = np.percentile(x_vals, 25)
    q3 = np.percentile(x_vals, 75)
    x_iqr = q3 - q1
    x_box = [q1, q3]
    x_whisker = [q1 - 1.5 * x_iqr, q3 + 1.5 * x_iqr]
    lower_edges = np.round([x_whisker[0], x_box[0]], 2)
    upper_edges = np.round([x_box[1], x_whisker[1]])
    return f"{lower_edges} | {x_median:.2f} | {upper_edges}"


def _get_data(args: argparse.Namespace):
    """Helper function to get data from SQL database."""
    calib_data = {}
    for file in sorted(ROOT.joinpath("data/calib").glob("*.json")):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            year, month, day = file.stem.split("-")
            calib_data[f"{day}.{month}.{year}"] = data

    df_eval_2d = pd.read_csv(args.file_eval_2d).sort_values("datetime")
    df_eval_3d = pd.read_csv(args.file_eval_3d).sort_values("datetime")
    df_eval_2d["datetime"] = pd.to_datetime(df_eval_2d["datetime"], utc=True)
    df_eval_3d["datetime"] = pd.to_datetime(df_eval_3d["datetime"], utc=True)

    # Add date str columns
    df_eval_2d["date"] = df_eval_2d["datetime"].dt.strftime("%d.%m.%Y")
    df_eval_3d["date"] = df_eval_3d["datetime"].dt.strftime("%d.%m.%Y")

    # Compute results
    cols_3d_gt = ["x_gt_m", "y_gt_m", "z_gt_m"]
    cols_3d_meas = ["x_m", "y_m", "z_m"]
    results_gt = _reprojection_errors(df_eval_2d, df_eval_3d, calib_data, cols_3d_gt)

    # Remove rows if x/y/z_gt_m is more than 10 cm away from x/y/z_m
    n_samples = df_eval_3d.shape[0]
    df_eval_3d["distance"] = np.linalg.norm(
        df_eval_3d[["x_gt_m", "y_gt_m", "z_gt_m"]].to_numpy()
        - df_eval_3d[["x_m", "y_m", "z_m"]].to_numpy(),
        axis=1,
    )
    df_eval_3d = df_eval_3d[df_eval_3d["distance"] < 0.15]
    log.info(f"Removed {n_samples - df_eval_3d.shape[0]}/{n_samples} samples.")
    results_meas = _reprojection_errors(
        df_eval_2d, df_eval_3d, calib_data, cols_3d_meas
    )

    return results_gt, results_meas


def main(args: argparse.Namespace):
    """Entry point."""
    results_gt, results_meas = _get_data(args)

    # Create box plot for each day and group by sensor
    summaries = [[], []]
    titles = ["Labelled positions", "Refined positions"]
    colors = [next(draw.COLORS) for _ in range(2)]
    fig, ax = plt.subplots(ncols=2, figsize=(8.4, 3.9), sharey=True)
    for j, results in enumerate([results_gt, results_meas]):
        all_values = [np.concatenate(list(x.values())) for x in results.values()]
        all_values = np.concatenate(all_values)
        log.info(f"#{j}: Overall median: {np.median(all_values):.2f}")
        log.info(f"#{j}: Overall std: {np.std(all_values):.2f}")

        # Draw vertical line
        ax[j].axhline(np.median(all_values), color="red", linestyle="--")

        for idx, (date, sub_data) in enumerate(results.items()):
            x_vals = list(sub_data.values())
            pos = 2 * idx + 1
            positions = [pos - 0.2, pos + 0.5]
            for x_val, pos, color in zip(x_vals, positions, colors):
                bplot = ax[j].boxplot(
                    x_val,
                    positions=[pos],
                    vert=True,
                    widths=0.5,
                    patch_artist=True,
                    medianprops={"color": "black"},
                )

                all_bplot_data = []
                for key, value in bplot.items():
                    if key not in ["whiskers", "medians", "caps"]:
                        continue
                    for idx, entry in enumerate(value):
                        all_bplot_data.append({f"{key}_{idx}": entry.get_ydata()})

                whiskers_x = [
                    whisker.get_xdata().round(2) for whisker in bplot["whiskers"]
                ]
                whiskers_x_grouped = list(zip(whiskers_x[::2], whiskers_x[1::2]))
                for x_values, whiskers in zip(x_vals, whiskers_x_grouped):
                    info = _box_info(x_values)
                    log.info(f"#{j}: {date} np  = {info}")
                    log.info(
                        f"#{j}: {date} box = {whiskers[0][::-1]} |    | {whiskers[1]}"
                    )

                # Patch face colors
                for patch in bplot["boxes"]:
                    patch.set_facecolor(color)

            # Collect data fro summary
            summaries[j].append(
                {
                    "date": date,
                    "sensor": "qb2_0",
                    "n": len(x_vals[0]),
                    "median": np.median(x_vals[0]),
                    "std": np.std(x_vals[0]),
                }
            )
            summaries[j].append(
                {
                    "date": date,
                    "sensor": "qb2_1",
                    "n": len(x_vals[1]),
                    "median": np.median(x_vals[1]),
                    "std": np.std(x_vals[1]),
                }
            )

        ax[j].set_xticks([2 * i + 1.5 for i in range(len(results))])
        ax[j].set_title(titles[j])
        ax[j].set_ylim(-1, 17)
        ax[j].grid()

    # Create dataframe and print summary
    for j, summary in enumerate(summaries):
        df_sum = (pd.DataFrame(summary).groupby(["sensor", "date"])).max()
        df_sum = df_sum.round(2)
        log.info(f"#{j} Summary:\n{df_sum.to_string(index=True)}")

    # Common x-axis-label
    # fig.supylabel("Reprojection error in px")
    print(results_gt.keys())
    date_labels = [date[:-5] for date in results_gt.keys()]
    ax[0].set_ylabel("Reprojection error in px")
    ax[0].set_xticklabels(date_labels, rotation=0)
    ax[1].set_xticklabels(date_labels, rotation=0)

    # Common legend
    legend_handles = [
        mpatches.Patch(facecolor=colors[0], edgecolor="black", label=r"$L_s$"),
        mpatches.Patch(facecolor=colors[1], edgecolor="black", label=r"$L_d$"),
        Line2D([0], [0], color="red", linestyle="--", label="Overall median"),
    ]

    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.08)
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncols=3
    )
    fig.savefig(ROOT.joinpath(f"tmp/{NAME}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file-eval-2d", type=Path, default=data.root() / "analysis/eval_2d.csv"
    )
    argparser.add_argument(
        "--file-eval-3d", type=Path, default=data.root() / "analysis/eval_3d.csv"
    )
    main(argparser.parse_args())
