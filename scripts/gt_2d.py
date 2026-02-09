"""
Created on Wed Oct 30 2024
Copyright (c) 2024 Munich University of Applied Sciences

Script to generate a ground truth csv file from coco annotations
"""

import argparse
import json
import pprint
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger as log
from rich import progress

from mufora import aux, data, table


def draw_cirlce(img: np.ndarray, row: dict, filename: str):
    """Compute pixel entropy"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        tuple([int(row["x_px"]), int(row["y_px"])]),
        int(row["radius_px"]),
        255,
        -1,
    )

    # Save sample image
    mask_green = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))
    img_vis = cv2.addWeighted(img, 0.5, mask_green, 0.5, 0)
    cv2.imwrite(str(filename), img_vis)


def main(args: argparse.Namespace):
    """Main function."""

    with open(args.file_input, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    df_annos = pd.DataFrame(data_json["annotations"])
    df_images = pd.DataFrame(data_json["images"])

    # Check that there is only one annotation per image
    if len(df_annos["image_id"].unique()) != len(df_annos):
        df_annos["file_name"] = df_annos["image_id"].map(df_images["file_name"])
        non_unique = df_annos[["file_name", "image_id"]].groupby("file_name").count()
        non_unique.rename(columns={"image_id": "count"}, inplace=True)
        info = non_unique[non_unique["count"] > 1].to_dict()
        log.error(f"Non unique image ids:\n{pprint.pformat(info)}")
        return

    # Set indexes
    df_annos.set_index("image_id", inplace=True)
    df_images.set_index("id", inplace=True)

    data_processed = []
    for idx, anno in progress.track(
        df_annos.iterrows(), description="Processing", total=len(df_annos)
    ):
        img_info = df_images.loc[idx]
        folder = img_info.file_name.split(".")[0].replace("_png", "")
        x, y, w, h = anno.bbox

        row = {
            "date": table.folder2date(folder),
            "time": table.folder2time(folder),
            "sensor": table.folder2sensor(folder),
            "weather": table.folder2weather(folder),
            "distance": table.folder2distance(folder),
            "intensity": table.folder2intensity(folder),
            "x_px": np.round(x + w / 2, 2),
            "y_px": np.round(y + h / 2, 2),
            "radius_px": min(w, h) / 2,
        }

        data_processed.append(row)

        # In debug more store image visualizations in tmp
        if args.debug:
            dir_debug = Path("tmp/gt_2d")
            dir_debug.mkdir(exist_ok=True, parents=True)
            img = cv2.imread(str(file_coco.parent / img_info.file_name))
            draw_cirlce(img, row, dir_debug / f"{folder}.png")

    df = (
        pd.DataFrame(data_processed)
        .sort_values(["weather", "distance", "intensity", "date", "time"])
        .reset_index(drop=True)
    )
    aux.summary_count(df)
    df["distance"] = df["distance"].astype(int)
    df["intensity"] = df["intensity"].astype(float)
    log.info(df)
    log.info(df.dtypes)

    # Remove duplicates and only keep the last
    relevant_columns = ["weather", "distance", "intensity", "date", "sensor"]
    indexes = df[relevant_columns].drop_duplicates(keep="last").index

    # Also remove nans
    indexes = df[df["weather"].notnull()].index.intersection(indexes)

    df_keep = df.loc[indexes]
    df_remove = df[~df.index.isin(indexes)]
    log.info(f"Final dataframe:\n{df_keep.to_string()}")
    log.warning(f"Removed duplicated data:\n{df_remove.to_string()}")
    aux.summary_count(df_keep[df_keep["sensor"] == "cam_l"], title="cam_l")
    aux.summary_count(df_keep[df_keep["sensor"] == "cam_r"], title="cam_r")

    # Export
    table.save(df_keep, args.file_output)


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
