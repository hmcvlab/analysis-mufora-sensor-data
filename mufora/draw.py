"""
Created on Wed Sep 25 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from loguru import logger as log

from mufora import filters, pose

COLORS = itertools.cycle(plt.colormaps["tab10"].colors)
MARKERS = itertools.cycle(["o", "v", "x", "^", "<", ">", "s", "p", "P", "+"])
COLOR_MAP = {k: next(COLORS) for k in range(5, 55, 5)}
MARKER_MAP = {k: next(MARKERS) for k in range(5, 55, 5)}


def circles(img: np.ndarray, circles_df: pd.DataFrame) -> np.ndarray:
    """Save image in tmp folder"""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for _, row in circles_df.head(3).iterrows():
        # draw the outer circle
        font = cv2.FONT_HERSHEY_SIMPLEX
        center = tuple(row[["x", "y"]])
        radius = int(row["radius"])
        color = list(map(int, np.squeeze(row["color"])))
        red = (0, 0, 255)
        cv2.circle(vis, center, radius, color, 1)

        # draw the center of the circle
        cv2.circle(vis, center, 2, red, 1)
        cv2.putText(vis, f"({center})", center, font, 0.3, red)
        if "t" in row:
            txt_3d = row["t"]
            cv2.putText(vis, txt_3d, (center[0], center[1] + 10), font, 0.3, red)

        # Draw points
        for point in pose.circle2points(row):
            cv2.circle(vis, tuple(point), 1, (0, 255, 255), 1)

    return vis


def bounding_box_points(center: np.ndarray, radius: float) -> o3d.geometry.PointCloud:
    """Draw the bounding box of a sphere."""
    if np.isnan(center).any() or np.isnan(radius):
        raise RuntimeError("Center or radius is NaN")

    bbox = o3d.geometry.AxisAlignedBoundingBox(center - radius, center + radius)
    box_pts = np.array(bbox.get_box_points(), dtype=np.float64)

    # Add intermediate points between the corners
    idx_corner_pairs = itertools.combinations(range(8), 2)
    for i0, i1 in idx_corner_pairs:
        pt0 = box_pts[i0]
        pt1 = box_pts[i1]
        if np.abs(np.linalg.norm(pt0 - pt1) - 2 * radius) > 1e-2:
            continue
        pts_new = np.linspace(box_pts[i0], box_pts[i1], 20)
        box_pts = np.concatenate([box_pts, pts_new])

    return box_pts


def sphere(
    pcd: np.ndarray, center: np.ndarray, radius: float
) -> o3d.geometry.PointCloud:
    """Draw the points of a sphere in a different color"""

    # Paint inliers in green
    idx_cone = filters.pcd_by_cone(pcd, radius, center).idx
    idx_roi = filters.pcd_by_roi(pcd, radius, center).idx

    # Add new bounding box points here
    bbox_pts = bounding_box_points(center, radius)
    pcd = np.concatenate([pcd, bbox_pts])
    idx_box = np.arange(pcd.shape[0] - bbox_pts.shape[0], pcd.shape[0])

    log.info(f"n_inliers_roi: {idx_roi.shape}")
    log.info(f"n_inliers_cone: {idx_cone.shape}")
    log.info(f"n_box: {idx_box.shape}")
    log.info(f"n_all: {pcd.shape[0]}")

    # Highlight points in point cloud
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(pcd.astype(np.float64))
    colors = np.tile([0.5, 0.5, 0.5], (pcd.shape[0], 1))
    for indexes, color in zip(
        [idx_cone, idx_roi, idx_box],
        [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]],
    ):
        if indexes.size == 0:
            continue
        colors[indexes] = color
    out_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Combine point cloud and inliers
    log.debug(f"Draw: {str(out_pcd)}")
    return out_pcd
