"""
Created on Tue Jul 08 2025
Copyright (c) 2025 Munich University of Applied Sciences

Filter functions used for all kind of purposes.
"""

import numpy as np
import pandas as pd
from loguru import logger as log

from aux import settings


def angles(center: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute angle between center and points"""
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)

    angles_pts = np.dot(points, center) / (
        np.linalg.norm(points, axis=1) * np.linalg.norm(center)
    )
    return np.arccos(angles_pts)


def pcd_by_distance(pcd: np.ndarray, config: settings.Calibration) -> pd.Series:
    """Function to filter the point cloud by distance from the origin."""
    res = pd.Series(index=["flag", "idx", "pcd", "n"], data=[np.array([])] * 3 + [0])
    if pcd.ndim > 1:
        norm = np.linalg.norm(pcd, axis=1)
        res.flag = (config.min_distance_m < norm) & (norm < config.max_distance_m)
        res.idx = np.squeeze(np.where(res.flag))
        res.pcd = pcd[res.idx]
        res.n = res.idx.size if res.ndim >= 1 else 0

    return res


def pcd_by_roi(pcd: np.ndarray, radius: float, center: np.ndarray) -> pd.Series:
    """Function that splits the point cloud into a region of interest and the rest."""
    res = pd.Series(index=["flag", "idx", "pcd", "n"], data=[np.array([])] * 3 + [0])
    if pcd.ndim > 1:
        distance = np.linalg.norm(pcd - center, axis=1)
        res.flag = distance < radius
        res.idx = np.squeeze(np.where(res.flag))
        res.pcd = pcd[res.idx]
        res.n = res.idx.size if res.ndim >= 1 else 0

    log.debug(f"Filtered pcd by roi:\n{res}")
    return res


def pcd_by_cone(pcd: np.ndarray, radius: float, center: np.ndarray) -> pd.Series:
    """Function that splits the point cloud into a region of interest which is defined
    by a cone. This cone is spanned over the center and the radius of the radius of
    the sphere (defined in the settins.Calibration)."""
    res = pd.Series(index=["flag", "idx", "pcd", "n"], data=[np.array([])] * 3 + [0])
    if np.isnan(center).any():
        log.warning("Center is NaN")

    # Find an offset point that is perpendicular to the center
    direction = np.cross(center, np.array([0, 0, 1]))
    direction /= np.linalg.norm(direction)
    offset = center + radius * direction

    # Compute angle between center and points
    angles_pts = angles(center, pcd)
    angle_offset = angles(center, offset)
    res.flag = (angles_pts - angle_offset) < 1e-3
    res.idx = np.squeeze(np.where(res.flag))
    res.pcd = pcd[res.idx]
    res.n = res.idx.size if res.ndim >= 1 else 0

    return res
