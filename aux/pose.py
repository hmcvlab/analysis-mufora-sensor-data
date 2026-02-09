"""
Created on Thu Sep 26 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def cart2hom(matrix: np.ndarray, axis=0):
    """Add a column of 1 ones to the matrix if axis=1, or a row of 1 ones if axis=0."""
    matrix = np.squeeze(matrix)
    if axis == 1:
        matrix = matrix.T

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif not matrix.shape[1] in (2, 3):
        raise RuntimeWarning(f"Matrix must have 2 or 3 columns, but has {matrix.shape}")
    return np.squeeze(np.hstack([matrix, np.ones((matrix.shape[0], 1))]))


def hom2cart(matrix: np.ndarray, axis=0):
    """Divide and remove the row last column if axis=1, or the column last row if
    axis=0."""
    matrix = np.squeeze(matrix)
    if axis == 1:
        matrix = matrix.T
    if matrix.ndim == 1:
        matrix = matrix[:-1] / matrix[-1]
        matrix = matrix.reshape(1, -1)
    else:
        matrix = matrix[:, :-1] / matrix[:, -1].reshape(-1, 1)

    if not matrix.shape[1] in (2, 3):
        raise RuntimeWarning(f"Matrix must have 2 or 3 columns, but has {matrix.shape}")
    return np.squeeze(matrix)


def from_vectors(
    rotation: List[float] = None, translation: List[float] = None
) -> np.ndarray:
    """Convert rotation and translation to a 4x4 matrix."""
    rotation = rotation or [0, 0, 0]
    translation = translation or [0, 0, 0]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def from_point_pairs(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Compute pose from point pairs"""
    pts_a = np.array(pts_a, copy=False)
    pts_b = np.array(pts_b, copy=False)

    if pts_a.shape != pts_b.shape:
        raise RuntimeWarning(
            f"Point sets must match shape: {pts_a.shape} != {pts_b.shape}"
        )

    # Align the point sets to prepare for SVD
    mean_a = np.mean(pts_a, axis=0)
    mean_b = np.mean(pts_b, axis=0)
    center_a = pts_a - mean_a
    center_b = pts_b - mean_b

    # SVD of the projection of b onto a
    center_proj = np.dot(center_b.T, center_a)
    vec_left, _, vec_right = np.linalg.svd(center_proj)

    # Compute rotation from singular vectors
    rot = np.dot(vec_right.T, vec_left.T)
    if np.linalg.det(rot) < 0:
        vec_right[2, :] *= -1
        rot = np.dot(vec_right.T, vec_left.T)

    # Compute translation by projecting the mean of b on a
    trans = mean_a - np.dot(rot, mean_b)

    # Combine matrices
    tmp_pose = np.eye(4)
    tmp_pose[:3, :3] = rot
    tmp_pose[:3, 3] = trans
    return tmp_pose


def from_circle(circle: pd.Series, config) -> np.ndarray:
    """Compute translation of detected sphere from center, radius and camera intrinsics.

    Args:
        circle (pd.Series): Detected circle containing 'x', 'y', 'radius'
        config (settings.Detect): Calibration settings

    Returns:
        np.ndarray: Pose matrix
    """
    # Extract parameters
    radius_m = config.radius_m
    intrinsic = config.intrinsic

    # Generate points
    points = circle2points(circle)

    # Compute translation
    points_hom = cart2hom(points) @ np.linalg.inv(intrinsic).T
    points_nor = points_hom / np.linalg.norm(points_hom, axis=1).reshape(-1, 1)

    # Solve equation system
    coeff = np.linalg.lstsq(points_nor, np.ones(points_nor.shape[0]), rcond=None)[0]
    coeff_nor = 1 / np.linalg.norm(coeff)

    return (radius_m / np.sqrt(1 - coeff_nor**2)) * (coeff_nor * coeff)


def circle2points(circle: pd.Series) -> np.ndarray:
    """Generate points for drawing and distance calculation"""
    # Extract parameters
    center = (circle["x"], circle["y"])
    radius = circle["radius"]

    # Generate points
    pts = [
        (radius * np.cos(angle) + center[0], radius * np.sin(angle) + center[1])
        for angle in np.linspace(0, 2 * np.pi, 30)
    ]
    return np.array(pts).astype(int).reshape(-1, 2)
