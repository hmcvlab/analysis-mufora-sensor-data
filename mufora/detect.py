"""
Created on Sat Sep 14 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import torch
from loguru import logger as log
from sklearn.neighbors import KDTree
from transformers import DetrForObjectDetection, DetrImageProcessor

from mufora import filters
from mufora.settings import Calibration

ROOT = Path(__file__).parent.resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_PROC = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

DIR_MODEL = f"{ROOT}/model"
log.info(f"Loading model from {DIR_MODEL}")
MODEL = DetrForObjectDetection.from_pretrained(DIR_MODEL)


def pixel_variance(img: np.ndarray, row: pd.Series):
    """Compute pixel variance"""
    mask = np.zeros_like(img)
    cv2.circle(mask, (row["x"], row["y"]), row["radius"], 255, -1)
    pixels = img[mask == 255]
    return np.var(pixels)


def normalized_pixel_entropy(img: np.ndarray, row: pd.Series):
    """Compute pixel entropy"""
    x = np.round(row["x"]).astype(int)
    y = np.round(row["y"]).astype(int)
    radius = np.round(row["radius"]).astype(int)
    mask = np.zeros_like(img)
    cv2.circle(mask, (x, y), radius, 255, -1)
    pixels = img[mask == 255]

    _, counts = np.unique(pixels, return_counts=True)
    entropy = scipy.stats.entropy(counts, base=2) / np.log2(np.min([256, len(pixels)]))
    if 0 > entropy > 1:
        log.error(f"Entropy out of range: {entropy}")
    if len(pixels) < 256:
        log.warning(f"Low pixel count: {len(pixels)}")
    return entropy


def circles(img: np.ndarray, config: Calibration = Calibration()) -> pd.DataFrame:
    """Detect circles using hough transformation"""
    img = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = (config.hough_gauss_kernel, config.hough_gauss_kernel)
    img = cv2.medianBlur(img, config.hough_median_kernel)
    img = cv2.GaussianBlur(src=img, ksize=kernel, sigmaX=0)
    tmp_circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=config.hough_threshold_lower,
        param2=config.hough_threshold_upper,
        minRadius=5,
        maxRadius=300,
    )
    if tmp_circles is None:
        raise RuntimeWarning("No circles found! - Tune parameters.")
    tmp_circles = np.squeeze(np.uint16(np.around(tmp_circles))).reshape(-1, 3)

    circles_df = pd.DataFrame(tmp_circles, columns=["x", "y", "radius"])
    circles_df["variance"] = circles_df.apply(lambda x: pixel_variance(img, x), axis=1)
    circles_df["variance"] = circles_df["variance"] / circles_df["variance"].max() * 255
    circles_df = circles_df.sort_values(by="variance").head(3)
    circles_df["entropy"] = circles_df.apply(
        lambda x: normalized_pixel_entropy(img, x), axis=1
    )

    # Normalize variance and apply colormap
    circles_df["color"] = list(
        cv2.applyColorMap(
            circles_df["variance"].to_numpy().astype(np.uint8), cv2.COLORMAP_SUMMER
        )
    )
    return circles_df


def deep_circles(img: np.ndarray, device=DEVICE, model=MODEL) -> pd.DataFrame:
    """Detect ball using deep learning model DETR trained on COCO data."""
    with torch.no_grad():
        # Preprocess image
        image_tensor = IMG_PROC(images=img, return_tensors="pt").to(device)

        # Inference
        model.to(device)
        pred = model(**image_tensor)

        # Postprocess
        target_size = torch.tensor([img.shape[:2]])
        results = IMG_PROC.post_process_object_detection(
            pred, threshold=0.5, target_sizes=target_size
        )[0]

    # Convert tensor results to dataframe
    tmp_circles = {key: value.tolist() for key, value in results.items()}
    df_tmp = pd.DataFrame(tmp_circles)
    # ball_class = 37
    # circle_df = df_tmp[df_tmp["labels"] == ball_class]
    circle_df = df_tmp
    if circle_df.empty:
        raise RuntimeWarning("No circles found! - Tune parameters.")

    # Compute center and radius
    boxes = np.array(circle_df["boxes"].to_list())
    circle_df["x"] = np.mean(boxes[:, [0, 2]], axis=1).round().astype(int)
    circle_df["y"] = np.mean(boxes[:, [1, 3]], axis=1).round().astype(int)
    circle_df["radius_x"] = (boxes[:, 2] - boxes[:, 0]) / 2
    circle_df["radius_y"] = (boxes[:, 3] - boxes[:, 1]) / 2
    circle_df["radius"] = (
        np.mean(circle_df[["radius_x", "radius_y"]], axis=1).round().astype(int)
    )
    circle_df["radius_score"] = circle_df[["radius_x", "radius_y"]].max(
        axis=1
    ) / circle_df[["radius_x", "radius_y"]].min(axis=1)

    # Sort by variance
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circle_df["variance"] = circle_df.apply(
        lambda x: pixel_variance(img_gray, x), axis=1
    )
    circle_df["variance"] = circle_df["variance"] / circle_df["variance"].max() * 255
    circle_df = circle_df.sort_values(by="variance").head(3)
    circle_df["entropy"] = circle_df.apply(
        lambda x: normalized_pixel_entropy(img, x), axis=1
    )

    # Normalize variance and apply colormap
    circle_df["color"] = list(
        cv2.applyColorMap(
            circle_df["variance"].to_numpy().astype(np.uint8), cv2.COLORMAP_SUMMER
        )
    )

    return circle_df


def sphere(pcd: np.ndarray, config: Calibration, gt_center=None) -> pd.Series:
    """Fit sphere into point cloud using RANSAC"""
    result = {
        "idx": np.nan,
        "quality": np.nan,
        "radius": np.nan,
        "center": np.nan,
        "n_inliers": np.nan,
        "n_outliers": pcd.size,
        "max_inliers": np.nan,
        "inlier_ratio": np.nan,
    }

    full_pcd = pcd.copy()
    # Restrict search area: Ball has to be within min and max distance
    if gt_center is not None:
        pcd = filters.pcd_by_roi(pcd, radius=config.radius_m * 2, center=gt_center).pcd
    if pcd.shape[0] < 4:
        log.warning(f"Filtered pcd too small: {pcd.shape[0]} points available")
        return pd.Series(result)

    # Build k-D tree for efficient neighbor search
    kdtree = KDTree(pcd)

    results = []
    for idx, idx_point in enumerate(
        np.random.choice(pcd.shape[0], size=config.max_iterations)
    ):
        # Find close points using k-D tree
        close_points_idx = kdtree.query_radius(
            pcd[idx_point].reshape(1, -1),
            r=config.radius_m * 2 + config.inlier_threshold_m,
        )
        sub_pcd = pcd[close_points_idx[0]]

        if sub_pcd.shape[0] < 4:
            continue  # Skip if less than 4 points are available

        # Randomly select 4 points for sphere fitting
        sample = sub_pcd[np.random.choice(sub_pcd.shape[0], size=4, replace=False)]
        try:
            x = result.copy()
            x.update(fit_sphere(sample, config))
            x = pd.Series(x)
        except (np.linalg.LinAlgError, RuntimeWarning):
            continue

        # Compute number of inliers
        distance_center = np.linalg.norm(sub_pcd - x.center, axis=1)
        distance_sphere = np.abs(distance_center - x.radius)
        x.n_inliers = (distance_sphere < config.inlier_threshold_m).sum()

        # Number of max and n inliers
        x.max_inliers = filters.pcd_by_cone(
            full_pcd, x.radius + config.inlier_threshold_m, x.center
        ).n
        if x.max_inliers < x.n_inliers:
            log.error("Less max inliers found than expected - something is wrong!")

        x.n_outliers = sub_pcd.shape[0] - x.n_inliers
        x.inlier_ratio = np.divide(x.n_inliers, x.max_inliers)
        x.idx = idx
        x["rmse"] = np.mean(distance_sphere)
        results.append(x)

    if len(results) == 0:
        raise RuntimeWarning("No spheres found! - Tune parameters.")

    # Combine results
    results = pd.DataFrame(results)

    # Cost function: radius difference and number of inliers
    radius_min = np.clip(results["radius"], 1e-3, config.radius_m)
    radius_max = np.clip(results["radius"], config.radius_m, 1e3)
    results["quality"] = 0.5 * (
        (radius_min / radius_max) + (results["n_inliers"] / results["max_inliers"])
    )
    results = results.sort_values(by="quality", ascending=False).head(5)
    results_str = results[
        [
            "idx",
            "quality",
            "max_inliers",
            "n_inliers",
            "n_outliers",
            "rmse",
            "center",
            "radius",
        ]
    ].to_string(index=False)
    log.debug(f"Sphere candidates:\n{results_str}")
    return results.iloc[0]


def fit_sphere(points: np.ndarray, settings: Calibration) -> dict:
    """Fit a sphere into 4 points"""
    eq_a = np.hstack([np.ones((points.shape[0], 1)), -2 * points])
    eq_b = np.sum(-(points**2), axis=1, keepdims=True)
    center = np.squeeze(np.linalg.solve(eq_a, eq_b))
    radius = np.sqrt(np.sum(center[1:] ** 2) - center[0])

    # Discard fits that are not close to the expected radius
    if np.abs(radius - settings.radius_m) > settings.inlier_threshold_m:
        raise RuntimeWarning("Sphere radius is not close enough!")

    return {"center": np.round(center[1:], 3), "radius": np.round(radius, 3)}
