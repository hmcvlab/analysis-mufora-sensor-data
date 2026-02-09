"""
Created on Tue Sep 24 2024
Copyright (c) 2024 Munich University of Applied Sciences
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as log

from aux import pose

DEFAULT_INTRINSICS = [
    481.0692443847656,
    0.0,
    324.6004333496094,
    0.0,
    481.0692443847656,
    176.78298950195312,
    0.0,
    0.0,
    1.0,
]


# pylint: disable=too-many-instance-attributes
@dataclass
class Calibration:
    """Settings for calibration"""

    distortion: list = None
    ext_cam_1_cam_0: np.ndarray = None
    ext_lidar_0_to_cam_0: np.ndarray = None
    ext_lidar_1_to_cam_0: np.ndarray = None
    hough_gauss_kernel: int = 17
    hough_median_kernel: int = 11
    hough_threshold_lower: int = 67
    hough_threshold_upper: int = 24
    inlier_threshold_m: float = 0.03
    intrinsic: np.ndarray = None
    max_distance_m: float = 40
    max_iterations: int = 1000
    max_slope_s: float = 0.1
    min_distance_m: float = 4.5
    min_inliers: int = 166
    radius_m: float = 0.32
    whitelist: list = None

    def __post_init__(self):
        if isinstance(self.intrinsic, list):
            self.intrinsic = np.array(self.intrinsic).reshape(3, 3)
        if isinstance(self.ext_lidar_0_to_cam_0, dict):
            self.ext_lidar_0_to_cam_0 = pose.from_vectors(**self.ext_lidar_0_to_cam_0)
        if isinstance(self.ext_lidar_1_to_cam_0, dict):
            self.ext_lidar_1_to_cam_0 = pose.from_vectors(**self.ext_lidar_1_to_cam_0)
        if isinstance(self.ext_cam_1_cam_0, dict):
            self.ext_cam_1_cam_0 = pose.from_vectors(**self.ext_cam_1_cam_0)
        if self.whitelist is None:
            self.whitelist = [""]


def from_files(file_calib: Path = None, file_metadata: Path = None) -> Calibration:
    """Load settings from yaml file."""
    t0 = datetime.now()
    if file_calib is None:
        log.warning(f"Calibration file not provided: {file_calib}")
        data_calib = {}
    else:
        log.debug(f"Load calibration file: {file_calib}")
        with open(file_calib, encoding="utf-8") as f:
            data_calib = yaml.safe_load(f)

    t1 = datetime.now()

    if file_metadata is None:
        log.info(f"Metadata file not provided: {file_metadata}")
        data_meta = {
            "intrinsic": DEFAULT_INTRINSICS,
            "distortion": [0] * 8,
        }
    else:
        log.debug(f"Load metadata file: {file_metadata}")
        with open(file_metadata, encoding="utf-8") as f:
            tmp_data = yaml.safe_load(f)

        # Assuming left and right camera have the same intrinsic
        data_meta = {
            "intrinsic": tmp_data["cam_left"]["intrinsic"],
            "distortion": tmp_data["cam_left"]["distortion"],
        }
    t2 = datetime.now()

    # Merge
    data = {}
    data.update(data_calib)
    data.update(data_meta)

    # Remove all keys that are not in Calibration
    keys_inter = set(data.keys()).intersection(Calibration.__dataclass_fields__)
    data = {k: data[k] for k in keys_inter}

    log.debug(
        "Load settings from files took:"
        + f" {(t1 - t0).total_seconds():.2f}"
        + f" + {(t2 - t1).total_seconds():.2f}"
        + f" = {(t2 - t0).total_seconds():.2f}"
    )

    return Calibration(**data)
