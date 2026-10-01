"""
Created on Thu Oct 01 2026
Copyright (c) 2026 Munich University of Applied Sciences

Sanity check for the MuFoRa reproduction setup. Verifies that the dataset
root is found, the expected folders/files exist, and the annotation files
bundled with this repository are in place. Run before the analysis scripts:

    python scripts/check_setup.py
"""

import argparse
import sys
from pathlib import Path

from loguru import logger as log

from mufora import data

ROOT = Path(__file__).parent.parent
FILE_ANNOTATIONS_2D = (
    ROOT
    / "data/annotations/2d/carissma-indoor-multi-cam.v2i.coco/_annotations.coco.json"
)
DIR_ANNOTATIONS_3D = ROOT / "data/annotations/3d"
DIR_FOG = ROOT / "data/fog"
DIR_CALIB = ROOT / "data/calib"

FAILED = False


def check(description: str, ok: bool, details: str = ""):
    """Log a check result and remember failures."""
    global FAILED  # pylint: disable=global-statement
    status = "OK" if ok else "FAILED"
    msg = f"[{status}] {description}"
    if details:
        msg += f" ({details})"
    if ok:
        log.info(msg)
    else:
        FAILED = True
        log.error(msg)


def main(args: argparse.Namespace):
    """Entry point."""
    # Bundled repository files
    check(
        "2D annotations (COCO)",
        FILE_ANNOTATIONS_2D.is_file(),
        str(FILE_ANNOTATIONS_2D),
    )
    labels_3d = (
        list(DIR_ANNOTATIONS_3D.glob("*_qb2_*.json"))
        if DIR_ANNOTATIONS_3D.is_dir()
        else []
    )
    check(
        "3D annotations",
        len(labels_3d) > 0,
        f"{len(labels_3d)} label files in {DIR_ANNOTATIONS_3D}",
    )
    fog_files = list(DIR_FOG.glob("*_fog.csv"))
    check("Fog visibility data", len(fog_files) > 0, f"{len(fog_files)} files")
    calib_files = list(DIR_CALIB.glob("*.json"))
    check("Camera-LiDAR extrinsics", len(calib_files) > 0, f"{len(calib_files)} files")

    # Dataset root
    try:
        root = data.root()
        check("Dataset root", True, str(root))
    except FileNotFoundError as e:
        check("Dataset root", False, str(e))
        sys.exit(1)

    # Dataset structure
    dir_rawdata = data.rawdata()
    check("rawdata/ directory", dir_rawdata.is_dir(), str(dir_rawdata))
    folders = data.relevant_raw_folders()
    check(
        "Relevant folders",
        len(folders) > 0,
        f"{len(folders)} folders matching {data.WHITELIST}",
    )

    n_calib, n_png, n_pcd, n_pcd_empty = 0, 0, 0, 0
    for folder in folders:
        n_calib += folder.joinpath("calib.yaml").is_file()
        n_png += len(list(folder.glob("*.png")))
        for pcd in folder.glob("*.pcd"):
            n_pcd += 1
            n_pcd_empty += pcd.stat().st_size < 100
    check(
        "Per-folder calib.yaml (optional, defaults are used if absent)",
        True,
        f"{n_calib}/{len(folders)} folders",
    )
    check("Camera images (*.png)", n_png > 0, f"{n_png} files")
    check("Point clouds (*.pcd)", n_pcd > 0, f"{n_pcd} files")
    check(
        "Unreadable point clouds (skipped by the evaluation)",
        True,
        f"{n_pcd_empty} empty files (known limitation of the published archive)",
    )

    # Output directory (created by the analysis scripts)
    dir_analysis = root / "analysis"
    try:
        dir_analysis.mkdir(exist_ok=True)
        check("analysis/ output directory", True, str(dir_analysis))
    except OSError as e:
        check("analysis/ output directory", False, str(e))

    if FAILED:
        log.error("Setup check failed - fix the issues above and re-run.")
        sys.exit(1)
    log.info("Setup check passed - ready to reproduce.")
    if args.debug:
        for folder in folders:
            log.debug(folder)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    main(argparser.parse_args())
