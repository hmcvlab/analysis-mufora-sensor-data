"""
Created on 2026-03-01
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from pathlib import Path

import pandas as pd

from mufora import aux, data


def main(args):
    """Entrypoint."""
    df_eval_2d = pd.read_csv(args.file_eval_2d)
    df_eval_3d = pd.read_csv(args.file_eval_3d)

    # Extract metrics columns: distance, intensity, metric and all that start with glcm_
    metric_cols = [c for c in df_eval_2d.columns if c.startswith("glcm_")]
    cols_2d = ["distance", "intensity", "weather", "metric"] + metric_cols
    cols_3d = ["distance", "intensity", "weather", "metric", "inlier_ratio"]

    # Summarize correlation
    _summarize_correlations(df_eval_2d, cols_2d)
    _summarize_correlations(df_eval_3d, cols_3d)


def _summarize_correlations(df_eval: pd.DataFrame, columns: list[str]):
    """Summarize correlation between columns"""
    df_eval["datetime"] = pd.to_datetime(df_eval["datetime"], utc=True)
    df_eval["date"] = df_eval["datetime"].dt.date
    print(df_eval.info())
    aux.summary_sensor_date(df_eval)

    # Group by weather and drop redundant columns
    dfs = []
    df_eval = df_eval[columns]
    for weather in ["rain", "fog"]:
        sub_df = df_eval[df_eval["weather"] == weather]
        df_res = sub_df.drop("weather", axis=1).corr().head(2).round(2).reset_index()
        df_res["weather"] = weather
        df_res.drop(["distance", "intensity"], axis=1, inplace=True)
        dfs.append(df_res)

    df_res = pd.concat(dfs, axis=0)

    # Set index and weather and multi-index
    df_res = df_res.sort_values(["index", "weather"])
    df_res.set_index(["index", "weather"], inplace=True)

    print(df_res)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file-eval-2d", type=Path, default=data.root() / "analysis/eval_2d.csv"
    )
    argparser.add_argument(
        "--file-eval-3d", type=Path, default=data.root() / "analysis/eval_3d.csv"
    )
    main(argparser.parse_args())
