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
    df_eval = pd.read_csv(args.file)
    df_eval["datetime"] = pd.to_datetime(df_eval["datetime"], utc=True)
    df_eval["date"] = df_eval["datetime"].dt.date
    print(df_eval.info())
    aux.summary_sensor_date(df_eval)

    # Extract metrics columns: distance, intensity, metric and all that start with glcm_
    metric_cols = [c for c in df_eval.columns if c.startswith("glcm_")]
    df_eval = df_eval[["distance", "intensity", "metric", "weather"] + metric_cols]

    # Group by weather and drop redundant columns
    dfs = []
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
        "--file", type=Path, default=data.root() / "analysis/eval_2d.csv"
    )
    main(argparser.parse_args())
