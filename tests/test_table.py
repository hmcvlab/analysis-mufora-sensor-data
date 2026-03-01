"""
Created on 2026-03-01
Copyright (c) 2026 Munich University of Applied Sciences
"""

import numpy as np
import pandas as pd
import pytest

from mufora import table


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {
                "datetime": ["2024-03-01 00:00:00", "2024-03-01 00:00:00"],
                "sensor": ["qb2_0", "qb2_1"],
                "metric": [0.1, 0.2],
            },
            {
                "datetime": ["2024-03-01 00:00:00", "2024-03-01 00:00:00"],
                "sensor": ["qb2_0", "qb2_1"],
                "metric": [0.1, 0.2],
            },
        ),
        (
            {
                "datetime": ["2024-03-01 00:00:00"],
                "sensor": ["qb2_0"],
                "metric": [0.5],
            },
            {
                "datetime": ["2024-03-01 00:00:00", "2024-03-01 00:00:00"],
                "sensor": ["qb2_1", "qb2_0"],
                "metric": [0.2, 0.5],
            },
        ),
    ],
)
def test_save(data, expected, tmp_path):
    """Test if data is merged correctly"""
    # Arrange
    data_old = {
        "datetime": ["2024-03-01 00:00:00", "2024-03-01 00:00:00"],
        "sensor": ["qb2_0", "qb2_1"],
        "metric": [0.1, 0.2],
    }
    df_old = pd.DataFrame(data_old)
    df = pd.DataFrame(data)
    filename = tmp_path / "test.csv"

    # Act
    table.save(df_old, filename)
    table.save(df, filename)

    # Assert
    df = pd.read_csv(filename)
    df_expected = pd.DataFrame(expected)
    np.testing.assert_array_equal(df["metric"], df_expected["metric"])
    np.testing.assert_array_equal(df["sensor"], df_expected["sensor"])
    np.testing.assert_array_equal(df["datetime"], df_expected["datetime"])
