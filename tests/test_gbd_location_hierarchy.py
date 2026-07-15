# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for selecting national locations from the public GBD hierarchy."""

import pandas as pd

from workflow.scripts.prepare_gbd_food_group_intake import (
    build_national_location_map,
)


def test_builds_level_three_location_map(tmp_path):
    hierarchy = pd.DataFrame(
        {
            "Location ID": [102, 8, 533, 999],
            "Location Name": [
                "United States of America",
                "Taiwan (Province of China)",
                "Georgia",
                "Georgia",
            ],
            "Level": [3, 3, 3, 4],
        }
    )
    path = tmp_path / "hierarchy.xlsx"
    with pd.ExcelWriter(path) as writer:
        hierarchy.to_excel(
            writer, sheet_name="GBD 2021 Locations Hierarchy", index=False
        )
    assert build_national_location_map(path) == {102: "USA", 8: "TWN", 533: "GEO"}
