# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import pandas as pd
import pytest

from workflow.scripts.analysis.extract_barrier_constraints import (
    extract_barrier_constraints,
)


def test_joint_water_cap_dual_is_reported_as_positive_usd_per_m3() -> None:
    constraints = pd.DataFrame(
        {
            "type": ["water_scarcity_joint_cap", "conversion_cap"],
            "sense": ["<=", "<="],
            "constant": [10_000.0, 2.0],
            "mu": [-0.0015, -4.0],
        },
        index=["water_scarcity_joint_cap", "conversion_cap"],
    )
    n = SimpleNamespace(global_constraints=SimpleNamespace(static=constraints))

    result = extract_barrier_constraints(n)

    water = result.set_index("name").loc["water_scarcity_joint_cap"]
    other = result.set_index("name").loc["conversion_cap"]
    assert float(water["shadow_price_usd_per_m3"]) == pytest.approx(1.5)
    assert pd.isna(other["shadow_price_usd_per_m3"])
