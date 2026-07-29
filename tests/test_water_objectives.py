# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import pandas as pd
import pytest

from workflow.scripts.analysis.extract_fixed_water_characterization import (
    extract_fixed_water_characterization,
)
from workflow.scripts.solve_model.affordability import production_cost_links
from workflow.scripts.solve_model.core import (
    add_water_metric_pricing_to_objective,
    add_water_scarcity_joint_cap,
)


def _water_network() -> SimpleNamespace:
    links = pd.DataFrame(
        {
            "carrier": ["water_supply"] * 5,
            "source": [
                "renewable",
                "renewable",
                "renewable",
                "groundwater_renewable",
                "groundwater_nonrenewable",
            ],
            "region": ["r1", "r1", "r2", "r1", "r1"],
            "period": [0, 0, 0, -1, -1],
            "efficiency2": [1.0, 5.0, 2.0, 8.0, 1.0],
            "marginal_cost": [1e-8, 5e-8, 2e-8, 8e-8, 0.2],
        },
        index=["r1_low", "r1_high", "r2", "r1_gw", "r1_fossil"],
    )
    dynamic = SimpleNamespace(
        p0=pd.DataFrame(
            [[3.0, 1.0, 0.0, 2.0, 4.0]],
            columns=links.index,
        )
    )
    return SimpleNamespace(
        links=SimpleNamespace(static=links, dynamic=dynamic),
        meta={},
    )


def test_fixed_characterization_uses_reference_marginal_tier() -> None:
    result = extract_fixed_water_characterization(_water_network())
    fixed = result.set_index("link")["fixed_cf"]

    assert fixed["r1_low"] == pytest.approx(5.0)
    assert fixed["r1_high"] == pytest.approx(5.0)
    assert fixed["r2"] == pytest.approx(2.0)
    assert fixed["r1_gw"] == pytest.approx(8.0)
    assert pd.isna(fixed["r1_fossil"])

    draws = result.set_index("link")["draw_mm3"]
    assert draws.to_dict() == pytest.approx(
        {"r1_low": 3.0, "r1_high": 1.0, "r2": 0.0, "r1_gw": 2.0, "r1_fossil": 4.0}
    )


def test_volume_pricing_weights_every_source_equally() -> None:
    n = _water_network()

    add_water_metric_pricing_to_objective(n, "volume", 1.0, 100.0, None)

    added = n.links.static["marginal_cost"] - pd.Series(
        [1e-8, 5e-8, 2e-8, 8e-8, 0.2],
        index=n.links.static.index,
    )
    assert added.to_numpy() == pytest.approx([0.001] * 5)
    assert n.links.static["water_metric_cost_adder"].to_numpy() == pytest.approx(
        [0.001] * 5
    )
    assert production_cost_links(n)["r1_fossil"] == pytest.approx(0.2)
    assert n.meta["water_objective_metric"] == "volume"


def test_fixed_pricing_uses_frozen_cfs_and_fossil_weight(
    tmp_path,
) -> None:
    n = _water_network()
    reference = extract_fixed_water_characterization(n)
    path = tmp_path / "fixed.parquet"
    reference.to_parquet(path, index=False)

    add_water_metric_pricing_to_objective(
        n, "fixed_characterization", 1.0, 100.0, str(path)
    )

    added = n.links.static["marginal_cost"] - pd.Series(
        [1e-8, 5e-8, 2e-8, 8e-8, 0.2],
        index=n.links.static.index,
    )
    assert added.to_numpy() == pytest.approx([0.005, 0.005, 0.002, 0.008, 0.1])


def test_joint_cap_selects_store_variables_by_name() -> None:
    class FakeVariable:
        def __init__(self) -> None:
            self.selections = []

        def sel(self, **kwargs):
            self.selections.append(kwargs)
            return self

        def __add__(self, other):
            return self

        def __rmul__(self, other):
            return self

        def __le__(self, other):
            return self

    class FakeModel:
        def __init__(self) -> None:
            self.variable = FakeVariable()
            self.variables = {"Store-e": self.variable}

        def add_constraints(self, *args, **kwargs) -> None:
            pass

    class FakeGlobalConstraints:
        def add(self, *args, **kwargs) -> None:
            pass

    model = FakeModel()
    n = SimpleNamespace(
        model=model,
        snapshots=pd.Index(["now"]),
        global_constraints=FakeGlobalConstraints(),
    )

    add_water_scarcity_joint_cap(n, 10.0, 100.0)

    assert {"name": "store:impact:water_scarcity"} in model.variable.selections
    assert {"name": "store:impact:groundwater_depletion"} in model.variable.selections
