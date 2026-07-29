# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Freeze water-supply CFs at a realized reference solution.

For each renewable ``(region, period, source)`` supply group, the fixed CF is
the CF of the highest tier used by the reference. If the reference draws
nothing from a group, its lowest available CF is used. The resulting link-level
table lets a solve apply a conventional fixed characterization factor while
retaining exactly the same physical supply tiers and feasible set.
"""

import pandas as pd
import pypsa


def extract_fixed_water_characterization(n: pypsa.Network) -> pd.DataFrame:
    """Return fixed CFs and realized draws for every water-supply link.

    Fossil groundwater has no renewable scarcity CF, so its ``fixed_cf`` is
    missing. Keeping its dispatch in the same table makes it possible to
    cross-score every solution with the separately chosen fossil-water weight.
    """
    links = n.links.static
    tiers = links[links["carrier"] == "water_supply"].copy()
    if tiers.empty:
        raise ValueError("Reference solution contains no water-supply links")

    tiers["draw_mm3"] = (
        n.links.dynamic.p0.iloc[0].reindex(tiers.index).fillna(0.0).clip(lower=0.0)
    )
    tiers["cf"] = tiers["efficiency2"].astype(float)
    renewable = tiers[tiers["source"] != "groundwater_nonrenewable"]
    if renewable.empty:
        raise ValueError("Reference solution contains no renewable water tiers")
    group_columns = ["region", "period", "source"]

    rows = []
    for _, group in renewable.groupby(group_columns, sort=False, dropna=False):
        used = group[group["draw_mm3"] > 1e-8]
        fixed_cf = float(used["cf"].max() if not used.empty else group["cf"].min())
        rows.append(
            pd.DataFrame(
                {
                    "link": group.index,
                    "region": group["region"].to_numpy(),
                    "period": group["period"].to_numpy(),
                    "source": group["source"].to_numpy(),
                    "fixed_cf": fixed_cf,
                    "endogenous_cf": group["cf"].to_numpy(),
                    "draw_mm3": group["draw_mm3"].to_numpy(),
                    "reference_group_draw_mm3": float(group["draw_mm3"].sum()),
                }
            )
        )
    fossil = tiers[tiers["source"] == "groundwater_nonrenewable"]
    if not fossil.empty:
        rows.append(
            pd.DataFrame(
                {
                    "link": fossil.index,
                    "region": fossil["region"].to_numpy(),
                    "period": fossil["period"].to_numpy(),
                    "source": fossil["source"].to_numpy(),
                    "fixed_cf": float("nan"),
                    "endogenous_cf": fossil["cf"].to_numpy(),
                    "draw_mm3": fossil["draw_mm3"].to_numpy(),
                    "reference_group_draw_mm3": float("nan"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
