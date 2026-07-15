# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for WHO Global Health Estimates mortality retrieval and preparation."""

import pandas as pd
import pytest

from workflow.scripts.prepare_who_ghe_mortality import (
    AGE_MAP,
    OPEN_AGE_SOURCE,
    prepare_mortality,
)
from workflow.scripts.retrieve_who_ghe_mortality import API_URL, retrieve
from workflow.validation.health_map import validate_health_map


def _raw_country(country: str, cause_id: int = 1130) -> pd.DataFrame:
    rates = [250.0] * (len(AGE_MAP) + 1)
    rates[-1] = 900.0
    return pd.DataFrame(
        {
            "DIM_COUNTRY_CODE": country,
            "DIM_YEAR_CODE": 2020,
            "DIM_AGEGROUP_CODE": [*AGE_MAP, OPEN_AGE_SOURCE],
            "DIM_SEX_CODE": "TOTAL",
            "DIM_GHECAUSE_CODE": cause_id,
            "VAL_DTHS_RATE100K_NUMERIC": rates,
        }
    )


def test_prepares_rates_expands_open_age_and_applies_proxy():
    output = prepare_mortality(
        raw=_raw_country("WSM"),
        countries=["ASM"],
        causes=["CHD"],
        cause_ids={"CHD": 1130},
        year=2020,
    )
    assert len(output) == 21
    assert set(output["country"]) == {"ASM"}
    assert {"85-89", "90-94", "95+"} <= set(output["age"])
    assert OPEN_AGE_SOURCE not in set(output["age"])
    oldest = output[output["age"].isin(["85-89", "90-94", "95+"])]
    assert set(oldest["value"]) == {9.0}
    assert set(output.loc[~output.index.isin(oldest.index), "value"]) == {2.5}
    assert list(output.columns) == ["age", "cause", "country", "year", "value"]


def test_rejects_incomplete_age_coverage():
    raw = _raw_country("USA")
    raw = raw[raw["DIM_AGEGROUP_CODE"] != "Y40T44"]
    with pytest.raises(ValueError, match="missing configured country/cause/age"):
        prepare_mortality(raw, ["USA"], ["CHD"], {"CHD": 1130}, 2020)


@pytest.mark.parametrize("invalid_rate", ["not a number", float("inf"), -1.0])
def test_rejects_invalid_rate(invalid_rate):
    raw = _raw_country("USA")
    raw["VAL_DTHS_RATE100K_NUMERIC"] = raw["VAL_DTHS_RATE100K_NUMERIC"].astype(object)
    raw.loc[0, "VAL_DTHS_RATE100K_NUMERIC"] = invalid_rate
    with pytest.raises(ValueError, match="invalid mortality rates"):
        prepare_mortality(raw, ["USA"], ["CHD"], {"CHD": 1130}, 2020)


def _health_config() -> dict:
    return {
        "health": {
            "mortality_source": "who_ghe",
            "risk_factors": ["fruits"],
            "causes": ["CHD", "Stroke"],
            "risk_cause_map": {"fruits": ["CHD", "Stroke"]},
            "ghe_cause_id": {"CHD": 1130, "Stroke": 1141},
        },
        "food_groups": {"max_per_capita": {"fruits": 1000}},
    }


def test_health_validation_rejects_missing_ghe_cause(tmp_path):
    config = _health_config()
    del config["health"]["ghe_cause_id"]["Stroke"]
    with pytest.raises(ValueError, match=r"missing causes.*Stroke"):
        validate_health_map(config, tmp_path)


def test_health_validation_rejects_duplicate_ghe_cause_id(tmp_path):
    config = _health_config()
    config["health"]["ghe_cause_id"]["Stroke"] = 1130
    with pytest.raises(ValueError, match="values must be unique"):
        validate_health_map(config, tmp_path)


def test_health_validation_allows_ids_for_unselected_causes(tmp_path):
    config = _health_config()
    config["health"]["causes"] = ["CHD"]
    config["health"]["risk_cause_map"] = {"fruits": ["CHD"]}
    validate_health_map(config, tmp_path)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []
        self.responses = [
            _Response({"value": [{"row": 1}], "@odata.nextLink": "next-page"}),
            _Response({"value": [{"row": 2}]}),
        ]

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_retrieval_follows_odata_pagination():
    session = _Session()
    output = retrieve(2020, [1130, 650], session=session)
    assert output["row"].tolist() == [1, 2]
    assert session.calls[0][0] == API_URL
    assert "DIM_YEAR_CODE eq 2020" in session.calls[0][1]["$filter"]
    assert "DIM_GHECAUSE_CODE eq 1130" in session.calls[0][1]["$filter"]
    assert session.calls[1][0] == "next-page"
    assert session.calls[1][1] is None
