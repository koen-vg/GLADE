# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert WHO GHE mortality rates to the canonical health-model table."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.scripts.logging_config import setup_script_logging

logger = logging.getLogger(__name__)

AGE_MAP = {
    "Y0T1": "<1",
    "Y1T4": "1-4",
    **{f"Y{start}T{start + 4}": f"{start}-{start + 4}" for start in range(5, 85, 5)},
}
OPEN_AGE_SOURCE = "YGE_85"
OPEN_AGE_TARGETS = ["85-89", "90-94", "95+"]

# WHO GHE has no separate rows for these territories. These proxies are used
# only to complete the configured country set.
COUNTRY_PROXIES = {
    "ASM": "WSM",
    "GUF": "FRA",
    "PRI": "USA",
    "PSE": "JOR",
    "TWN": "KOR",
}

REQUIRED_COLUMNS = {
    "DIM_COUNTRY_CODE",
    "DIM_YEAR_CODE",
    "DIM_AGEGROUP_CODE",
    "DIM_SEX_CODE",
    "DIM_GHECAUSE_CODE",
    "VAL_DTHS_RATE100K_NUMERIC",
}


def prepare_mortality(
    raw: pd.DataFrame,
    countries: list[str],
    causes: list[str],
    cause_ids: dict[str, int],
    year: int,
) -> pd.DataFrame:
    """Map WHO codes, proxy uncovered places, and return rates per 1,000."""
    missing_columns = REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(f"WHO GHE data is missing columns: {sorted(missing_columns)}")

    missing_cause_ids = set(causes) - set(cause_ids)
    if missing_cause_ids:
        raise ValueError(
            f"No WHO GHE cause identifier configured for {sorted(missing_cause_ids)}"
        )
    id_to_cause = {str(cause_ids[cause]): cause for cause in causes}
    if len(id_to_cause) != len(causes):
        raise ValueError("WHO GHE cause identifiers must be unique")

    data = raw.copy()
    data["DIM_YEAR_CODE"] = pd.to_numeric(data["DIM_YEAR_CODE"], errors="coerce")
    data["DIM_GHECAUSE_CODE"] = data["DIM_GHECAUSE_CODE"].astype(str)
    data = data[
        (data["DIM_YEAR_CODE"] == year)
        & (data["DIM_SEX_CODE"] == "TOTAL")
        & (data["DIM_GHECAUSE_CODE"].isin(id_to_cause))
        & (data["DIM_AGEGROUP_CODE"].isin([*AGE_MAP, OPEN_AGE_SOURCE]))
    ].copy()
    if data.empty:
        raise ValueError(f"No usable WHO GHE mortality rows for {year}")

    data["cause"] = data["DIM_GHECAUSE_CODE"].map(id_to_cause)
    data["value"] = pd.to_numeric(data["VAL_DTHS_RATE100K_NUMERIC"], errors="coerce")
    invalid = ~np.isfinite(data["value"]) | (data["value"] < 0)
    if invalid.any():
        raise ValueError(
            f"WHO GHE data contains {int(invalid.sum())} invalid mortality rates"
        )

    source_key = ["DIM_COUNTRY_CODE", "cause", "DIM_AGEGROUP_CODE"]
    duplicates = data.duplicated(source_key, keep=False)
    if duplicates.any():
        examples = (
            data.loc[duplicates, source_key].drop_duplicates().head().to_dict("records")
        )
        raise ValueError(f"WHO GHE data has duplicate mortality rows: {examples}")

    data["age"] = data["DIM_AGEGROUP_CODE"].map(AGE_MAP)
    closed = data[data["age"].notna()].copy()
    open_age = data[data["DIM_AGEGROUP_CODE"] == OPEN_AGE_SOURCE].copy()
    expanded = []
    for age in OPEN_AGE_TARGETS:
        part = open_age.copy()
        part["age"] = age
        expanded.append(part)
    data = pd.concat([closed, *expanded], ignore_index=True)
    data = data.rename(columns={"DIM_COUNTRY_CODE": "country"})
    data["year"] = year
    data["value"] = data["value"] / 100.0
    output = data[["age", "cause", "country", "year", "value"]]

    available_countries = set(output["country"])
    proxy_rows = []
    for country in sorted(set(countries) - available_countries):
        proxy = COUNTRY_PROXIES.get(country)
        if proxy is None or proxy not in available_countries:
            continue
        rows = output[output["country"] == proxy].copy()
        rows["country"] = country
        proxy_rows.append(rows)
        logger.info("Using %s mortality as proxy for %s", proxy, country)
    if proxy_rows:
        output = pd.concat([output, *proxy_rows], ignore_index=True)
    output = output[output["country"].isin(countries)].copy()

    expected_ages = set(AGE_MAP.values()) | set(OPEN_AGE_TARGETS)
    expected = pd.MultiIndex.from_product(
        [countries, causes, expected_ages], names=["country", "cause", "age"]
    )
    actual = pd.MultiIndex.from_frame(output[["country", "cause", "age"]])
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(
            "WHO GHE mortality is missing configured country/cause/age rows; "
            f"first entries: {list(missing[:10])}"
        )

    return output.sort_values(["country", "cause", "age"]).reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(snakemake.input.who_ghe_mortality)
    output = prepare_mortality(
        raw=raw,
        countries=list(snakemake.params.countries),
        causes=list(snakemake.params.causes),
        cause_ids=dict(snakemake.params.ghe_cause_id),
        year=int(snakemake.params.reference_year),
    )
    output_path = Path(snakemake.output.mortality)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, header=False)
    logger.info("Wrote %d rows to %s", len(output), output_path)


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)
    main()
