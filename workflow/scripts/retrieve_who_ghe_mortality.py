# SPDX-FileCopyrightText: 2026 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Retrieve age-specific both-sex mortality rates from the WHO GHE API."""

import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from workflow.scripts.logging_config import setup_script_logging

logger = logging.getLogger(__name__)

API_URL = "https://xmart-api-public.who.int/DEX_CMS/GHE_FULL"
SELECT_COLUMNS = [
    "DIM_COUNTRY_CODE",
    "DIM_YEAR_CODE",
    "DIM_AGEGROUP_CODE",
    "DIM_SEX_CODE",
    "DIM_GHECAUSE_CODE",
    "DIM_GHECAUSE_TITLE",
    "ATTR_POPULATION_NUMERIC",
    "VAL_DTHS_RATE100K_NUMERIC",
    "VAL_DTHS_COUNT_NUMERIC",
    "VAL_DTHS_COUNT_LOW",
    "VAL_DTHS_COUNT_HIGH",
]


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def retrieve(year: int, cause_ids: list[int], session=None) -> pd.DataFrame:
    """Return all API rows for one year, both sexes, and selected causes."""
    cause_filter = " or ".join(
        f"DIM_GHECAUSE_CODE eq {cause_id}" for cause_id in cause_ids
    )
    params = {
        "$filter": (
            f"DIM_YEAR_CODE eq {year} and DIM_SEX_CODE eq 'TOTAL' and ({cause_filter})"
        ),
        "$select": ",".join(SELECT_COLUMNS),
        "$top": 20000,
    }
    client = session or _session()
    url = API_URL
    rows = []
    while url:
        response = client.get(url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("value")
        if not isinstance(page, list):
            raise ValueError("WHO GHE API response has no list-valued 'value' field")
        rows.extend(page)
        url = payload.get("@odata.nextLink")
        params = None
    if not rows:
        raise ValueError(f"WHO GHE API returned no mortality rows for {year}")
    return pd.DataFrame(rows)


def main() -> None:
    year = int(snakemake.params.year)
    cause_ids = list(snakemake.params.cause_ids)
    logger.info("Retrieving WHO GHE mortality for %d and causes %s", year, cause_ids)
    mortality = retrieve(year, cause_ids)
    output = Path(snakemake.output.mortality)
    output.parent.mkdir(parents=True, exist_ok=True)
    mortality.to_csv(output, index=False)
    logger.info("Wrote %d rows to %s", len(mortality), output)


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)
    main()
