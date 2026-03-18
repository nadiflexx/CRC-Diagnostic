"""
Resilient GDC API client for colon cancer clinical data.
"""

import time

import pandas as pd
import requests  # type: ignore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.logger import log as logger
from config.paths import paths
from config.settings import gdc as gdc_cfg


class GDCApiClient:
    def __init__(self):
        self.base_url = gdc_cfg.GDC_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        before_sleep=lambda rs: logger.warning(
            f"Retrying GDC API... attempt {rs.attempt_number}"
        ),
    )
    def _request(self, endpoint, params=None, json_body=None):
        """
        Make a request to the GDC API.

        :param endpoint: The API endpoint.
        :param params: The query parameters.
        :param json_body: The JSON body for POST requests.
        :return: The JSON response from the API.
        """
        url = f"{self.base_url}/{endpoint}"
        if json_body:
            response = self.session.post(
                url, json=json_body, timeout=gdc_cfg.GDC_TIMEOUT
            )
        else:
            response = self.session.get(url, params=params, timeout=gdc_cfg.GDC_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def fetch_colon_cancer_cases(self, max_cases=1000):
        """
        Fetch colon cancer cases from the GDC API.

        :param max_cases: The maximum number of cases to fetch.
        :return: A list of flattened case dictionaries.
        """
        fields = [
            "case_id",
            "submitter_id",
            "demographic.gender",
            "demographic.race",
            "demographic.ethnicity",
            "demographic.age_at_index",
            "demographic.year_of_birth",
            "demographic.vital_status",
            "demographic.days_to_death",
            "diagnoses.age_at_diagnosis",
            "diagnoses.primary_diagnosis",
            "diagnoses.tumor_stage",
            "diagnoses.tumor_grade",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_pathologic_t",
            "diagnoses.ajcc_pathologic_n",
            "diagnoses.ajcc_pathologic_m",
            "diagnoses.morphology",
            "diagnoses.tissue_or_organ_of_origin",
            "diagnoses.site_of_resection_or_biopsy",
            "diagnoses.days_to_last_follow_up",
            "exposures.alcohol_history",
            "exposures.alcohol_intensity",
            "exposures.tobacco_smoking_status",
            "exposures.bmi",
            "exposures.pack_years_smoked",
        ]
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-COAD", "TCGA-READ"],
                    },
                }
            ],
        }

        all_cases = []
        offset = 0
        page_size = 100
        while offset < max_cases:
            current_size = min(page_size, max_cases - offset)
            body = {
                "filters": filters,
                "fields": ",".join(fields),
                "format": "JSON",
                "size": current_size,
                "from": offset,
            }
            logger.info(f"Fetching GDC cases: offset={offset}, size={current_size}")
            result = self._request("cases", json_body=body)
            hits = result.get("data", {}).get("hits", [])
            if not hits:
                break
            all_cases.extend(hits)
            offset += len(hits)
            total = result.get("data", {}).get("pagination", {}).get("total", 0)
            if offset >= total:
                break
            time.sleep(gdc_cfg.GDC_RATE_LIMIT_DELAY)

        logger.info(f"Total GDC cases: {len(all_cases)}")
        return self._flatten_cases(all_cases)

    def _flatten_cases(self, cases):
        """
        Flatten the case data.

        :param cases: The list of case dictionaries.
        :return: A DataFrame with flattened case data.
        """
        rows = []
        for case in cases:
            row = {
                "case_id": case.get("case_id"),
                "submitter_id": case.get("submitter_id"),
                "has_cancer": True,
            }
            demo = case.get("demographic", {})
            if isinstance(demo, list):
                demo = demo[0] if demo else {}
            row["gender"] = demo.get("gender")
            row["race"] = demo.get("race")
            row["ethnicity"] = demo.get("ethnicity")
            row["age_at_index"] = demo.get("age_at_index")
            row["vital_status"] = demo.get("vital_status")
            row["days_to_death"] = demo.get("days_to_death")
            diags = case.get("diagnoses", [])
            if diags:
                diag = diags[0]
                row["primary_diagnosis"] = diag.get("primary_diagnosis")
                row["tumor_stage"] = diag.get("tumor_stage")
                row["tumor_grade"] = diag.get("tumor_grade")
                row["ajcc_stage"] = diag.get("ajcc_pathologic_stage")
                row["ajcc_t"] = diag.get("ajcc_pathologic_t")
                row["ajcc_n"] = diag.get("ajcc_pathologic_n")
                row["ajcc_m"] = diag.get("ajcc_pathologic_m")
                row["morphology"] = diag.get("morphology")
                row["age_at_diagnosis"] = diag.get("age_at_diagnosis")
            exposures = case.get("exposures", [])
            if exposures:
                exp = exposures[0]
                row["alcohol_history"] = exp.get("alcohol_history")
                row["alcohol_intensity"] = exp.get("alcohol_intensity")
                row["tobacco_smoking_status"] = exp.get("tobacco_smoking_status")
                row["bmi"] = exp.get("bmi")
                row["pack_years_smoked"] = exp.get("pack_years_smoked")
            rows.append(row)
        return pd.DataFrame(rows)

    def save_to_csv(self, df, filename="gdc_colon_cases.csv"):
        """
        Save the DataFrame to a CSV file.
        :param df: The DataFrame to save.
        :param filename: The name of the CSV file.
        :return: The path to the saved CSV file.
        """
        path = paths.RAW_TABULAR / filename
        df.to_csv(path, index=False)
        logger.info(f"GDC data saved: {path}")
        return path


if __name__ == "__main__":
    client = GDCApiClient()
    df = client.fetch_colon_cancer_cases(max_cases=500)
    print(f"Cases: {len(df)}")
    client.save_to_csv(df)
