import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.config import DATASET_PATH
from src.model_service import ModelService

logger = logging.getLogger(__name__)

class AnalyticsService:
    _instance = None

    def __init__(self):
        self.dataset_stats = {}
        self._load_and_compute_analytics()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_and_compute_analytics(self):
        """Loads employee_data.csv and generates comprehensive HR metrics."""
        if not DATASET_PATH.exists():
            logger.warning(f"Dataset not found at {DATASET_PATH}")
            self.dataset_stats = {}
            return

        try:
            df = pd.read_csv(DATASET_PATH)
            total_employees = len(df)
            attrition_yes_count = int((df["Attrition"] == "Yes").sum())
            attrition_no_count = int((df["Attrition"] == "No").sum())
            overall_attrition_rate = round((attrition_yes_count / total_employees) * 100, 2)

            # Department breakdown
            dept_counts = df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
            dept_labels = list(dept_counts.index)
            dept_yes = [int(dept_counts.loc[d].get("Yes", 0)) for d in dept_labels]
            dept_no = [int(dept_counts.loc[d].get("No", 0)) for d in dept_labels]
            dept_rates = [
                round((dept_counts.loc[d].get("Yes", 0) / (dept_counts.loc[d].get("Yes", 0) + dept_counts.loc[d].get("No", 0))) * 100, 1)
                for d in dept_labels
            ]

            # Job Role breakdown
            role_counts = df.groupby(["JobRole", "Attrition"]).size().unstack(fill_value=0)
            role_labels = list(role_counts.index)
            role_yes = [int(role_counts.loc[r].get("Yes", 0)) for r in role_labels]
            role_no = [int(role_counts.loc[r].get("No", 0)) for r in role_labels]
            role_rates = [
                round((role_counts.loc[r].get("Yes", 0) / (role_counts.loc[r].get("Yes", 0) + role_counts.loc[r].get("No", 0))) * 100, 1)
                for r in role_labels
            ]

            # Overtime vs Attrition
            ot_counts = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
            ot_labels = ["Yes (Overtime)", "No (Standard)"]
            ot_yes = [int(ot_counts.loc["Yes"].get("Yes", 0)), int(ot_counts.loc["No"].get("Yes", 0))]
            ot_no = [int(ot_counts.loc["Yes"].get("No", 0)), int(ot_counts.loc["No"].get("No", 0))]

            # Job Satisfaction breakdown (1 to 4)
            js_counts = df.groupby(["JobSatisfaction", "Attrition"]).size().unstack(fill_value=0)
            js_labels = ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
            js_yes = [int(js_counts.loc[i].get("Yes", 0)) if i in js_counts.index else 0 for i in [1, 2, 3, 4]]
            js_no = [int(js_counts.loc[i].get("No", 0)) if i in js_counts.index else 0 for i in [1, 2, 3, 4]]

            # Work Life Balance breakdown (1 to 4)
            wlb_counts = df.groupby(["WorkLifeBalance", "Attrition"]).size().unstack(fill_value=0)
            wlb_labels = ["1 - Bad", "2 - Good", "3 - Better", "4 - Best"]
            wlb_yes = [int(wlb_counts.loc[i].get("Yes", 0)) if i in wlb_counts.index else 0 for i in [1, 2, 3, 4]]
            wlb_no = [int(wlb_counts.loc[i].get("No", 0)) if i in wlb_counts.index else 0 for i in [1, 2, 3, 4]]

            # Tenure & Monthly Income Stats
            avg_monthly_income = round(float(df["MonthlyIncome"].mean()), 2)
            avg_years_at_company = round(float(df["YearsAtCompany"].mean()), 1)
            avg_age = round(float(df["Age"].mean()), 1)

            # Run batch inference on entire dataset to get predicted risk distribution
            model_service = ModelService.get_instance()
            bulk_resp, _ = model_service.predict_batch_df(df, filename="dataset_baseline.csv")

            self.dataset_stats = {
                "summary": {
                    "total_employees": total_employees,
                    "actual_attrition_count": attrition_yes_count,
                    "actual_retention_count": attrition_no_count,
                    "actual_attrition_rate": overall_attrition_rate,
                    "predicted_high_risk": bulk_resp.high_risk_count,
                    "predicted_medium_risk": bulk_resp.medium_risk_count,
                    "predicted_low_risk": bulk_resp.low_risk_count,
                    "avg_monthly_income": avg_monthly_income,
                    "avg_years_at_company": avg_years_at_company,
                    "avg_age": avg_age,
                },
                "charts": {
                    "attrition_distribution": {
                        "labels": ["Stayed (No)", "Left (Yes)"],
                        "data": [attrition_no_count, attrition_yes_count],
                    },
                    "risk_distribution": {
                        "labels": ["Low Risk (<30%)", "Medium Risk (30-60%)", "High Risk (≥60%)"],
                        "data": [bulk_resp.low_risk_count, bulk_resp.medium_risk_count, bulk_resp.high_risk_count],
                    },
                    "department_attrition": {
                        "labels": dept_labels,
                        "attrition_yes": dept_yes,
                        "attrition_no": dept_no,
                        "rates": dept_rates,
                    },
                    "job_role_attrition": {
                        "labels": role_labels,
                        "attrition_yes": role_yes,
                        "attrition_no": role_no,
                        "rates": role_rates,
                    },
                    "overtime_impact": {
                        "labels": ot_labels,
                        "attrition_yes": ot_yes,
                        "attrition_no": ot_no,
                    },
                    "job_satisfaction": {
                        "labels": js_labels,
                        "attrition_yes": js_yes,
                        "attrition_no": js_no,
                    },
                    "work_life_balance": {
                        "labels": wlb_labels,
                        "attrition_yes": wlb_yes,
                        "attrition_no": wlb_no,
                    }
                },
                "sample_employees": df.head(100).to_dict(orient="records")
            }
            logger.info("Dataset analytics computed successfully.")
        except Exception as e:
            logger.error(f"Error computing dataset analytics: {e}")
            self.dataset_stats = {}

    def get_analytics(self) -> Dict[str, Any]:
        return self.dataset_stats
