import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

from src.config import (
    MODEL_PATH,
    DATASET_PATH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM,
    FEATURE_LABELS,
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.schemas import (
    EmployeeInput,
    PredictionResponse,
    PredictiveFactor,
    BulkPredictionResponse,
    BulkItemResult,
)

logger = logging.getLogger(__name__)

# System/Constant columns to exclude from HR factor display
EXCLUDED_FACTOR_COLS = {"EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"}

class ModelService:
    _instance = None

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.classifier = None
        self.feature_names = []
        self.baseline_feature_means = None
        self.training_defaults = {}
        self._load_model()
        self._compute_training_baselines()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Loads the trained scikit-learn pipeline from disk once."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info(f"Loading trained model from {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = self.model.named_steps["preprocessor"]
        self.classifier = self.model.named_steps["classifier"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        logger.info(f"Model loaded successfully with {len(self.feature_names)} transformed features.")

    def _compute_training_baselines(self):
        """Computes baseline statistics and default values from training data."""
        if DATASET_PATH.exists():
            try:
                df = pd.read_csv(DATASET_PATH)
                X = df.drop(columns=["Attrition"], errors="ignore")
                
                # Compute default values for missing columns
                for col in NUMERICAL_FEATURES:
                    if col in X.columns:
                        self.training_defaults[col] = float(X[col].median())
                    else:
                        self.training_defaults[col] = 0.0
                for col in CATEGORICAL_FEATURES:
                    if col in X.columns:
                        self.training_defaults[col] = str(X[col].mode()[0])
                    else:
                        self.training_defaults[col] = "Unknown"

                # Transformed features baseline mean
                X_trans = np.asarray(self.preprocessor.transform(X))
                self.baseline_feature_means = X_trans.mean(axis=0)
                logger.info("Baseline training feature means computed successfully.")
            except Exception as e:
                logger.warning(f"Could not compute dataset baselines: {e}")
                self.baseline_feature_means = np.zeros(len(self.feature_names))
        else:
            self.baseline_feature_means = np.zeros(len(self.feature_names))

    def _get_risk_category(self, probability: float) -> Tuple[str, str]:
        """Categorizes probability into Risk Level and badge CSS class."""
        if probability >= RISK_THRESHOLD_MEDIUM:
            return "High", "badge-high"
        elif probability >= RISK_THRESHOLD_LOW:
            return "Medium", "badge-medium"
        else:
            return "Low", "badge-low"

    def _generate_factor_description(self, clean_name: str, raw_val: Any, is_risk: bool) -> str:
        """Generates clear, natural language explanation for HR decision makers."""
        # OverTime
        if clean_name == "OverTime":
            if str(raw_val).strip().lower() in ["yes", "1", "1.0", "y"]:
                return "Frequent overtime assignments contribute to elevated burnout risk."
            return "Standard working hours without overtime protect employee retention."

        # Job Satisfaction & Environment
        if clean_name == "JobSatisfaction":
            try:
                val = int(raw_val)
            except Exception:
                val = 3
            if val <= 2:
                return f"Lower job satisfaction score ({val}/4) is a prominent predictor of disengagement."
            return f"High job satisfaction score ({val}/4) reinforces positive organizational commitment."

        if clean_name == "EnvironmentSatisfaction":
            try:
                val = int(raw_val)
            except Exception:
                val = 3
            if val <= 2:
                return f"Workplace environment rating ({val}/4) is below organizational benchmarks."
            return f"Favorable physical & cultural workplace rating ({val}/4) supports retention."

        if clean_name == "JobInvolvement":
            try:
                val = int(raw_val)
            except Exception:
                val = 3
            if val <= 2:
                return f"Reduced day-to-day job involvement ({val}/4) signals potential detachment."
            return f"Active daily job involvement ({val}/4) is a strong protective retention factor."

        if clean_name == "WorkLifeBalance":
            try:
                val = int(raw_val)
            except Exception:
                val = 3
            if val <= 2:
                return f"Low work-life balance score ({val}/4) indicates personal-professional friction."
            return f"Balanced work-life rating ({val}/4) fosters sustained productivity."

        # Stock Options & Compensation
        if clean_name == "StockOptionLevel":
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val == 0:
                return "Absence of equity/stock options (Level 0) reduces long-term incentive alignment."
            return f"Stock option incentives (Level {val}) provide meaningful long-term financial vesting."

        if clean_name == "MonthlyIncome":
            try:
                val = int(raw_val)
                return f"Monthly compensation (${val:,}) relative to industry benchmarks."
            except Exception:
                return f"Monthly compensation (${raw_val}) relative to industry benchmarks."

        # Tenure & Experience
        if clean_name == "YearsSinceLastPromotion":
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val >= 4:
                return f"Extended duration without career advancement ({val} years since last promotion)."
            return f"Recent promotional recognition ({val} years ago) maintains career momentum."

        if clean_name == "YearsWithCurrManager":
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val <= 1:
                return f"Short tenure with current reporting manager ({val} years) in adjustment phase."
            return f"Established reporting relationship with current manager ({val} years)."

        if clean_name == "NumCompaniesWorked":
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val >= 5:
                return f"High historical mobility ({val} previous employers) indicates higher transition propensity."
            return f"Stable historical employment track record ({val} previous companies)."

        if clean_name == "DistanceFromHome":
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val > 15:
                return f"Extended daily commute distance ({val} miles) adds daily fatigue."
            return f"Convenient commute distance ({val} miles) minimizes transit burden."

        if clean_name == "BusinessTravel":
            if "Frequently" in str(raw_val):
                return "Frequent business travel schedule creates heightened physical and scheduling demands."
            return "Low travel commitments maintain work routine stability."

        if clean_name == "MaritalStatus":
            return f"Marital status profile ({raw_val})."

        # Fallback
        label = FEATURE_LABELS.get(clean_name, clean_name)
        if is_risk:
            return f"{label} value ({raw_val}) increases model-estimated attrition risk."
        return f"{label} value ({raw_val}) decreases model-estimated attrition risk."

    def _generate_hr_recommendations(self, top_risk_factors: List[PredictiveFactor], risk_cat: str) -> List[str]:
        """Generates actionable HR recommendations based on the highest risk contributors."""
        recs = []
        risk_names = [f.feature_name for f in top_risk_factors]

        if any("OverTime" in n for n in risk_names):
            recs.append("Workload Optimization: Review project deadlines and consider reallocating overtime tasks or offering compensatory time off.")
        
        if any("StockOptionLevel" in n or "MonthlyIncome" in n for n in risk_names):
            recs.append("Compensation Review: Conduct a peer market salary review and consider equity/stock option grant allocations.")
            
        if any("JobSatisfaction" in n or "EnvironmentSatisfaction" in n for n in risk_names):
            recs.append("Engagement Check-in: Schedule a 1-on-1 pulse check to discuss team environment, tool availability, and pain points.")
            
        if any("YearsSinceLastPromotion" in n for n in risk_names):
            recs.append("Career Progression: Formulate a clear 6-to-12 month promotion trajectory and skill development milestones.")
            
        if any("WorkLifeBalance" in n or "DistanceFromHome" in n for n in risk_names):
            recs.append("Workplace Flexibility: Offer hybrid/remote schedule options to mitigate commute strain and balance personal priorities.")

        if not recs:
            if risk_cat == "High":
                recs.append("Schedule a confidential retention interview to understand current employee concerns and career aspirations.")
            elif risk_cat == "Medium":
                recs.append("Maintain periodic quarterly reviews to keep engagement and morale strong.")
            else:
                recs.append("Continue current supportive management practices and positive recognition.")

        return recs

    def compute_feature_attributions(self, df_sample: pd.DataFrame) -> Tuple[List[PredictiveFactor], List[PredictiveFactor]]:
        """
        Computes exact linear feature attributions (SHAP-equivalent log-odds contribution)
        Contribution_i = (x_i - mean_i) * beta_i
        """
        X_trans = np.asarray(self.preprocessor.transform(df_sample))[0]
        if self.baseline_feature_means is not None and len(self.baseline_feature_means) == len(X_trans):
            diff = X_trans - self.baseline_feature_means
        else:
            diff = X_trans

        coefs = self.classifier.coef_[0]
        contributions = diff * coefs

        raw_row = df_sample.iloc[0]

        factors = []
        for i, name in enumerate(self.feature_names):
            contrib = float(contributions[i])
            
            # Map back to original feature name & value
            if name.startswith("cat__"):
                parts = name[5:].split("_", 1)
                orig_col = parts[0]
            elif name.startswith("num__"):
                orig_col = name[5:]
            else:
                orig_col = name

            if orig_col in EXCLUDED_FACTOR_COLS:
                continue

            raw_v = raw_row.get(orig_col, "Unknown")
            if isinstance(raw_v, (np.integer, int)):
                val = int(raw_v)
            elif isinstance(raw_v, (np.floating, float)):
                val = float(raw_v)
            else:
                val = str(raw_v)

            label = FEATURE_LABELS.get(orig_col, orig_col)
            
            factors.append({
                "feature_name": orig_col,
                "feature_label": label,
                "feature_value": val,
                "importance_score": contrib,
            })

        # Aggregate contributions by original feature name
        aggregated = {}
        for f in factors:
            col = f["feature_name"]
            if col not in aggregated:
                aggregated[col] = f.copy()
            else:
                aggregated[col]["importance_score"] += f["importance_score"]

        agg_list = list(aggregated.values())

        # Sort by importance score
        risk_increasing = [f for f in agg_list if f["importance_score"] > 0.03]
        risk_decreasing = [f for f in agg_list if f["importance_score"] < -0.03]

        risk_increasing.sort(key=lambda x: x["importance_score"], reverse=True)
        risk_decreasing.sort(key=lambda x: x["importance_score"])

        top_risks = []
        for item in risk_increasing[:5]:
            top_risks.append(PredictiveFactor(
                feature_name=item["feature_name"],
                feature_label=item["feature_label"],
                feature_value=item["feature_value"],
                impact="Increases Risk",
                importance_score=round(float(item["importance_score"]), 3),
                description=self._generate_factor_description(item["feature_name"], item["feature_value"], is_risk=True)
            ))

        top_protective = []
        for item in risk_decreasing[:5]:
            top_protective.append(PredictiveFactor(
                feature_name=item["feature_name"],
                feature_label=item["feature_label"],
                feature_value=item["feature_value"],
                impact="Decreases Risk",
                importance_score=round(float(abs(item["importance_score"])), 3),
                description=self._generate_factor_description(item["feature_name"], item["feature_value"], is_risk=False)
            ))

        return top_risks, top_protective

    def predict_single(self, employee_input: EmployeeInput) -> PredictionResponse:
        """Runs end-to-end single employee prediction pipeline."""
        df_input = pd.DataFrame([employee_input.model_dump()])
        
        # Ensure all required features are present
        for col, default_val in self.training_defaults.items():
            if col not in df_input.columns:
                df_input[col] = default_val

        # Prediction & Probability
        pred_class = str(self.model.predict(df_input)[0])
        
        # Binary classification probabilities: [P('No'), P('Yes')]
        classes = list(self.model.classes_)
        proba_idx = classes.index("Yes") if "Yes" in classes else 1
        probabilities = self.model.predict_proba(df_input)[0]
        attrition_proba = float(probabilities[proba_idx])

        risk_category, badge_class = self._get_risk_category(attrition_proba)
        risk_pct = f"{attrition_proba * 100:.1f}%"
        summary_text = f"Model-estimated attrition probability: {risk_pct}"

        # Explainability: Feature attributions
        top_risks, top_protective = self.compute_feature_attributions(df_input)

        # Actionable recommendations
        recommendations = self._generate_hr_recommendations(top_risks, risk_category)

        return PredictionResponse(
            prediction=pred_class,
            attrition_risk_score=round(attrition_proba, 4),
            attrition_risk_percentage=risk_pct,
            risk_category=risk_category,
            risk_badge_class=badge_class,
            summary_wording=summary_text,
            top_risk_factors=top_risks,
            top_protective_factors=top_protective,
            hr_recommendations=recommendations,
        )

    def predict_batch_df(self, df_raw: pd.DataFrame, filename: str = "batch.csv") -> Tuple[BulkPredictionResponse, pd.DataFrame]:
        """
        Validates, cleans, and runs batch predictions on an uploaded DataFrame.
        Returns response summary and enriched output DataFrame.
        """
        df = df_raw.copy()
        
        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Check identifier column
        id_col = None
        for candidate in ["EmployeeNumber", "EmployeeId", "Employee_ID", "ID", "EmployeeCount"]:
            if candidate in df.columns:
                id_col = candidate
                break

        # Fill missing required columns with training defaults
        for col, default_val in self.training_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                # Fill nulls if any
                if pd.api.types.is_numeric_dtype(type(default_val)):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)
                else:
                    df[col] = df[col].fillna(default_val).astype(str)

        # Ensure correct column ordering for model
        X_df = df[ALL_FEATURE_COLUMNS].copy()

        # Batch prediction
        predictions = self.model.predict(X_df)
        probabilities = self.model.predict_proba(X_df)
        
        classes = list(self.model.classes_)
        yes_idx = classes.index("Yes") if "Yes" in classes else 1
        yes_probas = probabilities[:, yes_idx]

        results = []
        high_count = 0
        med_count = 0
        low_count = 0

        risk_categories = []
        risk_percentages = []

        for i, (pred, proba) in enumerate(zip(predictions, yes_probas)):
            proba_f = float(proba)
            risk_cat, _ = self._get_risk_category(proba_f)
            risk_categories.append(risk_cat)
            risk_percentages.append(f"{proba_f * 100:.1f}%")

            if risk_cat == "High":
                high_count += 1
            elif risk_cat == "Medium":
                med_count += 1
            else:
                low_count += 1

            emp_id = df.iloc[i][id_col] if id_col else (i + 1)
            if isinstance(emp_id, (np.integer, int)):
                emp_id = int(emp_id)
            elif isinstance(emp_id, (np.floating, float)):
                emp_id = int(emp_id)

            job_role = str(df.iloc[i].get("JobRole", "Not Specified"))
            dept = str(df.iloc[i].get("Department", "Not Specified"))
            
            ot = str(df.iloc[i].get("OverTime", "No"))
            js = df.iloc[i].get("JobSatisfaction", 3)
            factors_summary = f"Overtime: {ot} | Job Satisfaction: {js}/4"

            results.append(BulkItemResult(
                employee_id=emp_id,
                job_role=job_role,
                department=dept,
                prediction=str(pred),
                risk_probability=round(proba_f, 4),
                risk_percentage=f"{proba_f * 100:.1f}%",
                risk_category=risk_cat,
                top_factors_summary=factors_summary,
            ))

        # Enrich original DataFrame for CSV export
        enriched_df = df.copy()
        enriched_df["Predicted_Attrition"] = predictions
        enriched_df["Predicted_Risk_Probability"] = np.round(yes_probas, 4)
        enriched_df["Predicted_Risk_Percentage"] = risk_percentages
        enriched_df["Risk_Category"] = risk_categories

        total_processed = len(results)
        avg_proba = float(np.mean(yes_probas)) if total_processed > 0 else 0.0

        response = BulkPredictionResponse(
            total_processed=total_processed,
            high_risk_count=high_count,
            medium_risk_count=med_count,
            low_risk_count=low_count,
            average_risk_probability=round(avg_proba, 4),
            results=results,
            filename=filename,
        )

        return response, enriched_df
