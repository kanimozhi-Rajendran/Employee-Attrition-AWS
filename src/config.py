import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

DATASET_PATH = DATA_DIR / "employee_data.csv"
MODEL_PATH = MODELS_DIR / "attrition_model.pkl"

# Risk Categorization Thresholds
# Model outputs P(Attrition = 'Yes') in range [0, 1]
RISK_THRESHOLD_LOW = 0.30      # Below 30%: Low Risk
RISK_THRESHOLD_MEDIUM = 0.60   # 30% to 60%: Medium Risk
                                # Above 60%: High Risk

# Feature Definitions
CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime",
]

NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Human-friendly labels for features
FEATURE_LABELS = {
    "OverTime": "Overtime Workload",
    "MaritalStatus": "Marital Status",
    "BusinessTravel": "Business Travel Frequency",
    "Department": "Department",
    "JobRole": "Job Role",
    "EducationField": "Education Background",
    "Gender": "Gender",
    "NumCompaniesWorked": "Past Employers Count",
    "YearsSinceLastPromotion": "Years Since Last Promotion",
    "DistanceFromHome": "Commute Distance (miles)",
    "JobSatisfaction": "Job Satisfaction Level",
    "EnvironmentSatisfaction": "Workplace Environment Satisfaction",
    "JobInvolvement": "Job Involvement Level",
    "StockOptionLevel": "Stock Option Grant Level",
    "WorkLifeBalance": "Work-Life Balance Score",
    "MonthlyIncome": "Monthly Salary ($)",
    "YearsAtCompany": "Tenure at Company",
    "YearsWithCurrManager": "Years With Current Manager",
    "YearsInCurrentRole": "Years in Current Role",
    "TotalWorkingYears": "Total Working Experience",
    "Age": "Employee Age",
    "RelationshipSatisfaction": "Team Relationship Satisfaction",
    "TrainingTimesLastYear": "Training Sessions Last Year",
    "PercentSalaryHike": "Latest Salary Hike %",
    "PerformanceRating": "Performance Evaluation Rating",
}
