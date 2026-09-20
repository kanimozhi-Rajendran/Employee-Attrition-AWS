import os
import joblib
import pandas as pd

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Create employee input
employee = {
    "Age": 30,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 9999,
    "EnvironmentSatisfaction": 3,
    "Gender": "Female",
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
}

# Convert input into DataFrame
employee_df = pd.DataFrame([employee])

# Prediction
prediction = model.predict(employee_df)[0]

# Probability
probability = model.predict_proba(employee_df)[0]

print("====================================")
print("   EMPLOYEE ATTRITION PREDICTION")
print("====================================")

print(f"\nPrediction: {prediction}")

print("\nPrediction Probability:")
for class_name, prob in zip(model.classes_, probability):
    print(f"{class_name}: {prob:.2%}")

print("====================================")