import sys
from pathlib import Path
import io
import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.app import app
from src.schemas import EmployeeInput
from src.model_service import ModelService

client = TestClient(app)

def test_health_endpoint():
    """Verify health check returns healthy and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 55
    assert data["version"] == "2.0.0"

def test_single_prediction_low_risk():
    """Verify single prediction for low risk profile."""
    payload = {
        "Age": 45,
        "Gender": "Male",
        "MaritalStatus": "Married",
        "Department": "Research & Development",
        "JobRole": "Manager",
        "JobLevel": 4,
        "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 2,
        "Education": 4,
        "EducationField": "Life Sciences",
        "MonthlyIncome": 15000,
        "DailyRate": 1200,
        "HourlyRate": 80,
        "MonthlyRate": 20000,
        "PercentSalaryHike": 18,
        "StockOptionLevel": 2,
        "EnvironmentSatisfaction": 4,
        "JobSatisfaction": 4,
        "JobInvolvement": 4,
        "RelationshipSatisfaction": 4,
        "WorkLifeBalance": 4,
        "PerformanceRating": 4,
        "OverTime": "No",
        "NumCompaniesWorked": 1,
        "TotalWorkingYears": 20,
        "TrainingTimesLastYear": 3,
        "YearsAtCompany": 15,
        "YearsInCurrentRole": 10,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "No"
    assert data["risk_category"] == "Low"
    assert data["attrition_risk_score"] < 0.30
    assert "Model-estimated attrition probability" in data["summary_wording"]
    assert len(data["top_protective_factors"]) > 0
    assert len(data["hr_recommendations"]) > 0

def test_single_prediction_high_risk():
    """Verify single prediction for high risk profile (burnout/overtime)."""
    payload = {
        "Age": 28,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "JobRole": "Sales Representative",
        "JobLevel": 1,
        "BusinessTravel": "Travel_Frequently",
        "DistanceFromHome": 25,
        "Education": 2,
        "EducationField": "Marketing",
        "MonthlyIncome": 2500,
        "DailyRate": 400,
        "HourlyRate": 40,
        "MonthlyRate": 10000,
        "PercentSalaryHike": 11,
        "StockOptionLevel": 0,
        "EnvironmentSatisfaction": 1,
        "JobSatisfaction": 1,
        "JobInvolvement": 1,
        "RelationshipSatisfaction": 1,
        "WorkLifeBalance": 1,
        "PerformanceRating": 3,
        "OverTime": "Yes",
        "NumCompaniesWorked": 7,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 1,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 4,
        "YearsWithCurrManager": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Yes"
    assert data["risk_category"] == "High"
    assert data["attrition_risk_score"] >= 0.60
    assert len(data["top_risk_factors"]) > 0
    assert any("Overtime" in f["description"] or "OverTime" in f["feature_name"] for f in data["top_risk_factors"])

def test_validation_invalid_input():
    """Test validation errors for out-of-range inputs."""
    # Age out of range (> 70)
    payload = {"Age": 150}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_bulk_prediction_valid_csv():
    """Test bulk CSV upload and processing."""
    csv_data = (
        "Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,"
        "EnvironmentSatisfaction,Gender,HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,"
        "MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,OverTime,PercentSalaryHike,"
        "PerformanceRating,RelationshipSatisfaction,StockOptionLevel,TotalWorkingYears,"
        "TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager\n"
        "30,Travel_Rarely,800,Research & Development,5,3,Life Sciences,3,Female,60,3,2,Research Scientist,3,Single,5000,15000,2,No,15,3,3,1,8,3,3,5,3,1,3\n"
        "25,Travel_Frequently,400,Sales,20,2,Marketing,1,Male,40,1,1,Sales Representative,1,Single,2500,10000,6,Yes,11,3,1,0,3,1,1,1,1,2,0\n"
    )
    files = {"file": ("test_employees.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/predict-bulk", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["risk_category"] == "Low"
    assert data["results"][1]["risk_category"] == "High"

def test_bulk_prediction_empty_and_invalid_file():
    """Test handling of empty files and invalid file types."""
    # Empty CSV
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/predict-bulk", files=files)
    assert response.status_code == 400

    # Non-CSV
    files = {"file": ("bad_file.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/predict-bulk", files=files)
    assert response.status_code == 400

def test_analytics_endpoint():
    """Verify aggregated analytics endpoint."""
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "charts" in data
    assert data["summary"]["total_employees"] == 1470
    assert data["summary"]["actual_attrition_count"] == 237

def test_sample_presets_endpoint():
    """Verify sample presets endpoint."""
    response = client.get("/sample-presets")
    assert response.status_code == 200
    data = response.json()
    assert "high_risk" in data
    assert "low_risk" in data
    assert "moderate_risk" in data

def test_serve_dashboard():
    """Verify root HTML dashboard serving."""
    response = client.get("/")
    assert response.status_code == 200
    assert "HR RiskPulse" in response.text
