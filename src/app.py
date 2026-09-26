import io
import time
import logging
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from src.config import STATIC_DIR
from src.schemas import EmployeeInput, PredictionResponse, BulkPredictionResponse
from src.model_service import ModelService
from src.analytics_service import AnalyticsService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AttritionApp")

START_TIME = time.time()

# Initialize FastAPI App
app = FastAPI(
    title="Employee Attrition Risk & HR Decision Support API",
    description="Production ML API for predicting employee attrition risk, probability, and key driving factors.",
    version="2.0.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons at startup
@app.on_event("startup")
def startup_event():
    logger.info("Initializing ML Model Service and Analytics Service...")
    ModelService.get_instance()
    AnalyticsService.get_instance()
    logger.info("Application startup complete.")


# API Endpoints
@app.get("/health", summary="Health Check")
def health_check():
    """Returns the operational status of the service and model readiness."""
    model_service = ModelService.get_instance()
    is_ready = model_service.model is not None
    uptime_seconds = round(time.time() - START_TIME, 2)
    
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_type": "Scikit-Learn Logistic Regression Pipeline",
        "feature_count": len(model_service.feature_names),
        "uptime_seconds": uptime_seconds,
        "version": "2.0.0",
        "cloud_ready": True
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Single Employee Attrition Risk")
def predict_single_employee(employee: EmployeeInput):
    """
    Analyzes a single employee's profile.
    Returns prediction ('Yes'/'No'), probability (0.0 - 1.0), risk level (Low/Medium/High),
    and key predictive factors (both risk-increasing and protective).
    """
    try:
        model_service = ModelService.get_instance()
        result = model_service.predict_single(employee)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the prediction: {str(e)}"
        )


@app.post("/predict-bulk", response_model=BulkPredictionResponse, summary="Batch Prediction via CSV Upload")
async def predict_bulk_csv(file: UploadFile = File(...)):
    """
    Upload an employee CSV file for batch processing.
    Validates columns, applies model inference across all rows, and categorizes attrition risk.
    """
    if not file.filename.endswith((".csv", ".CSV")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid .csv file."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        df_upload = pd.read_csv(io.BytesIO(contents))
        if df_upload.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded CSV has no data rows."
            )

        model_service = ModelService.get_instance()
        bulk_resp, _ = model_service.predict_batch_df(df_upload, filename=file.filename)
        return bulk_resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process CSV file. Ensure it is valid comma-separated text. Details: {str(e)}"
        )


@app.get("/analytics", summary="Aggregated HR Analytics")
def get_analytics():
    """Returns dataset summary metrics and pre-computed chart distributions."""
    try:
        analytics_service = AnalyticsService.get_instance()
        data = analytics_service.get_analytics()
        return data
    except Exception as e:
        logger.error(f"Analytics retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@app.get("/sample-presets", summary="Get Sample Employee Profiles for Testing")
def get_sample_presets():
    """Returns pre-configured employee profiles for instant UI testing."""
    return {
        "high_risk": {
            "title": "High Attrition Risk Profile (Burnout & Low Equity)",
            "description": "Single employee, frequent travel, mandatory overtime, no stock options, low satisfaction, 4 years without promotion.",
            "data": {
                "Age": 29,
                "Gender": "Male",
                "MaritalStatus": "Single",
                "Department": "Sales",
                "JobRole": "Sales Representative",
                "JobLevel": 1,
                "BusinessTravel": "Travel_Frequently",
                "DistanceFromHome": 24,
                "Education": 3,
                "EducationField": "Marketing",
                "MonthlyIncome": 2800,
                "DailyRate": 500,
                "HourlyRate": 45,
                "MonthlyRate": 12000,
                "PercentSalaryHike": 11,
                "StockOptionLevel": 0,
                "EnvironmentSatisfaction": 1,
                "JobSatisfaction": 1,
                "JobInvolvement": 2,
                "RelationshipSatisfaction": 2,
                "WorkLifeBalance": 1,
                "PerformanceRating": 3,
                "OverTime": "Yes",
                "NumCompaniesWorked": 6,
                "TotalWorkingYears": 6,
                "TrainingTimesLastYear": 1,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 2,
                "YearsSinceLastPromotion": 4,
                "YearsWithCurrManager": 0
            }
        },
        "low_risk": {
            "title": "Low Attrition Risk Profile (High Engagement & Stability)",
            "description": "Married employee, high job satisfaction, stock options, standard hours, excellent work-life balance.",
            "data": {
                "Age": 42,
                "Gender": "Female",
                "MaritalStatus": "Married",
                "Department": "Research & Development",
                "JobRole": "Manager",
                "JobLevel": 4,
                "BusinessTravel": "Travel_Rarely",
                "DistanceFromHome": 3,
                "Education": 4,
                "EducationField": "Life Sciences",
                "MonthlyIncome": 13500,
                "DailyRate": 1100,
                "HourlyRate": 85,
                "MonthlyRate": 22000,
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
                "TotalWorkingYears": 18,
                "TrainingTimesLastYear": 4,
                "YearsAtCompany": 12,
                "YearsInCurrentRole": 8,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 7
            }
        },
        "moderate_risk": {
            "title": "Moderate Attrition Risk Profile (Mid-Career Transition)",
            "description": "Mid-level research scientist with average satisfaction, moderate travel, and moderate commute.",
            "data": {
                "Age": 34,
                "Gender": "Male",
                "MaritalStatus": "Single",
                "Department": "Research & Development",
                "JobRole": "Research Scientist",
                "JobLevel": 2,
                "BusinessTravel": "Travel_Rarely",
                "DistanceFromHome": 10,
                "Education": 3,
                "EducationField": "Medical",
                "MonthlyIncome": 5200,
                "DailyRate": 850,
                "HourlyRate": 65,
                "MonthlyRate": 16000,
                "PercentSalaryHike": 14,
                "StockOptionLevel": 1,
                "EnvironmentSatisfaction": 2,
                "JobSatisfaction": 2,
                "JobInvolvement": 3,
                "RelationshipSatisfaction": 3,
                "WorkLifeBalance": 2,
                "PerformanceRating": 3,
                "OverTime": "Yes",
                "NumCompaniesWorked": 3,
                "TotalWorkingYears": 9,
                "TrainingTimesLastYear": 2,
                "YearsAtCompany": 4,
                "YearsInCurrentRole": 2,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 2
            }
        }
    }


# Mount static assets and serve main UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse, summary="Main Web Dashboard")
@app.get("/dashboard", response_class=HTMLResponse, summary="Main Web Dashboard")
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Employee Attrition Risk Prediction API is Running.</h1><p>Visit /docs for Swagger documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
