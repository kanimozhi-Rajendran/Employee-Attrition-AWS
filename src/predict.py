import os
import sys
from pathlib import Path

# Add parent directory to sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.model_service import ModelService
from src.schemas import EmployeeInput

def main():
    print("================================================================")
    print("   EMPLOYEE ATTRITION RISK & HR DECISION SUPPORT SYSTEM")
    print("================================================================")

    # Initialize model service (loads model & baseline attributions once)
    service = ModelService.get_instance()

    # Sample Employee Profile
    sample_employee = EmployeeInput(
        Age=30,
        Gender="Female",
        MaritalStatus="Single",
        Department="Research & Development",
        JobRole="Research Scientist",
        JobLevel=2,
        BusinessTravel="Travel_Rarely",
        DistanceFromHome=5,
        Education=3,
        EducationField="Life Sciences",
        MonthlyIncome=5000,
        DailyRate=800,
        HourlyRate=60,
        MonthlyRate=15000,
        PercentSalaryHike=15,
        StockOptionLevel=1,
        EnvironmentSatisfaction=3,
        JobSatisfaction=3,
        JobInvolvement=3,
        RelationshipSatisfaction=3,
        WorkLifeBalance=3,
        PerformanceRating=3,
        OverTime="No",
        NumCompaniesWorked=2,
        TotalWorkingYears=8,
        TrainingTimesLastYear=3,
        YearsAtCompany=5,
        YearsInCurrentRole=3,
        YearsSinceLastPromotion=1,
        YearsWithCurrManager=3,
    )

    # Execute Prediction Pipeline
    result = service.predict_single(sample_employee)

    print("\n--- PREDICTION SUMMARY ---")
    print(f"Prediction Result     : {result.prediction}")
    print(f"Risk Level Category   : {result.risk_category} Risk")
    print(f"Attrition Probability : {result.attrition_risk_percentage}")
    print(f"Formal Summary        : {result.summary_wording}")

    print("\n--- TOP RISK-INCREASING FACTORS ---")
    if result.top_risk_factors:
        for f in result.top_risk_factors:
            print(f" [+] [{f.feature_label}]: {f.feature_value} -> {f.description}")
    else:
        print(" [i] No significant elevated risk factors identified.")

    print("\n--- TOP PROTECTIVE RETENTION FACTORS ---")
    if result.top_protective_factors:
        for f in result.top_protective_factors:
            print(f" [-] [{f.feature_label}]: {f.feature_value} -> {f.description}")
    else:
        print(" [i] Standard baseline protective attributes.")

    print("\n--- ACTIONABLE HR RECOMMENDATIONS ---")
    for rec in result.hr_recommendations:
        print(f" [>] {rec}")

    print("================================================================")

if __name__ == "__main__":
    main()