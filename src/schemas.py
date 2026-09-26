from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class EmployeeInput(BaseModel):
    # Demographics & Status
    Age: int = Field(30, ge=18, le=70, description="Age in years (18-70)")
    Gender: str = Field("Female", description="Gender: Female, Male")
    MaritalStatus: str = Field("Single", description="Marital Status: Single, Married, Divorced")
    
    # Organization & Role
    Department: str = Field("Research & Development", description="Department: Sales, Research & Development, Human Resources")
    JobRole: str = Field("Research Scientist", description="Job role in company")
    JobLevel: int = Field(2, ge=1, le=5, description="Job level (1 to 5)")
    BusinessTravel: str = Field("Travel_Rarely", description="Travel frequency: Travel_Rarely, Travel_Frequently, Non-Travel")
    DistanceFromHome: int = Field(5, ge=1, le=60, description="Commute distance in miles")
    
    # Education
    Education: int = Field(3, ge=1, le=5, description="Education level: 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor")
    EducationField: str = Field("Life Sciences", description="Field of study")
    
    # Compensation & Financials
    MonthlyIncome: int = Field(5000, ge=1000, le=30000, description="Monthly salary in USD")
    DailyRate: int = Field(800, ge=100, le=2000, description="Daily rate")
    HourlyRate: int = Field(60, ge=20, le=200, description="Hourly rate")
    MonthlyRate: int = Field(15000, ge=1000, le=40000, description="Monthly rate")
    PercentSalaryHike: int = Field(15, ge=0, le=50, description="Salary hike percentage")
    StockOptionLevel: int = Field(1, ge=0, le=3, description="Stock option grant level (0 to 3)")
    
    # Work Conditions & Satisfaction (1 to 4 scale)
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, description="Environment satisfaction (1-Low to 4-Very High)")
    JobSatisfaction: int = Field(3, ge=1, le=4, description="Job satisfaction (1-Low to 4-Very High)")
    JobInvolvement: int = Field(3, ge=1, le=4, description="Job involvement (1-Low to 4-Very High)")
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, description="Relationship satisfaction (1-Low to 4-Very High)")
    WorkLifeBalance: int = Field(3, ge=1, le=4, description="Work-life balance (1-Bad to 4-Best)")
    PerformanceRating: int = Field(3, ge=1, le=4, description="Performance rating (1-Low to 4-Outstanding)")
    OverTime: str = Field("No", description="Overtime work: Yes, No")
    
    # Experience & History
    NumCompaniesWorked: int = Field(2, ge=0, le=20, description="Number of prior companies")
    TotalWorkingYears: int = Field(8, ge=0, le=50, description="Total career experience in years")
    TrainingTimesLastYear: int = Field(3, ge=0, le=10, description="Training sessions attended last year")
    YearsAtCompany: int = Field(5, ge=0, le=40, description="Years with current company")
    YearsInCurrentRole: int = Field(3, ge=0, le=30, description="Years in current role")
    YearsSinceLastPromotion: int = Field(1, ge=0, le=30, description="Years since last promotion")
    YearsWithCurrManager: int = Field(3, ge=0, le=30, description="Years with current manager")
    
    # Constant/System fields (optional with standard defaults)
    EmployeeCount: Optional[int] = Field(1, description="Always 1")
    EmployeeNumber: Optional[int] = Field(9999, description="Employee identifier")
    Over18: Optional[str] = Field("Y", description="Over 18 confirmation")
    StandardHours: Optional[int] = Field(80, description="Standard working hours")


class PredictiveFactor(BaseModel):
    feature_name: str
    feature_label: str
    feature_value: Any
    impact: str  # "Increases Risk" or "Decreases Risk"
    importance_score: float
    description: str


class PredictionResponse(BaseModel):
    prediction: str                    # "Yes" or "No"
    attrition_risk_score: float        # e.g. 0.824
    attrition_risk_percentage: str     # e.g. "82.4%"
    risk_category: str                 # "Low", "Medium", "High"
    risk_badge_class: str              # "badge-low", "badge-medium", "badge-high"
    summary_wording: str               # "Model-estimated attrition probability: 82.4%"
    top_risk_factors: List[PredictiveFactor]
    top_protective_factors: List[PredictiveFactor]
    hr_recommendations: List[str]


class BulkItemResult(BaseModel):
    employee_id: Any
    job_role: str
    department: str
    prediction: str
    risk_probability: float
    risk_percentage: str
    risk_category: str
    top_factors_summary: str


class BulkPredictionResponse(BaseModel):
    total_processed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    average_risk_probability: float
    results: List[BulkItemResult]
    filename: Optional[str] = None
