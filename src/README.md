# Source Modules Overview (`src/`)

This directory contains the production microservice and ML services:

- `app.py`: FastAPI application serving REST endpoints (`/health`, `/predict`, `/predict-bulk`, `/analytics`, `/sample-presets`) and mounting the static UI dashboard.
- `model_service.py`: Singleton ML service managing the loaded scikit-learn pipeline (`attrition_model.pkl`), probability calibration, mathematical log-odds feature attribution (Linear SHAP equivalent), and rule-informed HR recommendations.
- `schemas.py`: Pydantic models for single employee inputs, bulk predictions, and factor attributions with boundary validation.
- `analytics_service.py`: Computes organization-wide statistics, department distributions, and chart payloads from `data/employee_data.csv`.
- `config.py`: Centralized configuration defining file paths, risk classification thresholds, and human-friendly feature labels.
- `predict.py`: Standalone CLI execution script demonstrating single employee prediction and explainability.

For complete project documentation, system architecture, and AWS cloud deployment instructions, refer to the root [README.md](../README.md).
