# No-Show Prediction API - Project Logic

## Overview

FastAPI-based prediction service that predicts patient no-shows for medical appointments using a trained ML model and the `noshow_lib` library.

## Architecture Flow

```
Request → API Route → Prediction Service → noshow_lib → Model → Response
```

## File Structure & Responsibilities

### **API Layer**

- **`main.py`**: FastAPI app initialization and health endpoint
- **`routes/prediction_routes.py`**: `/predict` endpoint handler
- **`models/schemas.py`**: Pydantic request/response models

### **Service Layer**

- **`services/prediction_service.py`**: Main prediction pipeline
  - Converts request to DataFrame
  - Calls `noshow_lib` for data processing & feature engineering
  - Loads model and makes prediction
- **`services/model_loader.py`**: Model caching singleton
  - Downloads model from Azure Blob on first call
  - Caches in memory for subsequent requests
- **`services/blob_service.py`**: Downloads joblib files from URLs

### **Configuration**

- **`config.py`**: Environment variables (MODEL_URL)
- **`constants.py`**: Static labels (Show/No-show)
- **`utils/logger.py`**: Logging configuration

## Data Flow

1. **Input**: Patient data (Age, Gender, AppointmentDay, etc.)
2. **Processing**:
   - Add required columns (AppointmentID, No-show placeholder)
   - `load_and_process_data()` - handles missing values, type casting
   - `build_features()` - creates ML features (waiting_days, age_group, etc.)
3. **Prediction**: Model predicts probability of no-show
4. **Output**: Prediction class (0/1) + probabilities

## Test Strategy

### **test_blob_service.py** (2 tests)

- ✅ Successful joblib download and parsing
- ✅ Network/parsing error handling

### **test_model_loader.py** (1 test)

- ✅ Model loading

### **test_prediction.py** (2 tests)

- ✅ Successful prediction with valid input
- ✅ Validation error with missing fields

## Key Dependencies

- **noshow_lib**: Data processing & feature engineering pipeline
- **FastAPI**: API framework
- **pandas**: DataFrame operations
- **joblib/scikit-learn**: Model loading and prediction
- **Azure Blob Storage**: Model storage (via HTTP URL)

## CI/CD Pipeline

1. **Build**: Install dependencies
2. **Lint**: Flake8 code quality check
3. **Tests**: Pytest validation (pass/fail)
4. **Security**: Bandit security scan
5. **Deploy**: Docker image to Azure Container Apps
