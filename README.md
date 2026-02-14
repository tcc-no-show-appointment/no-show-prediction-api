# No-Show Prediction API - Project Logic

## Overview

FastAPI-based prediction service that predicts patient no-shows for medical appointments using a trained ML model and the `noshow_lib` library.

## Architecture Flow

```
Request → API Route → Prediction Service → Config (Blob Storage) → build_features → Model → Response
```

## File Structure & Responsibilities

### **API Layer**

- **`main.py`**: FastAPI app initialization and health endpoint
- **`routes/prediction_routes.py`**: `/predict` endpoint handler
- **`models/schemas.py`**: Pydantic request/response models (Portuguese column names)

### **Service Layer**

- **`services/prediction_service.py`**: Main prediction pipeline
  - Downloads prod.yaml configuration from blob storage
  - Converts request to DataFrame (Portuguese columns)
  - Calls `build_features()` from noshow_lib for feature engineering
  - Filters features based on config
  - Loads model and makes prediction
- **`services/model_loader.py`**: Model caching singleton
  - Downloads model from Azure Blob on first call
  - Caches in memory for subsequent requests
- **`services/blob_service.py`**: Downloads files from Azure Blob Storage
  - Model files (.joblib)
  - Configuration files (prod.yaml)

### **Configuration**

- **`config.py`**: Environment variables (Azure credentials, model URL)
- **`constants.py`**: Static labels (Show/No-show)
- **`utils/logger.py`**: Logging configuration
- **`prod.yaml`**: Feature engineering config (downloaded from blob storage)

## Data Flow

1. **Input**: Patient data with Portuguese column names (id, Status, Marcacao, DataHoraConsulta, Idade, Sexo, CidadePaciente, BairroPaciente, TipoConvenio, idUnicoPaciente, UnidadeAtendimento, EnderecoUnidadeAtendimento, CEPUnidadeAtendimento, Especialidade)
2. **Configuration**: Download prod.yaml from blob storage (model_configuration folder)
3. **Processing**:
   - `build_features()` - handles column mapping, feature engineering
   - Creates temporal features (waiting_days, weekday, hour, cyclical encodings)
   - Creates patient history features (no_show_rate, previous appointments)
   - Creates contextual features (unit/specialty rates, holidays)
4. **Feature Selection**: Filter to model-expected features from config
5. **Prediction**: Model predicts probability of no-show
6. **Output**: Prediction class (0/1) + probabilities

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
