import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Default blob storage configuration
    BLOB_STORAGE_URL = os.getenv(
        "BLOB_STORAGE_URL", 
        "https://devstoragecenter.blob.core.windows.net/devconteiner"
    )
    
    # Model URL can be set explicitly or constructed from environment
    MODEL_URL = os.getenv(
        "MODEL_URL", 
        f"{BLOB_STORAGE_URL}/models/noshow_model_{ENVIRONMENT}.joblib"
    )