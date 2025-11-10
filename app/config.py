import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    CSV_URL = os.getenv("CSV_URL", "https://devstoragecenter.blob.core.windows.net/devconteiner/noshowappointments.csv")
    MODEL_URL = os.getenv("MODEL_URL", "https://devstoragecenter.blob.core.windows.net/devconteiner/rf_grid_model.pkl")
    
    HARDCODED_PATIENT_ID = int(os.getenv("HARDCODED_PATIENT_ID", "7542951368435"))
