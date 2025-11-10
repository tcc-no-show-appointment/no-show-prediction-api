from fastapi import FastAPI
from app.routes.prediction_routes import router as prediction_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="No-Show Prediction API",
    description="API for predicting whether patients will show up for their medical appointments",
    version="1.0.0"
)

app.include_router(prediction_router, tags=["predictions"])

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "message": "No-Show Prediction API is running"}
