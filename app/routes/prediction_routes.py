from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.prediction_service import predict, convert_to_native_types
from app.config import Config
from app.models.schemas import PredictionResponse, ErrorResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/predict",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Patient not found"}
    },
    summary="Get no-show prediction",
    description="Returns a prediction for whether a patient will show up for their appointment"
)
async def get_prediction():
    try:
        logger.info(f"Received prediction request for patient {Config.HARDCODED_PATIENT_ID}")
        result = await predict(Config.HARDCODED_PATIENT_ID)
        result = convert_to_native_types(result)
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error(f"Patient not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
