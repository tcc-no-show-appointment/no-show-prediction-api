from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.prediction_service import predict
from app.models.schemas import PredictionRequest, PredictionResponse, ErrorResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_prediction(request: PredictionRequest):
    """Predict whether a patient will show up for their appointment"""
    try:
        logger.info(f"Received prediction request for patient {request.idUnicoPaciente} (appointment {request.id})")
        result = await predict(request.model_dump())
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
