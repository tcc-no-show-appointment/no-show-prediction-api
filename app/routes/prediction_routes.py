from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.prediction_service import predict, predict_batch
from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
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


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_batch_predictions(request: BatchPredictionRequest):
    """
    Predict whether patients will show up for their appointments (batch processing).
    
    This endpoint processes multiple appointments in a single request using vectorized
    operations for optimal performance. All appointments are processed together in
    a single inference pass, making it much more efficient than calling /predict
    multiple times.
    
    **Performance advantages:**
    - Single model inference call for all appointments
    - Vectorized feature engineering
    - Reduced network overhead
    - Optimal for bulk processing scenarios
    
    **Limits:**
    - Maximum 1000 appointments per request
    """
    try:
        total_appointments = len(request.appointments)
        logger.info(f"Received batch prediction request for {total_appointments} appointments")
        
        # Convert pydantic models to dicts
        appointments_data = [apt.model_dump() for apt in request.appointments]
        
        # Process batch
        result = await predict_batch(appointments_data)
        
        logger.info(f"Batch prediction completed: {result['predicted_show']} Show, {result['predicted_no_show']} No-Show")
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as e:
        logger.error(f"Invalid input data in batch: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
