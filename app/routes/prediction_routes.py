from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.prediction_service import predict, predict_batch, predict_range
from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    RangePredictionRequest,
    RangePredictionResponse,
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


@router.post(
    "/predict/range",
    response_model=RangePredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_range_predictions(request: RangePredictionRequest):
    """
    Predict patient no-show probability across a range of dates.
    
    This endpoint analyzes how appointment date affects no-show probability for
    a specific patient/appointment. Useful for:
    - Calendar visualization of no-show risk
    - Finding optimal appointment dates
    - Understanding temporal patterns in patient behavior
    
    **How it works:**
    - Takes one appointment with all patient/appointment details
    - Generates predictions for the same appointment on different dates
    - Returns no-show probability for each date in the range
    
    **Use case example:**
    Patient has an appointment on Nov 23. You want to see if scheduling it on
    Nov 24, 25, 26, or 27 would have lower no-show risk. This endpoint returns
    the probability for all those dates so you can visualize it in a calendar
    and choose the best date.
    
    **Limits:**
    - Range: 3-5 days starting from the appointment date
    - Single patient/appointment per request
    """
    try:
        logger.info(
            f"Received range prediction request for patient {request.appointment.idUnicoPaciente}, "
            f"range: {request.range_days} days"
        )
        
        # Convert pydantic model to dict
        appointment_data = request.appointment.model_dump()
        
        # Process range prediction
        result = await predict_range(appointment_data, request.range_days)
        
        logger.info(
            f"Range prediction completed: avg no-show risk {result['summary']['avg_probability_no_show']:.2%}, "
            f"best date: {result['summary']['best_date']}"
        )
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as e:
        logger.error(f"Invalid input data in range prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during range prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
