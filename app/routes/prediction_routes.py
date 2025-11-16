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
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get no-show prediction",
    description="Returns a prediction for whether a patient will show up for their appointment based on provided data"
)
async def get_prediction(request: PredictionRequest):
    """
    Predict whether a patient will show up for their appointment.
    
    The prediction pipeline:
    1. Processes the raw input data
    2. Engineers features from the processed data
    3. Loads the trained model from Azure Blob Storage
    4. Makes prediction and returns probability scores
    
    Args:
        request (PredictionRequest): Patient and appointment data
        
    Returns:
        PredictionResponse: Prediction result with probabilities
    """
    try:
        logger.info(f"Received prediction request for patient {request.patient_id}")
        
        # Convert request to dictionary for processing
        raw_data = request.model_dump()
        
        # Call the prediction pipeline
        result = await predict(raw_data)
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
