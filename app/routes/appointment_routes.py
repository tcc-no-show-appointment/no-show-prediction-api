from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.services.appointment_service import AppointmentService
from app.models.schemas import (
    AppointmentCreate,
    AppointmentBatchCreate,
    AppointmentBatchResponse,
    FeedbackBatchRequest,
    FeedbackBatchResponse,
    AppointmentResponse,
    AppointmentStatusUpdate,
    AppointmentListResponse,
    ErrorResponse
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/appointments",
    response_model=AppointmentListResponse,
    responses={500: {"model": ErrorResponse}}
)
async def list_appointments(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page (max 1000)"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    db: Session = Depends(get_db)
):
    """
    Get list of appointments with pagination.
    
    Returns appointments ordered by creation date (most recent first).
    Supports filtering by patient ID.
    
    **Pagination:**
    - page: Page number (starts at 1)
    - page_size: Number of items per page (max 100)
    
    **Filtering:**
    - patient_id: Optional filter to get appointments for specific patient
    """
    try:
        skip = (page - 1) * page_size
        
        logger.info(f"Fetching appointments list: page={page}, page_size={page_size}, patient_id={patient_id}")
        
        appointments, total = AppointmentService.get_appointments(
            db=db,
            skip=skip,
            limit=page_size,
            patient_id=patient_id
        )
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "appointments": appointments
        }
        
    except Exception as e:
        logger.error(f"Error fetching appointments list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/appointments/batch",
    response_model=AppointmentBatchResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_appointments_batch(
    batch: AppointmentBatchCreate,
    db: Session = Depends(get_db)
):
    """
    Create multiple appointments in a single request.

    Processes all appointments in one database transaction, which is
    significantly more efficient than calling POST /appointments repeatedly.

    **Limits:**
    - Maximum 250 appointments per request
    """
    try:
        logger.info(f"Creating batch of {len(batch.appointments)} appointments")

        created, failed = AppointmentService.create_appointments_batch(
            db=db,
            appointments_data=batch.appointments
        )

        logger.info(f"Batch complete: {len(created)} created, {failed} failed")
        return {
            "total": len(batch.appointments),
            "created": len(created),
            "failed": failed,
            "appointments": created,
        }

    except Exception as e:
        logger.error(f"Error in batch appointment creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/appointments",
    response_model=AppointmentResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_appointment(
    appointment: AppointmentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new appointment with prediction data.
    
    This endpoint saves an appointment after the user confirms saving.
    The prediction data (probabilities, prediction class) should be included.
    The actual_no_show field is initially NULL and will be updated later via PATCH.
    
    **Typical flow:**
    1. User inputs appointment data
    2. System calls /predict to get prediction
    3. User confirms saving
    4. System calls this endpoint with appointment + prediction data
    5. Later, user updates actual outcome via PATCH /appointments/{id}
    """
    try:
        logger.info(f"Creating new appointment for patient: {appointment.patient_id}")
        
        created_appointment = AppointmentService.create_appointment(db, appointment)
        
        logger.info(f"Appointment created successfully with ID: {created_appointment.appointment_prediction_id}")
        return created_appointment
        
    except ValueError as e:
        logger.error(f"Invalid appointment data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating appointment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch(
    "/appointments/{appointment_id}",
    response_model=AppointmentResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_appointment_status(
    appointment_id: int,
    status_update: AppointmentStatusUpdate,
    db: Session = Depends(get_db)
):
    """
    Update appointment status with actual outcome.
    
    This endpoint is called when the appointment date has passed and
    the user logs what actually happened.
    
    **Possible status values:**
    - "Realizado" = Patient showed up
    - "Falta" = Patient did not show up (no-show)
    - "Cancelado" = Appointment was canceled
    
    **Use case:**
    After an appointment date passes, clinic staff update the system with
    what actually happened. This data can later be used for model retraining.
    
    **Note:**
    When appointment_status is updated to a final state (Realizado/Falta/Cancelado),
    this record becomes a candidate for moving to the appointment_training_data table
    for future model training.
    """
    try:
        logger.info(f"Updating status for appointment {appointment_id}")
        
        updated_appointment = AppointmentService.update_appointment_status(
            db=db,
            appointment_id=appointment_id,
            status_update=status_update
        )
        
        if not updated_appointment:
            logger.warning(f"Appointment {appointment_id} not found")
            raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
        
        logger.info(f"Appointment {appointment_id} status updated successfully")
        return updated_appointment
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid status data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating appointment status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch(
    "/appointments/feedback/batch",
    response_model=FeedbackBatchResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_appointments_feedback_batch(
    batch: FeedbackBatchRequest,
    db: Session = Depends(get_db)
):
    """
    Register attendance feedback for multiple appointments in a single request.

    More efficient than calling PATCH /appointments/feedback/:id repeatedly.

    **Limits:**
    - Maximum 250 entries per request
    """
    try:
        logger.info(f"Batch feedback update for {len(batch.feedbacks)} appointments")

        updated, failed = AppointmentService.update_appointments_feedback_batch(
            db=db,
            feedbacks=batch.feedbacks
        )

        logger.info(f"Batch feedback complete: {len(updated)} updated, {failed} failed")
        return {
            "total": len(batch.feedbacks),
            "updated": len(updated),
            "failed": failed,
            "appointments": updated,
        }

    except Exception as e:
        logger.error(f"Error in batch feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch(
    "/appointments/feedback/{appointment_id}",
    response_model=AppointmentResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_appointment_feedback(
    appointment_id: int,
    status_update: AppointmentStatusUpdate,
    db: Session = Depends(get_db)
):
    """
    Register patient attendance feedback (actual outcome).

    Accepts the actual outcome (Realizado / Falta / Cancelado) and persists
    it to dbo.appointment_predictions.  Training data ingestion happens via
    file upload to the Training API — not from individual prediction feedback.
    """
    try:
        logger.info(f"Registering feedback for appointment {appointment_id}")

        updated_appointment = AppointmentService.update_appointment_status(
            db=db,
            appointment_id=appointment_id,
            status_update=status_update,
        )

        if not updated_appointment:
            raise HTTPException(
                status_code=404,
                detail=f"Appointment {appointment_id} not found"
            )

        return updated_appointment

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid feedback data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/appointments/{appointment_id}",
    response_model=AppointmentResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_appointment(
    appointment_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a single appointment by ID.
    
    Returns complete appointment details including prediction data
    and feedback (if already provided).
    """
    try:
        logger.info(f"Fetching appointment {appointment_id}")
        
        appointment = AppointmentService.get_appointment_by_id(db, appointment_id)
        
        if not appointment:
            logger.warning(f"Appointment {appointment_id} not found")
            raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
        
        return appointment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching appointment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
