from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Tuple
from datetime import datetime
from app.models.db_models import AppointmentPrediction
from app.models.schemas import AppointmentCreate, AppointmentStatusUpdate
from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AppointmentService:
    """Service layer for appointment operations"""
    
    @staticmethod
    def get_appointments(
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        patient_id: Optional[str] = None
    ) -> Tuple[List[AppointmentPrediction], int]:
        """
        Get appointments with pagination and optional filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            patient_id: Optional patient ID filter
            
        Returns:
            Tuple of (appointments list, total count)
        """
        logger.info(f"Fetching appointments: skip={skip}, limit={limit}, patient_id={patient_id}")
        
        query = db.query(AppointmentPrediction)
        
        # Apply filters
        if patient_id:
            query = query.filter(AppointmentPrediction.patient_id == patient_id)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination and ordering (most recent first)
        appointments = query.order_by(desc(AppointmentPrediction.created_at)).offset(skip).limit(limit).all()
        
        logger.info(f"Found {total} total appointments, returning {len(appointments)}")
        return appointments, total
    
    @staticmethod
    def get_appointment_by_id(db: Session, appointment_id: int) -> Optional[AppointmentPrediction]:
        """
        Get a single appointment by ID.
        
        Args:
            db: Database session
            appointment_id: Appointment prediction ID
            
        Returns:
            Appointment or None if not found
        """
        logger.info(f"Fetching appointment with ID: {appointment_id}")
        appointment = db.query(AppointmentPrediction).filter(
            AppointmentPrediction.appointment_prediction_id == appointment_id
        ).first()
        
        if appointment:
            logger.info(f"Appointment {appointment_id} found")
        else:
            logger.warning(f"Appointment {appointment_id} not found")
        
        return appointment
    
    @staticmethod
    def create_appointment(db: Session, appointment_data: AppointmentCreate) -> AppointmentPrediction:
        """
        Create a new appointment record.
        
        Args:
            db: Database session
            appointment_data: Appointment data to create
            
        Returns:
            Created appointment record
        """
        logger.info(f"Creating new appointment for patient: {appointment_data.patient_id}")
        
        # Convert datetime strings to datetime objects if needed
        scheduled_at = None
        appointment_at = None
        
        if appointment_data.scheduled_at:
            if isinstance(appointment_data.scheduled_at, str):
                scheduled_at = datetime.fromisoformat(appointment_data.scheduled_at.replace('Z', '+00:00'))
            else:
                scheduled_at = appointment_data.scheduled_at
        
        if appointment_data.appointment_at:
            if isinstance(appointment_data.appointment_at, str):
                appointment_at = datetime.fromisoformat(appointment_data.appointment_at.replace('Z', '+00:00'))
            else:
                appointment_at = appointment_data.appointment_at
        
        # Create appointment record
        db_appointment = AppointmentPrediction(
            model_name=appointment_data.model_name,
            patient_id=appointment_data.patient_id,
            appointment_status=appointment_data.appointment_status,
            scheduled_at=scheduled_at,
            appointment_at=appointment_at,
            patient_age=appointment_data.patient_age,
            patient_sex=appointment_data.patient_sex,
            patient_city=appointment_data.patient_city,
            patient_neighborhood=appointment_data.patient_neighborhood,
            insurance_type=appointment_data.insurance_type,
            unit_name=appointment_data.unit_name,
            unit_address=appointment_data.unit_address,
            unit_zipcode=appointment_data.unit_zipcode,
            specialty=appointment_data.specialty,
            prediction_class=appointment_data.prediction_class,
            prediction_label=appointment_data.prediction_label,
            probability_show=appointment_data.probability_show,
            probability_no_show=appointment_data.probability_no_show,
            probability_no_show_normalized=appointment_data.probability_no_show_normalized,
            threshold=appointment_data.threshold,
        )
        
        db.add(db_appointment)
        db.commit()
        db.refresh(db_appointment)
        
        logger.info(f"Appointment created with ID: {db_appointment.appointment_prediction_id}")
        return db_appointment
    
    @staticmethod
    def create_appointments_batch(
        db: Session,
        appointments_data: list
    ) -> tuple:
        """
        Create multiple appointment records in a single database transaction.

        Args:
            db: Database session
            appointments_data: List of AppointmentCreate objects

        Returns:
            Tuple of (created_appointments list, failed_count int)
        """
        logger.info(f"Creating batch of {len(appointments_data)} appointments")
        created = []
        failed = 0

        for appointment_data in appointments_data:
            try:
                scheduled_at = None
                appointment_at = None

                if appointment_data.scheduled_at:
                    if isinstance(appointment_data.scheduled_at, str):
                        scheduled_at = datetime.fromisoformat(
                            appointment_data.scheduled_at.replace('Z', '+00:00')
                        )
                    else:
                        scheduled_at = appointment_data.scheduled_at

                if appointment_data.appointment_at:
                    if isinstance(appointment_data.appointment_at, str):
                        appointment_at = datetime.fromisoformat(
                            appointment_data.appointment_at.replace('Z', '+00:00')
                        )
                    else:
                        appointment_at = appointment_data.appointment_at

                db_appointment = AppointmentPrediction(
                    model_name=appointment_data.model_name,
                    patient_id=appointment_data.patient_id,
                    appointment_status=appointment_data.appointment_status,
                    scheduled_at=scheduled_at,
                    appointment_at=appointment_at,
                    patient_age=appointment_data.patient_age,
                    patient_sex=appointment_data.patient_sex,
                    patient_city=appointment_data.patient_city,
                    patient_neighborhood=appointment_data.patient_neighborhood,
                    insurance_type=appointment_data.insurance_type,
                    unit_name=appointment_data.unit_name,
                    unit_address=appointment_data.unit_address,
                    unit_zipcode=appointment_data.unit_zipcode,
                    specialty=appointment_data.specialty,
                    prediction_class=appointment_data.prediction_class,
                    prediction_label=appointment_data.prediction_label,
                    probability_show=appointment_data.probability_show,
                    probability_no_show=appointment_data.probability_no_show,
                    probability_no_show_normalized=appointment_data.probability_no_show_normalized,
                    threshold=appointment_data.threshold,
                )
                db.add(db_appointment)
                created.append(db_appointment)
            except Exception as e:
                logger.warning(f"Failed to stage appointment for patient {appointment_data.patient_id}: {e}")
                failed += 1

        try:
            db.commit()
            for appt in created:
                db.refresh(appt)
            logger.info(f"Batch created: {len(created)} appointments, {failed} failed")
        except Exception as e:
            db.rollback()
            logger.error(f"Batch commit failed: {e}", exc_info=True)
            raise

        return created, failed

    @staticmethod
    def update_appointment_status(
        db: Session, 
        appointment_id: int, 
        status_update: 'AppointmentStatusUpdate'
    ) -> Optional[AppointmentPrediction]:
        """Update appointment status (what actually happened).Args:            db: Database session
            appointment_id: Appointment prediction ID
            status_update: Status update data (appointment_status)
            
        Returns:
            Updated appointment or None if not found
        """
        logger.info(f"Updating status for appointment {appointment_id}: status={status_update.appointment_status}")
        
        db_appointment = AppointmentService.get_appointment_by_id(db, appointment_id)
        
        if not db_appointment:
            logger.warning(f"Cannot update status: appointment {appointment_id} not found")
            return None
        
        # Update status
        db_appointment.appointment_status = status_update.appointment_status
        db_appointment.updated_at = datetime.now()
        
        db.commit()
        db.refresh(db_appointment)
        
        logger.info(f"Appointment {appointment_id} status updated successfully")
        return db_appointment

    @staticmethod
    def update_appointments_feedback_batch(
        db: Session,
        feedbacks: list
    ) -> tuple:
        """
        Update status for multiple appointments in a single database transaction.

        Args:
            db: Database session
            feedbacks: List of FeedbackItem objects (appointment_id + appointment_status)

        Returns:
            Tuple of (updated_appointments list, failed_count int)
        """
        logger.info(f"Batch feedback update for {len(feedbacks)} appointments")
        updated = []
        failed = 0

        for item in feedbacks:
            db_appointment = AppointmentService.get_appointment_by_id(db, item.appointment_id)
            if not db_appointment:
                logger.warning(f"Appointment {item.appointment_id} not found, skipping")
                failed += 1
                continue
            db_appointment.appointment_status = item.appointment_status
            db_appointment.updated_at = datetime.now()
            updated.append(db_appointment)

        try:
            db.commit()
            for appt in updated:
                db.refresh(appt)
            logger.info(f"Batch feedback complete: {len(updated)} updated, {failed} failed")
        except Exception as e:
            db.rollback()
            logger.error(f"Batch feedback commit failed: {e}", exc_info=True)
            raise

        return updated, failed
