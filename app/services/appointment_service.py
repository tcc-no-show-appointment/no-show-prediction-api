import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from app.models.db_models import AppointmentPrediction
from app.models.schemas import AppointmentCreate, AppointmentStatusUpdate
from app.config import config
from app.utils.logger import get_logger
from app.constants import FEATURES_TO_TRAINING_MAP
from noshow_lib.feature_engineering import build_features

logger = get_logger(__name__)

_ODBC_MAX_PARAMS = 2100


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
            probability_no_show=appointment_data.probability_no_show
        )
        
        db.add(db_appointment)
        db.commit()
        db.refresh(db_appointment)
        
        logger.info(f"Appointment created with ID: {db_appointment.appointment_prediction_id}")
        return db_appointment
    
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
    def process_feedback_to_training(
        db: Session,
        appointment: AppointmentPrediction,
        noshow_yaml_config: Dict[str, Any],
    ) -> bool:
        """
        Run the feedback appointment through the noshow_lib FE pipeline
        and persist the engineered features to appointment_training_data.

        Only applicable for final statuses: Realizado or Falta.
        Returns True if data was saved, False otherwise.
        """
        status = (appointment.appointment_status or "").strip().lower()
        if status not in {"realizado", "falta"}:
            logger.info(
                f"Skipping training pipeline for appointment {appointment.appointment_prediction_id}: "
                f"status '{appointment.appointment_status}' is not a trainable outcome."
            )
            return False

        logger.info(
            f"Running FE pipeline for appointment {appointment.appointment_prediction_id} "
            f"(status={appointment.appointment_status})"
        )

        # Build a single-row DataFrame using the English column names already
        # stored in appointment_predictions.  The _rename_columns step in
        # build_features is a no-op for already-English columns.
        row = {
            "appointment_id": appointment.appointment_prediction_id,
            "patient_id": appointment.patient_id,
            "appointment_status": appointment.appointment_status,
            "scheduled_at": appointment.scheduled_at,
            "appointment_at": appointment.appointment_at,
            "patient_age": appointment.patient_age,
            "patient_sex": appointment.patient_sex,
            "patient_city": appointment.patient_city,
            "patient_neighborhood": appointment.patient_neighborhood,
            "insurance_type": appointment.insurance_type,
            "unit_name": appointment.unit_name,
            "unit_address": appointment.unit_address,
            "unit_cep": appointment.unit_zipcode,
            "specialty": appointment.specialty,
        }
        raw_df = pd.DataFrame([row])

        try:
            features_df = build_features(raw_df, noshow_yaml_config)
        except Exception as e:
            logger.error(
                f"Feature engineering failed for appointment "
                f"{appointment.appointment_prediction_id}: {e}",
                exc_info=True,
            )
            return False

        if features_df.empty:
            logger.warning(
                f"Feature engineering returned empty DataFrame for appointment "
                f"{appointment.appointment_prediction_id}. Skipping save."
            )
            return False

        # Map feature column names → DB column names (drop unmapped columns)
        available = {
            src: dest
            for src, dest in FEATURES_TO_TRAINING_MAP.items()
            if src in features_df.columns
        }
        db_df = features_df[list(available.keys())].rename(columns=available).copy()
        db_df["raw_appointment_id"] = None
        db_df["source"] = "prediction"
        db_df["created_at"] = datetime.now()

        schema = config.DB_SCHEMA
        table_name = config.DB_TABLE_TRAINING_DATA
        full_table = f"{schema}.{table_name}"

        num_cols = len(db_df.columns)
        safe_chunksize = max(1, (_ODBC_MAX_PARAMS - 1) // num_cols)

        engine = db.get_bind()
        db_df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=safe_chunksize,
        )

        logger.info(
            f"Saved {len(db_df)} engineered row(s) to {full_table} for "
            f"appointment {appointment.appointment_prediction_id}"
        )
        return True
