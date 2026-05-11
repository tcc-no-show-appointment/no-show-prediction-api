from sqlalchemy import Column, Integer, String, Float, DateTime, SmallInteger
from datetime import datetime
from app.database import Base
from app.config import config


class AppointmentPrediction(Base):
    """Maps to appointment_predictions table - stores predictions and user feedback."""
    __tablename__ = config.DB_TABLE_APPOINTMENTS
    __table_args__ = {"schema": config.DB_SCHEMA}

    appointment_prediction_id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=True)
    
    # Context columns
    patient_id = Column(String(100), nullable=True)
    appointment_status = Column(String(50), nullable=True) #Realizado, Falta e Cancelado
    scheduled_at = Column(DateTime, nullable=True)
    appointment_at = Column(DateTime, nullable=True)
    patient_age = Column(Integer, nullable=True)
    patient_sex = Column(String(10), nullable=True)
    patient_city = Column(String(150), nullable=True)
    patient_neighborhood = Column(String(150), nullable=True)
    insurance_type = Column(String(100), nullable=True)
    unit_name = Column(String(150), nullable=True)
    unit_address = Column(String(255), nullable=True)
    unit_zipcode = Column(String(20), nullable=True)
    specialty = Column(String(150), nullable=True)
    
    # Model output columns
    prediction_class = Column(Integer, nullable=True)
    prediction_label = Column(String(20), nullable=True)
    probability_show = Column(Float, nullable=True)
    probability_no_show = Column(Float, nullable=True)
    probability_no_show_normalized = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)
