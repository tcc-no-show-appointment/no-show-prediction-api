PREDICTION_LABEL_SHOW = "Show"
PREDICTION_LABEL_NO_SHOW = "No-show"

# Mapping from noshow_lib feature names → appointment_training_data DB column names.
# Kept in sync with data_persistence.py in the training API.
FEATURES_TO_TRAINING_MAP = {
    # Context columns
    "patient_id": "patient_id",
    "appointment_status": "appointment_status",
    "scheduled_at": "scheduled_at",
    "appointment_at": "appointment_at",
    "patient_age": "patient_age",
    "patient_sex": "patient_sex",
    "patient_city": "patient_city",
    "patient_neighborhood": "patient_neighborhood",
    "insurance_type": "insurance_type",
    "unit_name": "unit_name",
    "unit_address": "unit_address",
    "unit_cep": "unit_zipcode",
    "specialty": "specialty",
    "specialty_group": "specialty_group",
    # Target
    "no_show": "no_show",
    # Temporal features
    "waiting_days": "waiting_days",
    "is_same_day": "is_same_day",
    "appointment_weekday": "appointment_weekday",
    "appointment_day_of_month": "appointment_day_of_month",
    "appointment_week_of_month": "appointment_week_of_month",
    "is_month_start": "is_month_start",
    "is_month_end": "is_month_end",
    "hour_appointment": "hour_appointment",
    "time_of_day": "time_of_day",
    "is_weekend": "is_weekend",
    "is_holiday": "is_holiday",
    "is_pre_holiday": "is_pre_holiday",
    "is_post_holiday": "is_post_holiday",
    "is_bridge_day": "is_bridge_day",
    "is_holiday_window": "is_holiday_window",
    "month_sin": "month_sin",
    "month_cos": "month_cos",
    "weekday_sin": "weekday_sin",
    "weekday_cos": "weekday_cos",
    "hour_sin": "hour_sin",
    "hour_cos": "hour_cos",
    # Patient demographic
    "age_group": "age_group",
    # Patient history features
    "has_patient_history": "has_patient_history",
    "previous_appointments_count": "previous_appointments_count",
    "patient_tenure_days": "patient_tenure_days",
    "past_no_shows": "past_no_shows",
    "previous_no_show": "previous_no_show",
    "consecutive_no_shows_2": "consecutive_no_shows_2",
    "past_cancellations_count": "past_cancellations_count",
    "cancellation_rate": "cancellation_rate",
    "no_show_rate_patient": "no_show_rate_patient",
    "no_show_rate_patient_smoothed": "no_show_rate_patient_smoothed",
    "no_show_rate_recent_3": "no_show_rate_recent_3",
    "no_show_rate_recent_5": "no_show_rate_recent_5",
    "days_since_last_visit": "days_since_last_visit",
    "days_since_last_no_show": "days_since_last_no_show",
    "appointments_in_same_schedule_day": "appointments_in_same_schedule_day",
    # Behavioral / interaction features
    "is_diff_specialty": "is_diff_specialty",
    "waiting_days_delta": "waiting_days_delta",
    "age_x_waiting_days": "age_x_waiting_days",
    "no_show_rate_x_waiting_days": "no_show_rate_x_waiting_days",
    "age_x_specialty_risk": "age_x_specialty_risk",
    "gender_age_profile": "gender_age_profile",
    # Contextual rate features
    "specialty_no_show_rate": "specialty_no_show_rate",
    "unit_no_show_rate": "unit_no_show_rate",
    "specialty_group_no_show_rate": "specialty_group_no_show_rate",
    "insurance_no_show_rate": "insurance_no_show_rate",
    "neighborhood_risk_score": "neighborhood_risk_score",
    "specialty_high_no_show_flag": "specialty_high_no_show_flag",
    # Geo features
    "same_city": "same_city",
    "is_local_resident": "is_local_resident",
    "same_cep_prefix5": "same_cep_prefix5",
}

# All specialty_group values produced by noshow_lib feature engineering.
# Used to pre-populate the effective_models dict with the default fallback
# for any specialty that does not yet have a dedicated trained model.
KNOWN_SPECIALTY_GROUPS = [
    "CLINICA_ESPECIALIZADA",
    "CLINICA_GERAL_E_TRIAGEM",
    "CIRURGICO_E_VASCULAR",
    "EXAMES_E_PROCEDIMENTOS",
    "MATERNO_INFANTIL",
    "ORTOPEDIA",
    "SAUDE_MENTAL",
    "TERAPIAS_REABILITACAO",
    "OUTRAS_ESPECIALIDADES",
]