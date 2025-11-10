import pandas as pd
import numpy as np
from app.constants import (
    WAITING_DAYS_BINS,
    WAITING_DAYS_LABELS,
    DAY_PART_MORNING_START,
    DAY_PART_MORNING_END,
    DAY_PART_AFTERNOON_END,
    WEEKEND_START
)


def extract_features_from_row(row):
    scheduled_day = pd.to_datetime(row['ScheduledDay'])
    appointment_day = pd.to_datetime(row['AppointmentDay'])
    
    waiting_days = (appointment_day - scheduled_day).days
    
    scheduled_hour = scheduled_day.hour
    
    appointment_weekday = appointment_day.weekday()
    
    waiting_days_bucket = str(pd.cut([waiting_days], bins=WAITING_DAYS_BINS, labels=WAITING_DAYS_LABELS)[0])
    
    is_weekend = 1 if appointment_weekday >= WEEKEND_START else 0
    
    day_part_morning = 1 if DAY_PART_MORNING_START <= scheduled_hour < DAY_PART_MORNING_END else 0
    day_part_afternoon = 1 if DAY_PART_MORNING_END <= scheduled_hour < DAY_PART_AFTERNOON_END else 0
    day_part_evening = 1 if scheduled_hour >= DAY_PART_AFTERNOON_END or scheduled_hour < DAY_PART_MORNING_START else 0
    
    weekday_sin = np.sin(2 * np.pi * appointment_weekday / 7)
    weekday_cos = np.cos(2 * np.pi * appointment_weekday / 7)
    
    wait_x_sms = waiting_days * row['SMS_received']
    
    features = {
        "Gender": [row['Gender']],
        "Age": [row['Age']],
        "Neighbourhood": [row['Neighbourhood']],
        "Scholarship": [row['Scholarship']],
        "Hipertension": [row['Hipertension']],
        "Diabetes": [row['Diabetes']],
        "Alcoholism": [row['Alcoholism']],
        "Handcap": [row['Handcap']],
        "SMS_received": [row['SMS_received']],
        "waiting_days": [waiting_days],
        "scheduled_hour": [scheduled_hour],
        "appointment_weekday": [appointment_weekday],
        "waiting_days_bucket": [waiting_days_bucket],
        "is_weekend": [is_weekend],
        "day_part_morning": [day_part_morning],
        "day_part_afternoon": [day_part_afternoon],
        "day_part_evening": [day_part_evening],
        "weekday_sin": [weekday_sin],
        "weekday_cos": [weekday_cos],
        "wait_x_sms": [wait_x_sms]
    }
    
    return features
