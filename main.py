from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import os
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import __main__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_grid_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "noshowappointments.csv")
HARDCODED_PATIENT_ID = 622257462899397

app = FastAPI()

df_appointments = pd.read_csv(CSV_PATH)

def to_float32(X):
    return X.astype(np.float32)

__main__.to_float32 = to_float32

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def extract_features_from_row(row):
    scheduled_day = pd.to_datetime(row['ScheduledDay'])
    appointment_day = pd.to_datetime(row['AppointmentDay'])
    
    waiting_days = (appointment_day - scheduled_day).days
    
    scheduled_hour = scheduled_day.hour
    
    appointment_weekday = appointment_day.weekday()
    
    bins = [-1, 0, 3, 7, 14, 10**9]
    labels = ["wait_0", "wait_1_3", "wait_4_7", "wait_8_14", "wait_15p"]
    waiting_days_bucket = str(pd.cut([waiting_days], bins=bins, labels=labels)[0])
    
    is_weekend = 1 if appointment_weekday >= 5 else 0
    
    day_part_morning = 1 if 6 <= scheduled_hour < 12 else 0
    day_part_afternoon = 1 if 12 <= scheduled_hour < 18 else 0
    day_part_evening = 1 if scheduled_hour >= 18 or scheduled_hour < 6 else 0
    
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

async def predict(patient_id):
    matches = df_appointments[df_appointments['PatientId'] == patient_id]
    if matches.empty:
        raise ValueError(f"PatientId {patient_id} not found in the dataset.")
    row = matches.iloc[0]
    row_index = matches.index[0]
    features = extract_features_from_row(row)
    df = pd.DataFrame(features)
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return {
        "row_index": row_index,
        "patient_id": row['PatientId'],
        "appointment_id": row['AppointmentID'],
        "actual_outcome": row['No-show'],
        "input_features": features,
        "prediction": int(prediction[0]),
        "prediction_label": "No-show" if prediction[0] == 1 else "Show",
        "probability_show": float(probability[0][0]),
        "probability_no_show": float(probability[0][1])
    }

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()
    return data

@app.get("/predict")
async def get_prediction():
    try:
        result = await predict(HARDCODED_PATIENT_ID)
        result = convert_to_native_types(result)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
