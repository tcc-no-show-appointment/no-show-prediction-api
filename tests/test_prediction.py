
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from fastapi.testclient import TestClient
from app.main import app

# client = TestClient(app)


# def test_successful_prediction():
#     """Test real end-to-end prediction with valid input"""
#     request_data = {
#         "Age": 62,
#         "Gender": "F",
#         "ScheduledDay": "2024-11-16T08:00:00",
#         "AppointmentDay": "2024-11-23T14:00:00",
#         "Neighbourhood": "JARDIM CAMBURI",
#         "Scholarship": 0,
#         "Hipertension": 1,
#         "Diabetes": 0,
#         "Alcoholism": 0,
#         "Handcap": 0,
#         "SMS_received": 1
#     }
    
#     response = client.post("/predict", json=request_data)
    
#     assert response.status_code == 200
#     data = response.json()
#     assert data["prediction"] in [0, 1]
#     assert data["prediction_label"] in ["Show", "No-show"]
#     assert 0 <= data["probability_show"] <= 1
#     assert 0 <= data["probability_no_show"] <= 1
#     assert abs(data["probability_show"] + data["probability_no_show"] - 1.0) < 0.01


# def test_prediction_validation_error():
#     """Test validation error with missing required fields"""
#     request_data = {"Age": 45}
    
#     response = client.post("/predict", json=request_data)
#     assert response.status_code == 422
