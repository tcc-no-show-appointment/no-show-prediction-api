
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from fastapi.testclient import TestClient
from app.main import app

# client = TestClient(app)


# def test_successful_prediction():
#     """Test real end-to-end prediction with valid input using Portuguese column names"""
#     request_data = {
#        "id": 5642903,
#        "Marcacao": "2024-11-16T08:00:00",
#        "Status": "Realizado",
#        "DataHoraConsulta": "2024-11-23T14:00:00",
#        "Idade": 62,
#        "Sexo": "F",
#        "CidadePaciente": "SAO PAULO",
#        "BairroPaciente": "BELA VISTA",
#        "TipoConvenio": "Enfermaria",
#        "idUnicoPaciente": "ID369425000",
#        "UnidadeAtendimento": "CAMPO BELO",
#        "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
#        "CEPUnidadeAtendimento": "04617-015",
#        "Especialidade": "CARDIOLOGIA"
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
#     request_data = {"Idade": 45}
    
#     response = client.post("/predict", json=request_data)
#     assert response.status_code == 422
