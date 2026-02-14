from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    Uses Portuguese column names matching config.yaml required_columns.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 5642903,
                "Status": "Realizado",
                "Marcacao": "2024-11-16T08:00:00",
                "DataHoraConsulta": "2024-11-23T14:00:00",
                "Idade": 62,
                "Sexo": "F",
                "CidadePaciente": "SAO PAULO",
                "BairroPaciente": "BELA VISTA",
                "TipoConvenio": "Enfermaria",
                "idUnicoPaciente": "ID369425000",
                "UnidadeAtendimento": "CAMPO BELO",
                "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                "CEPUnidadeAtendimento": "04617-015",
                "Especialidade": "CARDIOLOGIA"
            }
        }
    )
    
    # Required columns based on config.yaml schema
    id: Optional[int] = Field(None, description="Appointment ID")
    Status: Optional[str] = Field(default="Realizado", description="Appointment status (Realizado/Falta)")
    Marcacao: str = Field(..., description="Scheduled date and time")
    DataHoraConsulta: str = Field(..., description="Appointment date and time")
    Idade: int = Field(..., ge=0, le=120, description="Patient age")
    Sexo: str = Field(..., description="Patient gender (M/F)")
    CidadePaciente: str = Field(..., description="Patient city")
    BairroPaciente: str = Field(..., description="Patient neighborhood")
    TipoConvenio: str = Field(..., description="Insurance type")
    idUnicoPaciente: Optional[str] = Field(None, description="Unique patient ID")
    UnidadeAtendimento: str = Field(..., description="Healthcare unit name")
    EnderecoUnidadeAtendimento: str = Field(..., description="Healthcare unit address")
    CEPUnidadeAtendimento: str = Field(..., description="Healthcare unit postal code")
    Especialidade: str = Field(..., description="Medical specialty")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }
    )
    
    prediction: int
    prediction_label: str
    probability_show: float
    probability_no_show: float


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
