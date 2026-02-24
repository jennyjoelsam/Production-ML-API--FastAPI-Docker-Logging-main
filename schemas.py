"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    
    features: List[float] = Field(
        ..., 
        description="List of numerical features for prediction",
        examples=[[1.0, 2.0, 3.0, 4.0]]
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction"
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features list cannot be empty")
        if len(v) != 4:
            raise ValueError("Expected exactly 4 features")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    prediction: float = Field(..., description="Predicted value")
    probability: Optional[float] = Field(
        default=None,
        description="Confidence score or probability"
    )
    model_version: str = Field(..., description="Model version used")
    input_features: List[float] = Field(..., description="Features used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0.0,
                "probability": 0.95,
                "model_version": "v1.0.0",
                "input_features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(default=None, description="Loaded model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    detail: str = Field(..., description="Error description")
    error_code: str = Field(..., description="Error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid input features",
                "error_code": "VALIDATION_ERROR"
            }
        }
