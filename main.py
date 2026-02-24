"""
Production ML Inference API using FastAPI.
Provides endpoints for model health checks and predictions with request logging.
"""
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse, HealthCheckResponse, ErrorResponse
from app.model_loader import ModelLoader

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize model loader (singleton)
model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Loads model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting ML Inference API")
    
    # Try to load model from environment variable, otherwise use demo model
    import os
    model_path = os.getenv("MODEL_PATH", None)
    
    if model_path:
        logger.info(f"Loading model from path: {model_path}")
        if not model_loader.load_sklearn_model(model_path):
            logger.warning("Failed to load model from specified path, using demo model")
            model_loader.create_demo_model()
    else:
        logger.info("No model path provided, using demo model")
        model_loader.create_demo_model()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Inference API")


# Create FastAPI app
app = FastAPI(
    title="ML Inference API",
    description="Production-ready ML inference API with request logging and validation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests and responses.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    start_time = datetime.utcnow()
    
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"[{request_id}] Response: {response.status_code} - "
            f"Duration: {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as e:
        process_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            f"[{request_id}] Error processing request: {str(e)} - "
            f"Duration: {process_time:.3f}s"
        )
        raise


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Check API and model health status"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns model loading status and version information.
    """
    logger.info("Health check requested")
    
    return HealthCheckResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        model_loaded=model_loader.is_loaded(),
        model_version=model_loader.get_model_version() if model_loader.is_loaded() else None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Make Prediction",
    description="Get prediction for input features",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid input validation"
        },
        503: {
            "model": ErrorResponse,
            "description": "Model not loaded"
        }
    }
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction using loaded model.
    
    Args:
        request: PredictionRequest with features and optional model version
        
    Returns:
        PredictionResponse with prediction, probability, and metadata
        
    Raises:
        HTTPException: If validation fails or model is not loaded
    """
    # Check if model is loaded
    if not model_loader.is_loaded():
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check health endpoint.",
        )
    
    logger.info(f"Prediction requested with features: {request.features}")
    
    # Make prediction
    result = model_loader.predict(request.features)
    
    if result is None:
        logger.error("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during prediction. Check logs for details.",
        )
    
    response = PredictionResponse(
        prediction=result['prediction'],
        probability=result['probability'],
        model_version=result['model_version'],
        input_features=request.features,
    )
    
    logger.info(
        f"Prediction successful. Prediction: {response.prediction}, "
        f"Probability: {response.probability}"
    )
    
    return response


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Welcome endpoint with API information"
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.
    """
    return {
        "name": "ML Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
        "docs": "See /docs for interactive API documentation",
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "VALIDATION_ERROR"
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error. Check logs for details.",
            "error_code": "INTERNAL_ERROR"
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
