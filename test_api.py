"""
Unit and integration tests for ML Inference API.
"""
import pytest
from fastapi import status
from app.schemas import PredictionRequest


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_check_model_loaded(self, client, model_loader):
        """Test health check when model is loaded."""
        assert model_loader.is_loaded()
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "health" in data["endpoints"]
        assert "predict" in data["endpoints"]


class TestPredictEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_valid_input(self, client, model_loader, valid_features):
        """Test prediction with valid input."""
        assert model_loader.is_loaded()
        
        payload = {
            "features": valid_features,
            "model_version": "latest"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "input_features" in data
        assert data["input_features"] == valid_features
        assert isinstance(data["prediction"], (int, float))
    
    def test_predict_invalid_features_count(self, client, model_loader, invalid_features):
        """Test prediction with invalid number of features."""
        assert model_loader.is_loaded()
        
        payload = {
            "features": invalid_features,
            "model_version": "latest"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_empty_features(self, client, model_loader):
        """Test prediction with empty features list."""
        assert model_loader.is_loaded()
        
        payload = {
            "features": [],
            "model_version": "latest"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_features(self, client, model_loader):
        """Test prediction with missing features field."""
        assert model_loader.is_loaded()
        
        payload = {
            "model_version": "latest"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_response_headers(self, client, model_loader, valid_features):
        """Test that response includes custom headers."""
        assert model_loader.is_loaded()
        
        payload = {
            "features": valid_features,
            "model_version": "latest"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_200_OK
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers


class TestPredictionRequest:
    """Tests for prediction request validation."""
    
    def test_valid_prediction_request(self, valid_features):
        """Test valid prediction request creation."""
        request = PredictionRequest(
            features=valid_features,
            model_version="latest"
        )
        assert request.features == valid_features
        assert request.model_version == "latest"
    
    def test_prediction_request_default_version(self, valid_features):
        """Test prediction request with default model version."""
        request = PredictionRequest(features=valid_features)
        assert request.model_version == "latest"
    
    def test_invalid_prediction_request_features_count(self, invalid_features):
        """Test invalid prediction request with wrong feature count."""
        with pytest.raises(ValueError):
            PredictionRequest(features=invalid_features)
    
    def test_invalid_prediction_request_empty(self):
        """Test invalid prediction request with empty features."""
        with pytest.raises(ValueError):
            PredictionRequest(features=[])


class TestModelLoader:
    """Tests for model loader functionality."""
    
    def test_model_loader_singleton(self):
        """Test that ModelLoader is a singleton."""
        from app.model_loader import ModelLoader
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        assert loader1 is loader2
    
    def test_demo_model_creation(self):
        """Test creating a demo model."""
        from app.model_loader import ModelLoader
        ModelLoader._instance = None  # Reset singleton
        
        loader = ModelLoader()
        assert loader.create_demo_model()
        assert loader.is_loaded()
        assert loader.get_model_version() == "demo"
    
    def test_model_prediction(self, model_loader, valid_features):
        """Test model prediction functionality."""
        result = model_loader.predict(valid_features)
        assert result is not None
        assert "prediction" in result
        assert "probability" in result
        assert "model_version" in result
    
    def test_model_not_loaded_prediction(self):
        """Test prediction when model is not loaded."""
        from app.model_loader import ModelLoader
        ModelLoader._instance = None  # Reset singleton
        
        loader = ModelLoader()
        result = loader.predict([1.0, 2.0, 3.0, 4.0])
        assert result is None
    
    def test_model_version_setter(self, model_loader):
        """Test setting model version."""
        model_loader.set_model_version("v2.0.0")
        assert model_loader.get_model_version() == "v2.0.0"


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_non_existent_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_predict_without_model(self, client):
        """Test prediction when model is not loaded."""
        from app.model_loader import ModelLoader
        ModelLoader._instance = None  # Reset singleton to unloaded state
        
        payload = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        
        response = client.post("/predict", json=payload)
        # Should return 503 since model is not loaded
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_non_numeric_features(self, client, model_loader):
        """Test prediction with non-numeric features."""
        assert model_loader.is_loaded()
        
        payload = {
            "features": ["a", "b", "c", "d"]
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
