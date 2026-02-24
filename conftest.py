"""
Pytest configuration and fixtures for ML API tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import ModelLoader


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    """Setup before all tests."""
    # Reset model loader singleton for testing
    ModelLoader._instance = None
    yield
    # Cleanup after all tests
    ModelLoader._instance = None


@pytest.fixture
def client():
    """Provide FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def model_loader():
    """Provide model loader instance with demo model."""
    loader = ModelLoader()
    if not loader.is_loaded():
        loader.create_demo_model()
    return loader


@pytest.fixture
def valid_features():
    """Provide valid input features for tests."""
    return [5.1, 3.5, 1.4, 0.2]


@pytest.fixture
def invalid_features():
    """Provide invalid input features for tests."""
    return [1.0, 2.0]  # Only 2 features instead of 4
