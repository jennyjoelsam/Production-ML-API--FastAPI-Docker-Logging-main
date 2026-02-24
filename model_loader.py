"""
Model loader module for handling ML model loading and caching.
Supports both scikit-learn and PyTorch models.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and cache ML models for inference."""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[Any] = None
    _model_version: str = "v1.0.0"
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_sklearn_model(self, model_path: str) -> bool:
        """
        Load a scikit-learn model from pickle file.
        
        Args:
            model_path: Path to the pickle file
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            logger.info(f"Successfully loaded sklearn model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {str(e)}")
            return False
    
    def load_pytorch_model(self, model_path: str, model_class: type) -> bool:
        """
        Load a PyTorch model.
        
        Args:
            model_path: Path to the model checkpoint
            model_class: The model class to instantiate
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            import torch
            
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = model_class()
            self._model.load_state_dict(torch.load(model_path, map_location=device))
            self._model.to(device)
            self._model.eval()
            
            logger.info(f"Successfully loaded PyTorch model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            return False
    
    def create_demo_model(self) -> bool:
        """
        Create a demo sklearn model for testing.
        Uses a simple iris classifier.
        
        Returns:
            True if model created successfully
        """
        try:
            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier
            
            # Load iris dataset and train a simple model
            iris = load_iris()
            X, y = iris.data, iris.target
            
            model = RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            self._model = model
            self._model_version = "demo"
            
            logger.info("Created demo sklearn RandomForest model")
            return True
        except Exception as e:
            logger.error(f"Failed to create demo model: {str(e)}")
            return False
    
    def predict(self, features: list) -> Optional[dict]:
        """
        Make prediction using loaded model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Dictionary with prediction and metadata, or None if error
        """
        try:
            if self._model is None:
                logger.error("No model loaded")
                return None
            
            # Convert to numpy array
            X = np.array([features])
            
            # Check if it's a sklearn model
            if hasattr(self._model, 'predict'):
                prediction = self._model.predict(X)[0]
                
                # Try to get probability if available
                probability = None
                if hasattr(self._model, 'predict_proba'):
                    proba = self._model.predict_proba(X)[0]
                    probability = float(np.max(proba))
                
                return {
                    'prediction': float(prediction),
                    'probability': probability,
                    'model_version': self._model_version
                }
            else:
                logger.error("Model does not have predict method")
                return None
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def get_model_version(self) -> str:
        """Get current model version."""
        return self._model_version
    
    def set_model_version(self, version: str) -> None:
        """Set model version string."""
        self._model_version = version
        logger.info(f"Model version set to {version}")
