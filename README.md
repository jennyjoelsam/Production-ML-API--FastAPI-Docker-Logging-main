# Production ML Inference API

A production-ready ML inference API built with **FastAPI**, **Docker**, and **Comprehensive Logging**. This project demonstrates best practices for deploying machine learning models in a scalable, maintainable, and observable manner.

## 🚀 Features

- **FastAPI Framework**: Modern, fast, and easy-to-use Python web framework
- **Model Support**: Supports both scikit-learn and PyTorch models
- **Request Validation**: Pydantic-based input validation with detailed error messages
- **Comprehensive Logging**: Request/response logging with rotation and structured formatting
- **Health Checks**: Built-in health endpoint for orchestration platforms
- **Error Handling**: Graceful error handling with meaningful error codes
- **Docker Support**: Multi-stage Dockerfile for optimized image size
- **CI/CD Pipeline**: GitHub Actions for automated testing and building
- **Test Coverage**: Comprehensive pytest test suite with fixtures
- **API Documentation**: Auto-generated interactive API docs (Swagger UI)

## 📋 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── model_loader.py         # Model loading and caching (singleton)
│   └── schemas.py              # Pydantic request/response models
├── tests/
│   ├── conftest.py             # Pytest configuration and fixtures
│   └── test_api.py             # Comprehensive API tests
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI configuration
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker ignore rules
└── README.md                   # This file
```

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Production ML API (FastAPI + Docker + Logging)"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t ml-inference-api:latest .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 ml-inference-api:latest
   ```

3. **With custom model path**
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_PATH="/models/my_model.pkl" \
     -v /path/to/models:/models \
     ml-inference-api:latest
   ```

## 📚 API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.0.0"
}
```

### Make Prediction

```http
POST /predict
Content-Type: application/json

{
  "features": [5.1, 3.5, 1.4, 0.2],
  "model_version": "latest"
}
```

**Response:**
```json
{
  "prediction": 0.0,
  "probability": 0.95,
  "model_version": "v1.0.0",
  "input_features": [5.1, 3.5, 1.4, 0.2]
}
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Run Specific Test Class

```bash
pytest tests/test_api.py::TestPredictEndpoint -v
```

### Run Tests with Output

```bash
pytest -v -s
```

## 🔍 Logging

The application includes comprehensive logging for:

- **Application Startup/Shutdown**: Logs when the API starts and stops
- **Request/Response**: Logs all incoming requests with request IDs and response times
- **Model Loading**: Logs model loading success/failure
- **Predictions**: Logs prediction requests and results
- **Errors**: Detailed error logging with stack traces

### Log Files

Logs are written to `logs/api.log` with:
- **Rotation**: 10MB max file size with 5 backup files
- **Format**: Timestamp, logger name, level, message, and file location

### Custom Request Headers

Each request includes:
- `X-Request-ID`: Unique request identifier for tracing
- `X-Process-Time`: Response processing time in seconds

## 🤖 Model Loading

The application supports two model loading approaches:

### Load Existing Model

Set the `MODEL_PATH` environment variable:
```bash
export MODEL_PATH="/path/to/model.pkl"
uvicorn app.main:app --reload
```

### Use Demo Model

If no model path is provided, a demo Iris classification model is created automatically for testing.

### Model Loader (Singleton)

The `ModelLoader` class uses the singleton pattern to ensure only one model is loaded in memory:

```python
from app.model_loader import ModelLoader

loader = ModelLoader()
loader.create_demo_model()
result = loader.predict([5.1, 3.5, 1.4, 0.2])
```

## 📦 Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **scikit-learn**: Machine learning library
- **torch**: Deep learning framework
- **numpy, pandas**: Data processing
- **pytest**: Testing framework
- **python-dotenv**: Environment variable management

See `requirements.txt` for complete list with versions.

## 🔐 Security Considerations

- **Non-root User**: Container runs as unprivileged user (appuser)
- **Input Validation**: All inputs validated with Pydantic
- **CORS Support**: Configurable for production environments
- **Error Messages**: Avoid leaking sensitive information in errors
- **Logging**: Sensitive data should be excluded from logs

## 🚀 Deployment

### Docker Compose (if scaling needed)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/iris_model.pkl
    volumes:
      - ./models:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference-api
  template:
    metadata:
      labels:
        app: ml-inference-api
    spec:
      containers:
      - name: api
        image: ml-inference-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /models/iris_model.pkl
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📊 CI/CD Pipeline

The GitHub Actions workflow (`ci.yml`) includes:

1. **Testing**: Runs on Python 3.9, 3.10, 3.11
2. **Code Quality**: Linting with flake8 and type checking with mypy
3. **Coverage**: Generates coverage reports
4. **Docker Build**: Builds multi-stage Docker image
5. **Security**: Trivy vulnerability scanning

## 🐛 Common Issues

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

### Import Errors

Ensure you're in the virtual environment and have installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Model Not Loading

Check that:
1. `MODEL_PATH` environment variable is set correctly
2. Model file exists and is readable
3. Check logs: `tail -f logs/api.log`

## 🤝 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
payload = {
    "features": [5.1, 3.5, 1.4, 0.2],
    "model_version": "latest"
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

**Happy Inferencing! 🎯**

