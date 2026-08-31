# Support Ticket Management

This project builds a customer support ticket classification system using text features from ticket subject and description. The workflow includes data validation, feature engineering, model training, MLflow experiment tracking, FastAPI inference API, monitoring, drift detection, and Docker packaging.

## Project Overview

The system:

- reads and validates support ticket data from `data/customer_support_tickets.csv`
- stores a SQLite feature table at `data/customer_support_tickets_features.db`
- creates a combined text feature using subject + description
- cleans the text using regex and punctuation removal
- trains multiple classifiers with TF-IDF vectorization
- logs the experiment runs to MLflow
- saves the best model and vectorizer as `support_model.pkl` and `support_vectorizer.pkl`
- exposes predictions through a FastAPI app
- logs predictions for monitoring and drift checks
- can be containerized with Docker

## Project Structure

- `training/train_model.py` - main training and API code
- `training/Dockerfile` - Docker image definition
- `requirements.txt` - Python dependencies for reproducibility
- `feature.json` - feature metadata and model contract
- `data/customer_support_tickets.csv` - source ticket dataset
- `data/customer_support_tickets_features.db` - SQLite version of the feature dataset
- `data/prediction_logs.csv` - prediction log file generated at runtime
- `mlruns/` - MLflow experiment artifacts
- `.dockerignore` - files excluded from Docker build context

## Environment Setup

The project was validated with Python 3.11. A Python 3.13/3.15 environment caused SciPy/scikit-learn import issues on Windows, so the recommended runtime is Python 3.11.

Create and activate a virtual environment:

```powershell
cd C:\Users\THIYA\support_ticket_management
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Install Dependencies

Install the project requirements:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The project dependencies include:

- fastapi
- uvicorn
- pandas
- scikit-learn
- mlflow
- joblib
- pydantic
- tomlkit

## Train the Model

From the project root:

```powershell
python training\train_model.py
```

This script performs the full pipeline:

1. checks the CSV dataset exists
2. validates expected columns
3. removes duplicates
4. builds the text feature from ticket subject + description
5. cleans text data
6. splits train/test data
7. trains multiple models
8. compares model performance
9. saves the best model and vectorizer
10. starts the FastAPI app object definition

## Run the API

Start the FastAPI app in a second terminal:

```powershell
uvicorn training.train_model:app --reload
```

Then open the interactive docs:

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- ReDoc: http://127.0.0.1:8000/redoc

### Example request

```json
{
  "subject": "Payment problem",
  "description": "My payment was deducted twice."
}
```

### Example response

```json
{
  "prediction": "Billing"
}
```

## API Endpoints

The app includes:

- `POST /predict` - predicts the ticket category
- `POST /predict_with_logging` - predicts and writes a prediction log entry
- `POST /drift_score` - compares current predictions with the training distribution

## MLflow Tracking

During training, MLflow is configured with the experiment name:

```python
EXPERIMENT_NAME = "CustomerSupportClassifier"
```

To view tracking logs locally:

```powershell
mlflow ui
```

Then open:

- http://127.0.0.1:5000

## Monitoring and Drift Detection

The project logs prediction data to `data/prediction_logs.csv` and `monitoring_log.csv` to support:

- monitoring prediction behavior
- tracking drift between new predictions and training distribution
- retraining decisions when the distribution changes beyond a threshold

The drift logic is implemented in the `drift_score` endpoint and compares prediction frequencies with the original training label distribution.

## Docker Setup

A Dockerfile is included in `training/Dockerfile`.

Build the image:

```powershell
docker build -f training\Dockerfile -t support-ticket-api .
```

Run the container:

```powershell
docker run -p 8000:8000 support-ticket-api
```

Then open:

- http://localhost:8000/docs

You can also push the image to Docker Hub:

```powershell
docker login
docker tag support-ticket-api yourusername/support-ticket-api:latest
docker push yourusername/support-ticket-api:latest
```

## Feature Metadata

The project includes a feature contract in `feature.json` that documents the expected features, preprocessing steps, and model metadata. This helps define the feature pipeline used during training and inference.

## Important Notes

- The project expects the CSV file at `data/customer_support_tickets.csv`.
- The app writes SQLite and log files under the `data/` folder.
- `mlruns/` contains experiment artifacts produced by MLflow.
- Docker builds should exclude local environment folders such as `.venv`, `mlruns`, and generated logs using `.dockerignore`.
- Python 3.11 is the recommended environment for compatibility with the training stack and Docker image.

## Quick Start Summary

```powershell
cd C:\Users\THIYA\support_ticket_management
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training\train_model.py
uvicorn training.train_model:app --reload
```

Then visit:

- http://127.0.0.1:8000/docs

## Conclusion

This project demonstrates a complete end-to-end ML pipeline for support ticket classification: data preparation, feature engineering, model selection, tracking, serving, monitoring, drift detection, and deployment-ready Docker packaging.