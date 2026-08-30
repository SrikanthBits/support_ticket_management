FROM python:3.11-slim
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn mlflow joblib pydantic

RUN python training/train_model.py

EXPOSE 8000

#train and serve the model using uvicorn
CMD ["python", "-m", "uvicorn", "training.train_model:app", "--host", "0.0.0.0", "--port", "8000"]