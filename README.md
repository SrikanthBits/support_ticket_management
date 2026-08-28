# Support Ticket Management

## Try the API in Swagger

The FastAPI application exposes an interactive Swagger UI. From the project
root, run the training script to create the model files:

```powershell
pip install fastapi uvicorn
python training/train_model.py
```

In a second terminal, start the API with Uvicorn:

```powershell
uvicorn training.train_model:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand
`POST /predict`, and choose **Try it out**. Swagger provides this example
request:

```json
{
	"subject": "Payment problem",
	"description": "My payment was deducted twice."
}
```

Select **Execute**. A successful response has this shape:

```json
{
	"prediction": "Billing"
}
```

FastAPI also exposes the generated OpenAPI document at
[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) and
alternative documentation at [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).