import os
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI

MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")

model = mlflow.pyfunc.load_model(
    f"models:/fraud_detection_model/{MODEL_STAGE}"
)

app = FastAPI()

@app.post("/predict")
def predict(features: list[float]):
    x = np.array(features).reshape(1, -1)
    p = model.predict(x)[0]
    return {"prediction": float(p)}
