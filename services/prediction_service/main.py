# import numpy as np

# from mlflow.sklearn import load_model
from fastapi import FastAPI, HTTPException, Query
from load_model import (
    load_model_from_mlflow,  # , load_dict_vectorizer_from_mlflow
)
from pydantic import BaseModel

RUN_ID = "169c869297b34d1e87976d919426b8ba"  # Example run ID
model = load_model_from_mlflow(RUN_ID)  # Example run ID


def transform_request(request):
    import numpy as np

    example_features = np.zeros((1, 518))
    return example_features


api = FastAPI(title="Prediction Service")


class PredictionRequest(BaseModel):
    pickup_location_code: str = Query(default="")
    dropoff_location_code: str = Query(default="")


class PredictionResponse(BaseModel):
    trip_duration_prediction: float


@api.get("/")
def root():
    return {"message": "Prediction service is alive."}


@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = transform_request(request.model_dump())
        prediction = model.predict(features)[0]
        return PredictionResponse(trip_duration_prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
