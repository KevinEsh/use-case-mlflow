# import numpy as np

# from mlflow.sklearn import load_model
from fastapi import FastAPI, HTTPException, Query
from load_model import load_model_from_mlflow
from pydantic import BaseModel

# RUN_ID = "169c869297b34d1e87976d919426b8ba"  # Example run ID
# model = load_model_from_mlflow(RUN_ID)  # Example run ID

MODEL = None  # Placeholder for the actual model loading logic

def transform_request(request):
    return request  # Placeholder for actual transformation logic
    # import numpy as np

    # example_features = np.zeros((1, 518))
    # return example_features


api = FastAPI(title="Prediction Service")


class PredictionRequest(BaseModel):
    pu: str = Query(default="")
    do: str = Query(default="")
    trip_distance: float = Query(default=0.0)


class PredictionResponse(BaseModel):
    trip_duration_prediction: float

class BulkPredictionRequest(BaseModel):
    requests: list[PredictionRequest]

#bulk prediction response. should be a dict with "trip_duration_prediction" as key and list of predictions as value
class BulkPredictionResponse(BaseModel):
    trip_duration_prediction: list[float]


@api.get("/")
def root():
    return {"message": "Prediction service is super alive."}

# POST request to configure the model
@api.post("/model")
def configure_model(run_id: str):
    # Placeholder for model loading logic
    global MODEL
    MODEL = load_model_from_mlflow(run_id)
    return {"message": f"Model configured with run ID: {run_id}"}


@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global MODEL

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = transform_request(request.model_dump())
        prediction = MODEL.predict(features)[0]
        return PredictionResponse(trip_duration_prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#bulk prediction endpoint
@api.post("/predict/bulk")
def bulk_predict(request: BulkPredictionRequest):
    global MODEL

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = MODEL.predict(request.model_dump())
        return BulkPredictionResponse(trip_duration_prediction=[float(pred) for pred in predictions])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))