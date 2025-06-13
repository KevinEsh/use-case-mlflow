import requests

params = {"pickup_location_code": "20", "dropoff_location_code": "B"}
response = requests.post(url="http://0.0.0.0:8000/predict", json=params)

print(response.content)

# import mlflow

# MLFLOW_TRACKING_URI = "http://172.17.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# logged_model = "runs:/169c869297b34d1e87976d919426b8ba/model"  # "s3://mlflowdata/169c869297b34d1e87976d919426b8ba/artifacts/model"
# loaded_model = mlflow.pyfunc.load_model(logged_model)


# Predict on a Pandas DataFrame.
# print(loaded_model)
# import pandas as pd
# loaded_model.predict(pd.DataFrame(data))
