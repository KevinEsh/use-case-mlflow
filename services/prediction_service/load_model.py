import mlflow

# MLFLOW_TRACKING_URI = "http://172.17.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model_from_mlflow(run_id: str):
    """
    Load a model from MLflow using the provided run ID.

    Args:
        run_id (str): The MLflow run ID of the model to load.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded MLflow model.
    """
    mlflow_model_uri = (
        f"s3://mlflowdata/{run_id}/artifacts/model"  # f"runs:/{run_id}/model"
    )
    return mlflow.pyfunc.load_model(mlflow_model_uri)
