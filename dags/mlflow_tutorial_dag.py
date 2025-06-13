"""
### Show three ways to use MLFlow with Airflow

This DAG shows how you can use the MLflowClientHook to create an experiment in MLFlow,
directly log metrics and parameters to MLFlow in a TaskFlow task via mlflow.sklearn.autolog() and
create a new model

"""

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from astro.dataframes.pandas import DataFrame
from mlflow_provider.hooks.client import MLflowClientHook
from mlflow_provider.operators.registry import CreateRegisteredModelOperator
from pendulum import datetime
from sklearn.pipeline import make_pipeline

## MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
MINIO_CONN_ID = "minio_local"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 100
EXPERIMENT_NAME = "taxi_trip_duration"
REGISTERED_MODEL_NAME = "lin_reg_predictor"
ARTIFACT_BUCKET = "mlflowdata"

TRAINDATA_FILENAME = "/usr/local/airflow/include/data/yellow_tripdata_2023-03.parquet"


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    # tags=["mlops_zoomcamp"],
    params={"filename": TRAINDATA_FILENAME},
)
def training_pipeline():
    @task
    def check_if_trainset_exists(**context) -> str:
        import os

        filename = context["params"]["filename"]
        print(f"Training filename to process {filename}")

        if not filename or not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        return

    @task
    def prepare_training_dataset(**context) -> tuple[str, str]:
        from include.tools.preprocessing import (
            get_num_rows,
            prepare_dataset,
            read_dataset,
        )

        filename = context["params"]["filename"]
        lzdf_raw = read_dataset(filename, mode="lazy")

        print(f"Training data size {get_num_rows(lzdf_raw)}")

        df_trainset, df_target = prepare_dataset(lzdf_raw)

        trainset_path = "/tmp/trainset.parquet"
        df_trainset.write_parquet(trainset_path)

        target_path = "/tmp/target.parquet"
        df_target.write_parquet(target_path)

        print(f"Training data size after cleaning {df_trainset.shape}")

        return trainset_path, target_path

    @task
    def transform_training_dataset(file_paths):
        from include.tools.feature_engineering import fit_transform_dictvect
        from include.tools.preprocessing import dump_pickle, read_dataset

        trainset_path, target_path = file_paths
        df_trainset = read_dataset(trainset_path)
        df_target = read_dataset(target_path)

        x_train, dv = fit_transform_dictvect(df_trainset)
        y_train = df_target.to_numpy().ravel()

        train_bundle_path = "/tmp/train_bundle.pkl"
        dump_pickle((x_train, y_train), train_bundle_path)

        dictvect_path = "/tmp/dictvect.pkl"
        dump_pickle(dv, dictvect_path)

        return train_bundle_path, dictvect_path

    create_buckets_if_not_exists = S3CreateBucketOperator(
        task_id="create_buckets_if_not_exists",
        aws_conn_id=MINIO_CONN_ID,
        bucket_name=ARTIFACT_BUCKET,
    )

    # 1. Use a hook from the MLFlow provider to interact with MLFlow within a TaskFlow task
    @task
    def create_mlflow_experiment(experiment_name, artifact_bucket, **context):
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""

        timestamp = context["ts"]

        mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        new_experiment_information = mlflow_hook.run(
            endpoint="api/2.0/mlflow/experiments/create",
            request_params={
                "name": experiment_name + "_" + timestamp,
                "artifact_location": f"s3://{artifact_bucket}/",
            },
        ).json()

        return new_experiment_information["experiment_id"]

    # 2. Use a mlflow.sklearn autologging in a TaskFlow task
    @task
    def train_model(experiment_id: str, train_files: list[str]) -> DataFrame:
        """Track feature scaling by sklearn in Mlflow."""
        import mlflow
        from sklearn.linear_model import LinearRegression

        from include.tools.preprocessing import load_pickle

        train_bundle_path, dictvect_path = train_files
        x_train, y_train = load_pickle(train_bundle_path)
        dictvect = load_pickle(dictvect_path)
        lin_reg_model = LinearRegression()

        pipeline = make_pipeline(dictvect, lin_reg_model)

        mlflow.sklearn.autolog(log_datasets=False, log_models=False)
        with mlflow.start_run(experiment_id=experiment_id, run_name="lin_reg") as run:
            # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            # mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            lin_reg_model.fit(x_train, y_train)
            mlflow.log_metrics(
                {
                    "intercept_": lin_reg_model.intercept_,
                    "model_size": lin_reg_model.__sizeof__(),
                }
            )
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return

    # 3. Use an operator from the MLFlow provider to interact with MLFlow directly
    create_registered_model = CreateRegisteredModelOperator(
        task_id="create_registered_model",
        name=REGISTERED_MODEL_NAME + "_" + "{{ ts }}",
        tags=[
            {"key": "model_type", "value": "regression"},
            {"key": "experiment_name", "value": EXPERIMENT_NAME},
        ],
    )

    # DAG structure definition
    # check = check_if_trainset_exists()
    temp_files_path = prepare_training_dataset()
    train_path = transform_training_dataset(temp_files_path)
    experiment_id = create_mlflow_experiment(EXPERIMENT_NAME, ARTIFACT_BUCKET)

    (
        check_if_trainset_exists()
        >> temp_files_path
        >> train_path
        >> create_buckets_if_not_exists
        >> experiment_id
        >> train_model(experiment_id, train_path)
        >> create_registered_model
    )


training_pipeline()
