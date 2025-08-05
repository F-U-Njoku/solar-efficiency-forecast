import os
from datetime import datetime
import pandas as pd
from airflow.sdk import dag, task
from airflow.models import Variable
from ml_pipeline.train import run_ml_experiments, setup_aws_credentials

DATA_PATH = Variable.get("DATA_PATH")
EXPERIMENT_NAME = Variable.get("EXPERIMENT_NAME")
S3_BUCKET = Variable.get("S3_BUCKET")
AWS_REGION = Variable.get("AWS_REGION")

@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlflow", "mlops", "airflow"],
)
def solar_experiment():

    @task()
    def check_aws():
        setup_aws_credentials()

    @task()
    def run_experiments():
        results, tracker = run_ml_experiments(
            data_path=DATA_PATH,
            experiment_name=EXPERIMENT_NAME,
            s3_bucket=S3_BUCKET,
            aws_region=AWS_REGION
        )
        print(f"✅ Experiment with id {tracker.experiment_id} completed")

    check_aws() >> run_experiments()
solar_experiment()
