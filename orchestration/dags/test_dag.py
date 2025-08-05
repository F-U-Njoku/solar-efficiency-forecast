from datetime import datetime
from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["test"],
)
def test_dag():
    @task()
    def simple_task():
        print("Hello from Airflow!")
        return "success"

    simple_task()


test_dag()
