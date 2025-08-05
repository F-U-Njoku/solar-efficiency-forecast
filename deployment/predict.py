import os
import pandas as pd
import joblib
import boto3
from flask import Flask, request, jsonify

RUN_ID = os.getenv("RUN_ID", "649fb2c8074f473c91c885b02a323c6e")
S3_BUCKET = os.getenv("S3_BUCKET", "solarefficiency")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "solar-experiment")

# Download pickle file from S3
s3_client = boto3.client('s3')
local_path = '/tmp/ridge_model.pkl'

s3_client.download_file(
    S3_BUCKET,
    f'mlflow-artifacts/{EXPERIMENT_NAME}/{RUN_ID}/artifacts/pipeline/Ridge_pipeline.pkl',
    local_path
)

# Load the pickle file
model = joblib.load(local_path)
print("✅ Ridge model loaded successfully!")


def prepare_features(solar):
    solar = pd.DataFrame([solar])

    change_dtype = ["humidity", "wind_speed", "pressure"]
    for col in change_dtype:
        solar[col] = pd.to_numeric(solar[col], errors='coerce')

    return solar


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask("solar-efficiency-prediction")


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    solar = request.get_json()

    features = prepare_features(solar)
    pred = predict(features)

    result = {
        'efficiency': pred,
        'model_version': RUN_ID,
        'model_type': 'Ridge'
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
