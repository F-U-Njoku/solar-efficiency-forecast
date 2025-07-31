import mlflow
import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import boto3
from datetime import datetime
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLExperimentTracker:
    def __init__(self, experiment_name="Solar_Efficiency_Prediction", s3_bucket=None, aws_region='eu-west-1'):
        """
        Initialize MLflow experiment tracker with S3 artifact storage

        Args:
            experiment_name: Name of the MLflow experiment
            s3_bucket: S3 bucket name for artifact storage (e.g., 'my-mlflow-artifacts')
            aws_region: AWS region for S3 bucket
        """
        self.experiment_name = experiment_name
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region

        # Configure MLflow with S3 artifact storage
        self._setup_mlflow()

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"s3://{s3_bucket}/mlflow-artifacts/{experiment_name}" if s3_bucket else None
            )
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

    def _setup_mlflow(self):
        """Setup MLflow configuration for S3"""

        # Set MLflow tracking URI (adjust as needed)
        mlflow.set_tracking_uri("postgresql://uche:8115@localhost/mlops")

        if self.s3_bucket:
            # Verify S3 access
            try:
                s3_client = boto3.client('s3', region_name=self.aws_region)
                s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"✅ S3 bucket '{self.s3_bucket}' is accessible")
            except Exception as e:
                logger.error(f"❌ Cannot access S3 bucket '{self.s3_bucket}': {e}")
                raise
        else:
            logger.warning("⚠️  No S3 bucket specified. Artifacts will be stored locally.")

    def prepare_data(self, df, target_col='efficiency', test_size=0.2):
        """Prepare data for training"""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Fix data types
        change_dtype = ["humidity", "wind_speed", "pressure"]
        for col in change_dtype:
            X[col] = pd.to_numeric(X[col], errors='coerce')


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def create_preprocessor(self, num_features, cat_features):
        """Create preprocessing pipeline"""
        # Numeric preprocessing: impute → scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical preprocessing: one-hot
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Column transformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

        return preprocessor

    def get_models(self):
        """Define models to experiment with"""
        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Lasso': {
                'model': Lasso(random_state=42),
                'params': {'classifier__alpha': [0.1, 1.0, 10.0]}
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {'classifier__alpha': [0.1, 1.0, 10.0]}
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'classifier__alpha': [0.1, 1.0, 10.0],
                    'classifier__l1_ratio': [0.1, 0.5, 0.9]
                }
            }
        }
        return models

    def run_experiment(self, model_name, model, preprocessor, X_train, X_test, y_train, y_test, params=None):
        """Run a single experiment with MLflow tracking and S3 artifact storage"""

        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Create pipeline
            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('classifier', model)
            ])

            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("s3_bucket", self.s3_bucket)
            mlflow.log_param("aws_region", self.aws_region)

            # Log model parameters
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                for param, value in model_params.items():
                    mlflow.log_param(f"model_{param}", value)

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=5, scoring='neg_mean_squared_error'
            )

            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            cv_score = 100 * (1 - np.sqrt(cv_mse))

            # Train final model
            pipeline.fit(X_train, y_train)

            # Predictions
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            # Metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Log metrics
            mlflow.log_metric("cv_mse", cv_mse)
            mlflow.log_metric("cv_std", cv_std)
            mlflow.log_metric("cv_custom_score", cv_score)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)

            # Log overfitting metric
            overfitting = train_r2 - test_r2
            mlflow.log_metric("overfitting_gap", overfitting)

            # Log model to MLflow (automatically goes to S3 if configured)
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=f"solar_efficiency_{model_name.lower()}"
            )

            # Save additional artifacts to S3 via MLflow
            self._log_additional_artifacts(pipeline, model_name, y_test, y_test_pred)

            print(f"✅ {model_name} experiment completed:")
            print(f"   CV Score: {cv_score:.2f}%")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Test MSE: {test_mse:.4f}")
            if self.s3_bucket:
                print(f"   Artifacts stored in: s3://{self.s3_bucket}/mlflow-artifacts/")

            return {
                'model_name': model_name,
                'cv_score': cv_score,
                'test_r2': test_r2,
                'test_mse': test_mse,
                'run_id': mlflow.active_run().info.run_id,
                'artifact_location': mlflow.active_run().info.artifact_uri
            }

    def _log_additional_artifacts(self, pipeline, model_name, y_test, y_test_pred):
        """Log additional artifacts like predictions and model summary"""

        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save predictions
            predictions_df = pd.DataFrame({
                'actual': y_test.values,
                'predicted': y_test_pred,
                'residuals': y_test.values - y_test_pred
            })
            pred_path = os.path.join(temp_dir, 'test_predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path, "predictions")

            # Save model summary
            model_info = {
                'model_name': model_name,
                'model_type': type(pipeline.named_steps['classifier']).__name__,
                'n_features': len(pipeline.named_steps['preprocess'].feature_names_in_) if hasattr(
                    pipeline.named_steps['preprocess'], 'feature_names_in_') else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'artifact_storage': f"s3://{self.s3_bucket}" if self.s3_bucket else "local"
            }

            summary_path = os.path.join(temp_dir, 'model_summary.txt')
            with open(summary_path, 'w') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")

            mlflow.log_artifact(summary_path, "model_info")

            # Save pipeline as pickle (in addition to MLflow's model format)
            pipeline_path = os.path.join(temp_dir, f'{model_name}_pipeline.pkl')
            joblib.dump(pipeline, pipeline_path)
            mlflow.log_artifact(pipeline_path, "pipeline")

    def run_all_experiments(self, df):
        """Run experiments for all models"""
        num_features = ['irradiance', 'soiling_ratio', 'current', 'panel_age',
                     'voltage', 'humidity', 'cloud_coverage', 'wind_speed']
        cat_features = ["string_id", "error_code", "installation_type"]
        print("🚀 Starting ML experiments with S3 artifact storage...")
        if self.s3_bucket:
            print(f"📦 Artifacts will be stored in: s3://{self.s3_bucket}/mlflow-artifacts/")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Create preprocessor
        preprocessor = self.create_preprocessor(num_features, cat_features)

        # Get models
        models = self.get_models()

        # Run experiments
        results = []

        for model_name, model_config in models.items():
            print(f"\n🔬 Running experiment: {model_name}")

            result = self.run_experiment(
                model_name=model_name,
                model=model_config['model'],
                preprocessor=preprocessor,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                params=model_config.get('params')
            )

            results.append(result)

        # Summary
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_score', ascending=False)

        print("\n📊 Experiment Summary:")
        print("=" * 80)
        print(results_df[['model_name', 'cv_score', 'test_r2', 'test_mse']].to_string(index=False))

        # Log best model info
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   CV Score: {best_model['cv_score']:.2f}%")
        print(f"   Test R²: {best_model['test_r2']:.4f}")
        print(f"   Artifact URI: {best_model['artifact_location']}")

        return results_df

    def load_model_from_s3(self, run_id, model_name="model"):
        """Load a model from S3 using MLflow"""
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)
        return model

    def list_s3_artifacts(self, run_id):
        """List all artifacts for a specific run stored in S3"""
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)

        print(f"Artifacts for run {run_id}:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")

        return artifacts


# Example usage functions
def run_ml_experiments(data_path="your_data.csv", s3_bucket="your-mlflow-bucket", aws_region='eu-west-1'):
    """Main function to run all experiments with S3 storage"""

    # Load data
    df = pd.read_csv(data_path)

    # Initialize experiment tracker with S3
    tracker = MLExperimentTracker(
        experiment_name="Solar_Efficiency_Prediction_S3",
        s3_bucket=s3_bucket,
        aws_region=aws_region
    )

    # Run all experiments
    results = tracker.run_all_experiments(df)

    return results, tracker


def setup_aws_credentials():
    """Helper function to verify AWS credentials setup"""
    try:
        # Check if credentials are available
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials:
            print("✅ AWS credentials found")
            print(f"   Access Key ID: {credentials.access_key[:8]}...")

            # Test S3 access
            s3_client = boto3.client('s3')
            s3_client.list_buckets()
            print("✅ S3 access confirmed")

        else:
            print("❌ No AWS credentials found")
            print("Please set up credentials using one of these methods:")
            print("1. AWS CLI: aws configure")
            print("2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("3. IAM roles (if running on EC2)")

    except Exception as e:
        print(f"❌ AWS setup error: {e}")


if __name__ == "__main__":
    # Check AWS setup
    setup_aws_credentials()

    results, tracker = run_ml_experiments(
        data_path="datasets/train.csv",
        s3_bucket="solarefficiency",
        aws_region="eu-west-1"
    )
    pass
