import streamlit as st
import pandas as pd
import joblib
import boto3
import os
import warnings
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Page config
st.set_page_config(
    page_title="Solar Panel Efficiency Predictor",
    page_icon="☀️",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load model from S3 (cached)"""
    RUN_ID = os.getenv("RUN_ID", "649fb2c8074f473c91c885b02a323c6e")
    S3_BUCKET = os.getenv("S3_BUCKET", "solarefficiency")
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "solar-experiment")

    try:
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
        return model, RUN_ID

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def prepare_features(data):
    """Prepare features for prediction"""
    df = pd.DataFrame([data])

    # Fix data types
    change_dtype = ["humidity", "wind_speed", "pressure"]
    for col in change_dtype:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def predict_efficiency(model, features):
    """Make prediction"""
    try:
        pred = model.predict(features)
        return float(pred[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Main app
def main():
    # Header
    st.title("☀️ Solar Panel Efficiency Predictor")
    st.markdown("---")

    # Load model
    model, run_id = load_model()

    if model is None:
        st.error("❌ Could not load model. Please check configuration.")
        st.stop()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.info(f"**Model Type:** Ridge Regression")
        st.info(f"**Run ID:** {run_id}")
        st.markdown("---")

        # Sample data button
        if st.button("📝 Load Sample Data"):
            st.session_state.load_sample = True

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Batch Predict", "📈 Analysis"])

    with tab1:
        st.header("Single Prediction")

        # Check if we should load sample data
        load_sample = st.session_state.get('load_sample', False)

        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Environmental")
                temperature = st.number_input("Temperature (°C)", value=25.0 if not load_sample else 17.62)
                irradiance = st.number_input("Irradiance (W/m²)", value=1000.0 if not load_sample else 85.45)
                humidity = st.number_input("Humidity (%)", value=50.0 if not load_sample else 90.82)
                cloud_coverage = st.number_input("Cloud Coverage (%)", value=10.0 if not load_sample else 33.51)
                wind_speed = st.number_input("Wind Speed (m/s)", value=5.0 if not load_sample else 7.18)
                pressure = st.number_input("Pressure (hPa)", value=1013.25 if not load_sample else 1034.78)

            with col2:
                st.subheader("Panel Specs")
                panel_age = st.number_input("Panel Age (years)", value=5.0 if not load_sample else 13.91)
                maintenance_count = st.number_input("Maintenance Count", value=2.0 if not load_sample else 6.0)
                soiling_ratio = st.number_input("Soiling Ratio", value=0.9 if not load_sample else 0.89, min_value=0.0,
                                                max_value=1.0)
                voltage = st.number_input("Voltage (V)", value=30.0 if not load_sample else 6.37)
                current = st.number_input("Current (A)", value=8.0 if not load_sample else 0.069)
                module_temperature = st.number_input("Module Temperature (°C)",
                                                     value=35.0 if not load_sample else 19.52)

            with col3:
                st.subheader("System Info")
                string_id = st.selectbox("String ID", ["A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "C5"],
                                         index=6 if load_sample else 0)
                error_code = st.selectbox("Error Code", ["E00", "E01", "E02", "E03", "E04", "E05"],
                                          index=1 if load_sample else 0)
                installation_type = st.selectbox("Installation Type", ["fixed", "tracking"],
                                                 index=1 if load_sample else 0)

            # Predict button
            predict_btn = st.form_submit_button("🔮 Predict Efficiency", use_container_width=True)

        if predict_btn:
            # Prepare data
            input_data = {
                'temperature': temperature,
                'irradiance': irradiance,
                'humidity': humidity,
                'panel_age': panel_age,
                'maintenance_count': maintenance_count,
                'soiling_ratio': soiling_ratio,
                'voltage': voltage,
                'current': current,
                'module_temperature': module_temperature,
                'cloud_coverage': cloud_coverage,
                'wind_speed': wind_speed,
                'pressure': pressure,
                'string_id': string_id,
                'error_code': error_code,
                'installation_type': installation_type
            }

            # Make prediction
            features = prepare_features(input_data)
            efficiency = predict_efficiency(model, features)*100

            if efficiency is not None:
                # Display result
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Efficiency gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=efficiency,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Efficiency (%)"},
                        delta={'reference': 80},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Efficiency interpretation
                if efficiency >= 85:
                    st.success(f"🎉 Excellent efficiency: {efficiency:.1f}%")
                elif efficiency >= 70:
                    st.info(f"👍 Good efficiency: {efficiency:.1f}%")
                elif efficiency >= 50:
                    st.warning(f"⚠️ Acceptable efficiency: {efficiency:.1f}%")
                else:
                    st.error(f"❌ Poor efficiency: {efficiency:.1f}%")

        # Reset sample data flag
        if load_sample:
            st.session_state.load_sample = False

    with tab2:
        st.header("Batch Prediction")

        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(df.head())

                if st.button("🚀 Predict All"):
                    predictions = []

                    progress_bar = st.progress(0)
                    for i, row in df.iterrows():
                        features = prepare_features(row.to_dict())
                        pred = predict_efficiency(model, features)
                        predictions.append(pred)
                        progress_bar.progress((i + 1) / len(df))

                    # Add predictions to dataframe
                    df['predicted_efficiency'] = predictions

                    # Display results
                    st.success(f"✅ Predicted efficiency for {len(df)} samples")
                    st.dataframe(df[['predicted_efficiency']])

                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab3:
        st.header("Feature Analysis")
        st.write("This tab could show feature importance, model metrics, etc.")

        # Sample analysis (you can expand this)
        sample_data = {
            'Feature': ['Irradiance', 'Temperature', 'Panel Age', 'Soiling Ratio', 'Voltage'],
            'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        }

        fig = px.bar(sample_data, x='Feature', y='Importance',
                     title="Feature Importance (Sample)")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
