import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    if uploaded_file is not None:
        # Handle bulk prediction file upload
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        response_bytes = BytesIO(response.content)
        response_df = pd.read_csv(response_bytes)

        # Download button for bulk predictions
        st.download_button(
            label="Download Predictions",
            data=response_bytes,
            file_name="Predictions.csv",
            key="result_download_button",
        )

    else:
        # Handle single text input prediction
        response = requests.post(prediction_endpoint, data={"text": user_input})
        response = response.json()  # Convert response to JSON
        
        # Check if 'prediction' exists in the response
        if 'prediction' in response:
            st.write(f"Predicted sentiment: {response['prediction']}")
        else:
            # Display error if 'prediction' is not in response
            st.error(f"Unexpected response format: {response}")
