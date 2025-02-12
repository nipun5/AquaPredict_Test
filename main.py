import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pickle
import io
import mlflow
import os

app = FastAPI(
    title="API for AquaPredict",
    description="Predicting Water Quality Parameters"
)

# Set the MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "nipun5"
repo_name = "AquaPredict_CDAC"
dagshub_token = "f1bc8f2f71568383b82e0ec42eb6bad23d3b1fa4"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Load the latest model from MLflow
def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("Best Model", stages=["Production"])
    run_id = versions[0].run_id
    print(run_id)
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")

model = load_model()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/predict_file/")
def predict_file(file: UploadFile = File(...)):
    # Read the uploaded file into a DataFrame
    contents = file.file.read()
    input_data = pd.read_csv(io.BytesIO(contents))

    # Store original columns for output
    original_data = input_data.copy()

    # Drop the target column 'Potability' or any extra columns if present
    if "Potability" in input_data.columns:
        input_data = input_data.drop(columns=["Potability"])

    # Define the expected columns
    expected_columns = [
        "ph", "Hardness", "Solids", "Chloramines",
        "Sulfate", "Conductivity", "Organic_carbon",
        "Trihalomethanes", "Turbidity"
    ]

    # Ensure only the required columns are used and drop others temporarily
    input_data = input_data[expected_columns]

    input_data = input_data.fillna(input_data.mean())

    # Perform predictions
    predictions = model.predict(input_data)

    # Add predictions to the original data
    original_data["Potable"] = ["Yes" if pred == 1 else "No" for pred in predictions]

    # Save the resulting DataFrame to a new CSV
    output_file_path = "output_with_predictions.csv"
    original_data.to_csv(output_file_path, index=False)

    # Calculate the overall probabilities
    potable_prob = sum(predictions) / len(predictions)
    non_potable_prob = 1 - potable_prob

    # Return the probabilities and the CSV file as a response
    return {
        "potable_probability": potable_prob,
        "non_potable_probability": non_potable_prob,
        "file_url": f"/download/{output_file_path}"
    }

@app.get("/download/{file_path:path}")
def download_file(file_path: str):
    return FileResponse(file_path, media_type="text/csv", filename="predictions.csv")