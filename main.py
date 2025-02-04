import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pickle
import io

app = FastAPI(
    title="API for AquaPredict",
    description="Predicting Water Quality Parameters"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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

    # Handle missing values (fill with mean)
    input_data = input_data.fillna(input_data.mean())

    # Perform predictions
    predictions = model.predict(input_data)

    # Add predictions to the original data
    original_data["Potable"] = ["Yes" if pred == 1 else "No" for pred in predictions]

    # Save the resulting DataFrame to a new CSV
    output_file_path = "output_with_predictions.csv"
    original_data.to_csv(output_file_path, index=False)

    # Return the resulting CSV file as a response
    return FileResponse(output_file_path, media_type="text/csv", filename="predictions.csv")
