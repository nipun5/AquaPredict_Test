import pandas as pd
from fastapi import FastAPI
from model_predict import WaterPredict
import pickle

app = FastAPI(
    title="API for AquaPredict",
    description="Predicting Water Quality Parameters"
)

with open("C:/Users/jainn/OneDrive/Documents/AquaPredict/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to AquaPredict"

@app.post("/predict/")
def predict(water: WaterPredict):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })
    prediction = model.predict(sample)
    if prediction == 1:
        return {"Prediction": "Potable"}
    else:
        return {"Prediction": "Not Potable"}
    
