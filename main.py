import pandas as pd
from fastapi import FastAPI
from model_predict import WaterPredict
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np

app = FastAPI(
    title="API for AquaPredict",
    description="Predicting Water Quality Parameters"
)

with open("model.pkl", "rb") as f:
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
    
@app.post("/metrics/")
def metrics():
    # Load test data and calculate metrics
    test_datapreprocessed = pd.read_csv('/data/preprocessed/test_data_preprocessed.csv')
    X_test = test_datapreprocessed.iloc[:, 0:-1].values
    y_test = test_datapreprocessed.iloc[:, -1].values
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_s = f1_score(y_test, y_pred)
    metrics_dict = {'accuracy': acc, 'precision': prec, 'recall': recall, 'f1_score': f1_s}
    
    # Save metrics to metrics.json
    with open('metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    return metrics_dict
