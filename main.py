import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle

app = FastAPI(
    title="API for AquaPredict",
    description="Predicting Water Quality Parameters"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>AquaPredict Dashboard</title>
        </head>
        <body>
            <h1>Welcome to AquaPredict</h1>
            <form action="/predict/" method="post">
                <label for="ph">pH:</label><br>
                <input type="number" step="any" id="ph" name="ph"><br>
                <label for="Hardness">Hardness:</label><br>
                <input type="number" step="any" id="Hardness" name="Hardness"><br>
                <label for="Solids">Solids:</label><br>
                <input type="number" step="any" id="Solids" name="Solids"><br>
                <label for="Chloramines">Chloramines:</label><br>
                <input type="number" step="any" id="Chloramines" name="Chloramines"><br>
                <label for="Sulfate">Sulfate:</label><br>
                <input type="number" step="any" id="Sulfate" name="Sulfate"><br>
                <label for="Conductivity">Conductivity:</label><br>
                <input type="number" step="any" id="Conductivity" name="Conductivity"><br>
                <label for="Organic_carbon">Organic Carbon:</label><br>
                <input type="number" step="any" id="Organic_carbon" name="Organic_carbon"><br>
                <label for="Trihalomethanes">Trihalomethanes:</label><br>
                <input type="number" step="any" id="Trihalomethanes" name="Trihalomethanes"><br>
                <label for="Turbidity">Turbidity:</label><br>
                <input type="number" step="any" id="Turbidity" name="Turbidity"><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict/", response_class=HTMLResponse)
def predict(
    ph: float = Form(...),
    Hardness: float = Form(...),
    Solids: float = Form(...),
    Chloramines: float = Form(...),
    Sulfate: float = Form(...),
    Conductivity: float = Form(...),
    Organic_carbon: float = Form(...),
    Trihalomethanes: float = Form(...),
    Turbidity: float = Form(...)
):
    sample = pd.DataFrame({
        'ph': [ph],
        'Hardness': [Hardness],
        'Solids': [Solids],
        'Chloramines': [Chloramines],
        'Sulfate': [Sulfate],
        'Conductivity': [Conductivity],
        'Organic_carbon': [Organic_carbon],
        'Trihalomethanes': [Trihalomethanes],
        'Turbidity': [Turbidity]
    })
    prediction = model.predict(sample)
    result = "Potable" if prediction == 1 else "Not Potable"
    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p>The water is: <strong>{result}</strong></p>
            <a href="/">Go Back</a>
        </body>
    </html>
    """
