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
            <title>Welcome to AquaPredict</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #8e2de2, #4a00e0);
                    color: white;
                    text-align: center;
                    padding: 20px;
                }
                .container {
                    max-width: 500px;
                    margin: 50px auto;
                    padding: 20px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                    color: black;
                }
                h1 {
                    font-size: 2rem;
                    margin-bottom: 10px;
                }
                p {
                    margin: 10px 0;
                    font-size: 1rem;
                }
                form {
                    display: flex;
                    flex-direction: column;
                }
                input[type="number"], input[type="submit"] {
                    padding: 10px;
                    margin: 5px 0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    font-size: 1rem;
                }
                input[type="submit"] {
                    background: linear-gradient(to right, #4a00e0, #8e2de2);
                    color: white;
                    cursor: pointer;
                    border: none;
                }
                input[type="submit"]:hover {
                    background: linear-gradient(to right, #8e2de2, #4a00e0);
                }
                footer {
                    margin-top: 20px;
                    font-size: 0.9rem;
                    color: white;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to AquaPredict</h1>
                <p>AquaPredict aims to address critical issues by leveraging machine learning and MLOps to predict water potability based on water quality metrics. By implementing an end-to-end MLOps workflow, we aim to provide an automated, maintainable, and collaborative solution for water quality analysis.</p>
                <form action="/predict/" method="post">
                    <input type="number" step="any" id="ph" name="ph" placeholder="pH">
                    <input type="number" step="any" id="Hardness" name="Hardness" placeholder="Hardness">
                    <input type="number" step="any" id="Solids" name="Solids" placeholder="Solids">
                    <input type="number" step="any" id="Chloramines" name="Chloramines" placeholder="Chloramines">
                    <input type="number" step="any" id="Sulfate" name="Sulfate" placeholder="Sulfate">
                    <input type="number" step="any" id="Conductivity" name="Conductivity" placeholder="Conductivity">
                    <input type="number" step="any" id="Organic_carbon" name="Organic_carbon" placeholder="Organic Carbon">
                    <input type="number" step="any" id="Trihalomethanes" name="Trihalomethanes" placeholder="Trihalomethanes">
                    <input type="number" step="any" id="Turbidity" name="Turbidity" placeholder="Turbidity">
                    <input type="submit" value="Predict">
                </form>
            </div>
            <footer>
                Nipun Jain 240810125012<br>
                Aryan Saxena 240810125013<br>
                Saket Kothari 240810125008<br>
                2024-2025 C
            </footer>
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
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #8e2de2, #4a00e0);
                    color: white;
                    text-align: center;
                    padding: 20px;
                }}
                .container {{
                    max-width: 500px;
                    margin: 50px auto;
                    padding: 20px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                    color: black;
                }}
                h1 {{
                    font-size: 2rem;
                    margin-bottom: 10px;
                }}
                p {{
                    margin: 10px 0;
                    font-size: 1rem;
                }}
                a {{
                    color: #4a00e0;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <p>The water is: <strong>{result}</strong></p>
                <a href="/">Go Back</a>
            </div>
        </body>
    </html>
    """
