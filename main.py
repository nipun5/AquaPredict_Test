import pandas as pd
from fastapi import FastAPI
from model_predict import WaterPredict
import pickle

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

// React Dashboard Component
import { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Toast, ToastProvider, useToast } from "@/components/ui/toast";

export default function AquaPredictDashboard() {
  const [inputData, setInputData] = useState({
    ph: "",
    Hardness: "",
    Solids: "",
    Chloramines: "",
    Sulfate: "",
    Conductivity: "",
    Organic_carbon: "",
    Trihalomethanes: "",
    Turbidity: "",
  });
  const [result, setResult] = useState(null);
  const { toast } = useToast();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputData({ ...inputData, [name]: value });
  };

  const handlePrediction = async () => {
    try {
      const response = await fetch("/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputData),
      });
      const data = await response.json();
      setResult(data);
      toast({ description: `Prediction: ${data.Prediction}` });
    } catch (error) {
      console.error("Error during prediction:", error);
      toast({ description: "Failed to get prediction. Try again later.", variant: "error" });
    }
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-2xl font-bold mb-4">AquaPredict Dashboard</h1>
      <Card className="max-w-xl mx-auto">
        <CardContent>
          <form onSubmit={(e) => e.preventDefault()}>
            <div className="grid grid-cols-2 gap-4">
              {Object.keys(inputData).map((key) => (
                <div key={key}>
                  <Label htmlFor={key}>{key}</Label>
                  <Input
                    id={key}
                    name={key}
                    value={inputData[key]}
                    onChange={handleInputChange}
                    type="number"
                    placeholder={`Enter ${key}`}
                  />
                </div>
              ))}
            </div>
            <Button onClick={handlePrediction} className="mt-4 w-full">
              Predict
            </Button>
          </form>
          {result && (
            <div className="mt-6 p-4 bg-green-100 rounded">
              <p className="font-bold">Prediction Result:</p>
              <p>{result.Prediction}</p>
            </div>
          )}
        </CardContent>
      </Card>
      <ToastProvider>
        <Toast />
      </ToastProvider>
    </div>
  );
}
