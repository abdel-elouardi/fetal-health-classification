import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Charger le modèle et le scaler
model  = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI(title="Fetal Health API")

# Structure des données reçues
class Donnees(BaseModel):
    features: List[float]

# Route principale
@app.get("/")
def accueil():
    return {"message": "Fetal Health API", "status": "ok"}

# Route de prédiction
@app.post("/predict")
def predire(data: Donnees):
    # Convertir en tableau numpy
    X = np.array(data.features).reshape(1, -1)
    # Normaliser
    X = scaler.transform(X)
    # Prédire
    prediction = model.predict(X)[0]
    # Convertir en label
    labels = {0: "Normal", 1: "Suspect", 2: "Pathologique"}
    return {
        "prediction": int(prediction),
        "label": labels[prediction]
    }
