
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import StringIO
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

app = FastAPI()

# Charger modèle
pipeline_rf = joblib.load("model_detection_faux_billets.pkl")

# Référence (jeu d’entraînement utilisé comme baseline)
billets = pd.read_csv("reference_data.csv")  
billets_features= billets[["margin_up", "margin_low","length"]]

@app.post("/monitoring/")
async def monitoring(file: UploadFile = File(...)):
    try:
        # Charger nouveau fichier
        content = await file.read()
        text_data = content.decode("utf-8")

        if ";" in text_data:
            df = pd.read_csv(StringIO(text_data), sep=";")
        else:
            df = pd.read_csv(StringIO(text_data))  # par défaut ','

        # Vérifier colonnes
        colonnes_utiles = ["margin_up", "margin_low","length"]
        colonnes_manquantes = [col for col in colonnes_utiles if col not in df.columns]

        if colonnes_manquantes:
            return {"error": f"Colonnes manquantes : {', '.join(colonnes_manquantes)}"}

        # Nettoyage
        current_features = df[colonnes_utiles].copy()
        current_features["margin_low"] = current_features["margin_low"].fillna(current_features["margin_low"].median())

        # Prédictions
        predictions = pipeline_rf.predict(current_features)
        current_features["prediction"] = predictions

        # Rapport Evidently (sans y_true)
        report = Report(metrics=[
            DataQualityPreset(),   # qualité des données
            DataDriftPreset(),     # dérive des features
            TargetDriftPreset()    # dérive des prédictions
        ])

        report.run(reference_data=billets_features, current_data=current_features)

        # Retourner en JSON
        return report.as_dict()

    except Exception as e:
        return {"error": str(e)}
