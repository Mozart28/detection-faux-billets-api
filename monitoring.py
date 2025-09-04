from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import StringIO
from evidently import Report
from evidently.metrics import ValueDrift, MissingValueCount, RowCount

app = FastAPI()

# Charger modèle
pipeline_rf = joblib.load("model_detection_faux_billets.pkl")

# Charger référence
billets = pd.read_csv("billets.csv", sep=";")
columns_to_monitor = ["margin_low", "margin_up", "length"]
billets_features = billets[columns_to_monitor]

@app.post("/monitoring/")
async def monitoring(file: UploadFile = File(...)):
    try:
        # Charger nouveau fichier
        content = await file.read()
        text_data = content.decode("utf-8")

        if ";" in text_data:
            df = pd.read_csv(StringIO(text_data), sep=";")
        else:
            df = pd.read_csv(StringIO(text_data))

        # Nettoyage des colonnes : supprimer espaces et mettre en minuscules
        df.columns = df.columns.str.strip().str.lower()

        # Vérifier colonnes
        colonnes_manquantes = [col for col in columns_to_monitor if col not in df.columns]
        if colonnes_manquantes:
            return {
                "error": f"Les colonnes suivantes sont manquantes : {', '.join(colonnes_manquantes)}"
            }

        # Sélectionner colonnes utiles
        current_features = df[columns_to_monitor].copy()
        current_features["margin_low"] = current_features["margin_low"].fillna(
            current_features["margin_low"].median()
        )

        # Prédictions
        predictions = pipeline_rf.predict(current_features)
        current_features["prediction"] = predictions  # ajout mais pas pour Evidently

        # Créer métriques Evidently
        metrics_list = [ValueDrift(column_name=col) for col in columns_to_monitor]
        metrics_list += [MissingValueCount(), RowCount()]

        report = Report(metrics=metrics_list)

        # Passer uniquement les colonnes surveillées
        report.run(
            reference_data=billets_features,
            current_data=current_features[columns_to_monitor]
        )

        # Sauvegarde HTML
        report.save_html("monitoring_report.html")

        # Retourner résultats JSON
        return report.as_dict()

    except Exception as e:
        return {"error": str(e)}
