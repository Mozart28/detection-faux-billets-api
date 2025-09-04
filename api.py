from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import StringIO

app = FastAPI()

# Charger le modèle
pipeline_rf = joblib.load("model_detection_faux_billets.pkl")

@app.post("/prediction/")
async def predict(fichier: UploadFile = File(...)):
    try:
        # Lecture du fichier
        contenu = await fichier.read()
        text_data = contenu.decode("utf-8")

        # Détection du séparateur
        sep = ";" if ";" in text_data else ","
        df = pd.read_csv(StringIO(text_data), sep=sep)

        # Colonnes nécessaires
        colonnes_utiles = ["margin_low", "margin_up", "length"]
        colonnes_manquantes = [col for col in colonnes_utiles if col not in df.columns]

        if colonnes_manquantes:
            return {"error": f"Colonnes manquantes : {', '.join(colonnes_manquantes)}"}

        # Préparation des données
        df_model = df[colonnes_utiles].copy()

        # Imputation uniquement si des NaN existent
        if df_model["margin_low"].isnull().any():
            df_model["margin_low"] = df_model["margin_low"].fillna(df_model["margin_low"].median())

        # Prédictions
        predictions = pipeline_rf.predict(df_model)
        proba_predictions_1 = pipeline_rf.predict_proba(df_model)[:, 1]
        proba_predictions_0 = pipeline_rf.predict_proba(df_model)[:, 0]


        df_model["predictions"] = predictions
        df_model["probabilités_0"] = [round(p, 2) for p in proba_predictions_0]
        df_model["probabilités_1"] = [round(p, 2) for p in proba_predictions_1]
        # Résumé
        counts = pd.Series(predictions).value_counts().to_dict()
        summary = {
            "vrai_billet": counts.get(0, 0),
            "faux_billet": counts.get(1, 0),
            "total": len(predictions)
        }

        return {
            "predictions": df_model.to_dict(orient="records"),
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
