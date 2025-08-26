from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import StringIO

app = FastAPI()


pipeline_log = joblib.load("model_detection_faux_billets.pkl")

@app.post("/prediction/")
async def predict(fichier: UploadFile = File(...)):
    try:
        contenu = await fichier.read()
        text_data = contenu.decode("utf-8")

        
        if ";" in text_data:
            df = pd.read_csv(StringIO(text_data), sep=";")
        else:
            df = pd.read_csv(StringIO(text_data))  

       
        colonnes_utiles = ["margin_low", "margin_up","length"]
        colonnes_manquantes = [col for col in colonnes_utiles if col not in df.columns]

        if colonnes_manquantes:
            return {"error": f"Colonnes manquantes : {', '.join(colonnes_manquantes)}"}

        
        df_model = df[colonnes_utiles].copy()
        df_model["margin_low"] = df_model["margin_low"].fillna(df_model["margin_low"].median())

        # Prédictions
        predictions = pipeline_log.predict(df_model)
        proba_predictions = pipeline_log.predict_proba(df_model)[:, 1]

        df_model["prediction"] = predictions
        df_model["proba"] = [round(p, 2) for p in proba_predictions]

        #  graphique
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
