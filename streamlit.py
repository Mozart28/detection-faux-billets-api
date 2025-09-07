import streamlit as st
st.set_page_config(
    page_title="Genuinely - Détection de faux billets",
    page_icon="💶",
    layout="wide"
)

import pandas as pd
import requests
from io import StringIO
import csv
import matplotlib.pyplot as plt
import time





#  Thème
theme = st.sidebar.selectbox("Choisir le thème", ["Clair", "Sombre"])
if theme == "Clair":
    bg_color = "#f5f7fa"
    text_color = "#17202a"
    btn_color = "#117A65"
    download_btn_color = "#2E86AB"
    kpi_bg_color = "#F0F0F0"
    df_bg_color = "#ffffff"
    df_text_color = "#17202a"
else:
    bg_color = "#1e1e1e"
    text_color = "#f5f5f5"
    btn_color = "#117A65"
    download_btn_color = "#2E86AB"
    kpi_bg_color = "#333333"
    df_bg_color = "#1e1e1e"
    df_text_color = "#f5f5f5"


st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.stButton>button {{
    background-color: {btn_color};
    color: white;
    height: 3em;
    width: 200px;
    border-radius: 10px;
    font-size: 16px;
}}
.stDownloadButton>button {{
    background-color: {download_btn_color};
    color: white;
    border-radius: 10px;
    font-size: 16px;
}}
div.stFileUploader label {{
    color: #117A65;
    font-weight: bold;
    font-size: 16px;
}}
div.stFileUploader {{
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center;'>
    <span style='color: gray; font-size:14px;'>Prédictions_by_Mozart</span>
    <h1 style='text-align: center; color: #2E86AB; flex-grow:1;'>💵 Genuinely 💵</h1>
</div>
<marquee behavior="scroll" direction="left" style="color: #117A65; font-size:20px;">
    🔍 Analysez l'authenticité de vos billets en temps réel !
</marquee>
""", unsafe_allow_html=True)

#  API du modèle Random Forest
API_URL = "https://detection-faux-billets-api-488d.onrender.com/prediction/"


# Upload fichiers CSV

st.header("Importez votre fichier CSV")
uploaded_file = st.file_uploader(
    label="📂 Glissez-déposez votre fichier CSV ici ou cliquez pour le sélectionner",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        sep = csv.Sniffer().sniff(content).delimiter
        df = pd.read_csv(StringIO(content), sep=sep)

        st.subheader("Aperçu des données")
        st.dataframe(df.style.set_properties(**{
            'background-color': df_bg_color,
            'color': df_text_color
        }))

        
        # Envoi à l’API pour prédictions
       
        st.header("Détection")
        if st.button(" Prédire"):
            with st.spinner("Détection en cours..."):
                files = {"fichier": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(API_URL, files=files)

                if response.status_code != 200:
                    st.error(f"Erreur API: {response.status_code}")
                else:
                    data = response.json()
                    if "error" in data:
                        st.error(data["error"])
                    elif "predictions" in data:
                        st.session_state.df_result = pd.DataFrame(data["predictions"])
                        st.session_state.summary = data.get("summary", {"vrai_billet": 0, "faux_billet": 0})
                        st.success(" Vos billets ont été vérifiés avec succès!")
                    else:
                        st.error("⚠ Réponse API invalide")

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")


# Résultats si dispo

if "df_result" in st.session_state:
    st.header("Résultats des prédictions")
    st.dataframe(st.session_state.df_result.style.set_properties(**{
        'background-color': df_bg_color,
        'color': df_text_color
    }))

    vrai = st.session_state.summary.get("vrai_billet", 0)
    faux = st.session_state.summary.get("faux_billet", 0)
    total = vrai + faux

    #  Métadonnées
    if st.button("Métadonnées"):
        st.markdown("""
        ###  Informations des colonnes des résultats
        - **length** : longueur du billet  
        - **margin_low** : marge entre le bas du billet et l'image  
        - **margin_up** : marge entre le haut du billet et l'image  
        - **prediction** : prédiction faite par l'algorithme (0 = vrai billet, 1 = faux billet)  
        - **probabilité_0** : probabilité que le billet soit un faux billet  
        - **probabilité_1** : probabilité que le billet soit un vrai billet 
        """)

    # KPI
    if st.button(" Voir les KPI"):
        pct_vrai = round(vrai / total * 100, 1) if total > 0 else 0
        pct_faux = round(faux / total * 100, 1) if total > 0 else 0
        col1, col2, col3 = st.columns(3)

        def animate_counter(col, label, value, color=text_color):
            placeholder = col.empty()
            for i in range(value + 1):
                placeholder.markdown(f"""
                    <div style="background-color:{kpi_bg_color};padding:20px;border-radius:10px;text-align:center;">
                        <h3>{label}</h3>
                        <h2 style="color:{color}">{i}</h2>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)

        animate_counter(col1, "Total billets", total)
        animate_counter(col2, "Vrais billets", vrai, "green")
        animate_counter(col3, "Faux billets", faux, "red")

    st.markdown("---")

    # Graphiques
    st.header("Graphiques")
    chart_type = st.selectbox("Type de graphique", ["Camembert", "Barplot"], key="graph_select")
    if st.button(" Voir graphique"):
        labels = ["Vrai billet", "Faux billet"]
        values = [vrai, faux]

        if sum(values) == 0:
            st.warning("Aucune donnée pour afficher le graphique.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            if chart_type == "Camembert":
                ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["green", "red"])
                ax.axis("equal")
                ax.set_title("Répartition des billets", color=text_color)
            else:
                ax.bar(labels, values, color=["green", "red"])
                ax.set_ylabel("Nombre", color=text_color)
                ax.set_title("Nombre de billets", color=text_color)
                ax.tick_params(axis='x', colors=text_color)
                ax.tick_params(axis='y', colors=text_color)
            st.pyplot(fig)

    # Distributions des dimensions
    st.sidebar.header("Autres graphiques")
    if st.sidebar.button(" Voir distributions"):
        numeric_cols = st.session_state.df_result.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.warning("Aucune colonne numérique pour générer les histogrammes.")
        else:
            st.header("Histogrammes des features")
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(st.session_state.df_result[col].dropna(), bins=20, color="#117A65", edgecolor="black")
                ax.set_title(f"Distribution de {col}", color=text_color)
                ax.set_xlabel(col, color=text_color)
                ax.set_ylabel("Fréquence", color=text_color)
                ax.tick_params(axis='x', colors=text_color)
                ax.tick_params(axis='y', colors=text_color)
                st.pyplot(fig)

    #  Télécharger CSV 
    st.header("Télécharger le fichier de prédictions")
    csv_data = st.session_state.df_result.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(
        label=" Télécharger CSV",
        data=csv_data,
        file_name="resultats_predictions.csv",
        mime="text/csv"
    )
