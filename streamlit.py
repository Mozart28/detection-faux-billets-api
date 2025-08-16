
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ----------------------
# CONFIGURATION APP
# ----------------------
st.set_page_config(page_title="Genuinely - Détection de faux billets", layout="wide")

# Titre et sous-titre
st.title("💵 Genuinely")
st.markdown("<marquee>Analysez l'authenticité de vos billets</marquee>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Options")

# Choix de fonctionnalité
option = st.sidebar.radio(
    "Sélectionnez une fonctionnalité :",
    ["KPI", "Graphiques Prédiction", "Autres graphiques"]
)

# Upload fichier
uploaded_file = st.file_uploader("📂 Importez un fichier CSV", type=["csv"])

# URL de l'API (à remplacer par ton URL Render)
API_URL = "https://ton-api-render.onrender.com/predict/"

if uploaded_file is not None:
    # Charger le CSV avec détection automatique du séparateur
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        st.error("❌ Erreur lors de la lecture du fichier. Vérifiez le séparateur (',' ou ';').")
        st.stop()

    # Appel à l'API FastAPI
    with st.spinner("🔎 Analyse en cours..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()

        if "error" in data:
            st.error(f"⚠️ Erreur API : {data['error']}")
        else:
            df_results = pd.DataFrame(data)

            # Affichage en fonction de l'option choisie
            if option == "KPI":
                st.subheader("📊 Indicateurs clés (KPI)")
                total = len(df_results)
                vrais = (df_results["prediction"] == 0).sum()
                faux = (df_results["prediction"] == 1).sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre total de billets", total)
                col2.metric("Billets authentiques", vrais)
                col3.metric("Billets suspects", faux)

            elif option == "Graphiques Prédiction":
                st.subheader("📈 Graphiques des Prédictions")
                chart_type = st.radio("Choisissez le type de graphique :", ["Camembert", "Barplot"])

                count_pred = df_results["prediction"].value_counts().reset_index()
                count_pred.columns = ["Classe", "Nombre"]
                count_pred["Classe"] = count_pred["Classe"].map({0: "Authentique", 1: "Faux"})

                if chart_type == "Camembert":
                    fig = px.pie(count_pred, values="Nombre", names="Classe", title="Répartition des billets")
                else:
                    fig = px.bar(count_pred, x="Classe", y="Nombre", title="Nombre de billets par classe",
                                 text="Nombre", color="Classe")

                st.plotly_chart(fig, use_container_width=True)

            elif option == "Autres graphiques":
                st.subheader("📊 Histogrammes des variables")
                features = [col for col in df_results.columns if col not in ["prediction", "proba"]]

                for col in features:
                    fig = px.histogram(df_results, x=col, nbins=30,
                                       title=f"Distribution de {col}", marginal="box")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("❌ Erreur API lors de la prédiction.")
