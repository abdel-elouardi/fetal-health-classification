import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Charger le modèle
model  = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Titre
st.title("🏥 Fetal Health Classification")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Données", "🔮 Prédiction"])

# PAGE 1 — Données
if page == "📊 Données":
    st.header("📊 Exploration des données")
    df = pd.read_csv('data/raw/fetal_health.csv')
    
    # Résumé
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Classes", df['fetal_health'].nunique())
    
    # Aperçu
    st.subheader("Aperçu")
    st.dataframe(df.head(10))
    
    # Distribution
    st.subheader("Distribution des classes")
    fig, ax = plt.subplots()
    df['fetal_health'].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xticklabels(["Normal", "Suspect", "Pathologique"], rotation=0)
    st.pyplot(fig)
    
    # Heatmap
    st.subheader("Corrélation")
    st.image("models/correlation.png")

# PAGE 2 — Prédiction
elif page == "🔮 Prédiction":
    st.header("🔮 Faire une prédiction")
    st.info("Entre les valeurs et clique sur Prédire")
    
    df = pd.read_csv('data/raw/fetal_health.csv')
    cols = [c for c in df.columns if c != 'fetal_health']
    
    # Filtrer les colonnes utilisées par le modèle
    n_features = model.n_features_in_
    cols = cols[:n_features]
    
    # Sliders pour chaque feature
    valeurs = []
    for col in cols:
        val = st.slider(col, 
                       float(df[col].min()), 
                       float(df[col].max()), 
                       float(df[col].mean()))
        valeurs.append(val)
    
    if st.button("🔮 Prédire", type="primary"):
        X = np.array(valeurs).reshape(1, -1)
        X = scaler.transform(X)
        pred = model.predict(X)[0]
        labels = {0: "✅ Normal", 1: "⚠️ Suspect", 2: "🚨 Pathologique"}
        couleurs = {0: "success", 1: "warning", 2: "error"}
        st.success(f"Résultat : {labels[pred]}")
