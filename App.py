# -*- coding: utf-8 -*-
"""
Streamlit App — Analyseur Hippique IA (avec scraping Geny, auto-apprentissage, et affichage direct du pronostic)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re, os, joblib, warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏇 Analyseur Hippique IA", page_icon="🐎", layout="wide")

# --------------------- UTILITAIRES ---------------------

def safe_float(val, default=0.0):
    try:
        return float(str(val).replace(',', '.'))
    except:
        return default

def music_to_score(music):
    digits = [int(x) for x in re.findall(r'\d+', str(music))]
    if not digits:
        return 0, 0, 0
    recent_win = sum(d == 1 for d in digits)
    top3 = sum(d <= 3 for d in digits)
    weights = np.linspace(1, 0.3, num=len(digits))
    weighted = np.sum((4 - np.array(digits)) * weights) / (len(digits) + 1e-6)
    return recent_win, top3, weighted

# --------------------- SCRAPER GENY ---------------------

@st.cache_data(ttl=600)
def scrape_geny_course(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, f"Erreur HTTP {r.status_code}"

        soup = BeautifulSoup(r.text, "html.parser")

        # repérage du tableau principal
        table = soup.find("table")
        if not table:
            return None, "Aucun tableau détecté"

        rows = table.find_all("tr")[1:]
        data = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 5:
                data.append({
                    "Numéro": cols[0],
                    "Cheval": cols[1],
                    "Musique": cols[2],
                    "Driver/Jockey": cols[3],
                    "Entraîneur": cols[4] if len(cols) > 4 else "",
                    "Cote": cols[-1] if len(cols) >= 6 else "0",
                    "Gains": cols[-2] if len(cols) >= 6 else "0"
                })
        if not data:
            return None, "Aucune donnée extraite"
        df = pd.DataFrame(data)
        return df, "Succès"
    except Exception as e:
        return None, f"Erreur scraping: {e}"

# --------------------- MODÈLE DL ---------------------

class ModelDL:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        os.makedirs("models", exist_ok=True)
        self.model_path = "models/geny_dl.keras"
        self.scaler_path = "models/geny_scaler.joblib"

    def build(self, input_dim):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=60):
        Xs = self.scaler.fit_transform(X)
        model = self.build(Xs.shape[1])
        cb = [callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        hist = model.fit(Xs, y, epochs=epochs, validation_split=0.2, verbose=0, callbacks=cb)
        self.model, self.history = model, hist.history
        model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict(self, X):
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                return np.zeros(len(X))
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).flatten()

# --------------------- FONCTIONS D'ANALYSE ---------------------

def prepare_data(df):
    df = df.copy()
    df["Cote_num"] = df["Cote"].apply(lambda x: safe_float(x, 99))
    df["Gains_num"] = df["Gains"].apply(lambda x: safe_float(re.sub(r"[^\d,\.]", "", x), 0))
    df["recent_win"], df["top3"], df["weighted"] = zip(*df["Musique"].apply(music_to_score))
    df = df.fillna(0)
    return df

def generate_predictions(df, model_dl):
    X = df[["Cote_num", "Gains_num", "recent_win", "top3", "weighted"]].values
    y_pseudo = 0.6 * (1 / (df["Cote_num"] + 0.1)) + 0.4 * (df["weighted"] / (df["weighted"].max() + 1e-6))
    model_dl.train(X, y_pseudo)
    preds = model_dl.predict(X)
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-6)
    df["Score"] = preds
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rang"] = df.index + 1
    return df

def generate_combos(df, n=15):
    horses = df["Cheval"].tolist()
    combos = []
    for i in range(min(len(horses)-2, n)):
        combos.append(tuple(horses[i:i+3]))
    return combos

# --------------------- INTERFACE STREAMLIT ---------------------

def main():
    st.markdown("<h1 style='text-align:center;color:#1e40af;'>🏇 Analyseur Hippique IA — Scraping & Pronostic</h1>", unsafe_allow_html=True)

    # BARRE DE RECHERCHE
    url = st.text_input("🔍 Entrez l’URL de la course Geny :", placeholder="https://www.geny.com/stats-pmu?id_course=...")
    lancer = st.button("Analyser la course")

    df_final = None
    if lancer and url:
        with st.spinner("⏳ Extraction des données depuis Geny..."):
            df, msg = scrape_geny_course(url)
            if df is None:
                st.error(f"❌ {msg}")
                return
            st.success(f"✅ {len(df)} chevaux extraits ({msg})")
            st.dataframe(df)
            df_final = df

    if df_final is not None:
        st.markdown("---")
        st.header("🤖 Analyse IA & Pronostic")
        df_prep = prepare_data(df_final)
        model = ModelDL()
        df_pred = generate_predictions(df_prep, model)

        st.subheader("🏆 Classement Prédictif")
        st.dataframe(df_pred[["Rang", "Cheval", "Cote", "Gains", "Score"]].style.format({"Score": "{:.3f}"}), use_container_width=True)

        st.subheader("🥇 Top 3 Chevaux Prédits")
        for i in range(3):
            row = df_pred.iloc[i]
            st.markdown(f"<div style='border-left:4px solid #f59e0b;padding:8px;margin:4px;background:#fff8e1;'>"
                        f"<b>{i+1}. {row['Cheval']}</b> — Cote {row['Cote']} | Score {row['Score']:.3f}</div>", unsafe_allow_html=True)

        st.subheader("🎲 Combinaisons e-Trio")
        combos = generate_combos(df_pred, n=20)
        for i, c in enumerate(combos, 1):
            st.write(f"{i}. {c[0]} — {c[1]} — {c[2]}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Télécharger CSV", df_pred.to_csv(index=False), file_name="pronostic_geny.csv", mime="text/csv")
        with col2:
            st.download_button("📋 Télécharger JSON", df_pred.to_json(orient="records", indent=2), file_name="pronostic_geny.json", mime="application/json")

        st.success("✅ Pronostic affiché et prêt à être exporté !")

if __name__ == "__main__":
    main()
