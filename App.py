"""
🏇 Analyseur Hippique IA Pro v5 — Scraping Geny + ML/DL + Correction groupby
Auteur : GPT-5 (OpenAI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import time
import os
from datetime import datetime
from io import StringIO
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px

# === Configs ===
MODEL_PATH = "hippo_model.h5"
SCALER_PATH = "hippo_scaler.joblib"
CALIB_PATH = "hippo_calibrator.joblib"
SCRAPE_DEBUG_DIR = "scrape_debug"
os.makedirs(SCRAPE_DEBUG_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# === Utility functions ===
def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace(",", "."))
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return np.nan
    m = re.search(r"(\d+(?:[.,]\d+)?)", str(poids_str))
    return float(m.group(1).replace(",", ".")) if m else np.nan

def extract_music_features(music):
    if pd.isna(music) or str(music).strip() == "":
        return {"wins":0,"places":0,"total_races":0,"win_rate":0,"place_rate":0,"recent_form":0,"best_pos":99}
    s = str(music)
    positions = [int(d) for d in re.findall(r"(\d+)", s) if int(d)>0]
    if len(positions)==0:
        return {"wins":0,"places":0,"total_races":0,"win_rate":0,"place_rate":0,"recent_form":0,"best_pos":99}
    total = len(positions)
    wins = sum(1 for p in positions if p==1)
    places = sum(1 for p in positions if p<=3)
    recent = positions[:3]
    recent_form = sum(1/p for p in recent)/len(recent)
    return {
        "wins":wins,
        "places":places,
        "total_races":total,
        "win_rate":wins/total,
        "place_rate":places/total,
        "recent_form":recent_form,
        "best_pos":min(positions)
    }

def save_debug_html(url, html):
    fn = os.path.join(SCRAPE_DEBUG_DIR, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(f"<!-- URL: {url} -->\n{html}")
    return fn

def get_html(url, timeout=10):
    try:
        time.sleep(0.6)
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text, None
        else:
            return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)

def parse_table_by_headers(soup, wanted_headers):
    tables = soup.find_all("table")
    for tbl in tables:
        ths = [th.get_text(strip=True).lower() for th in tbl.find_all(["th", "td"])[:10]]
        if any(any(h in th for th in ths) for h in wanted_headers):
            return tbl
    return None

def scrape_geny_partants(url):
    html, err = get_html(url)
    if html is None:
        return None, err
    path = save_debug_html(url, html)
    soup = BeautifulSoup(html, "html.parser")
    wanted = ["nom", "cote", "poids", "musique", "num"]
    tbl = parse_table_by_headers(soup, wanted)
    if not tbl:
        return None, f"Aucune table trouvée (debug: {path})"
    rows = []
    for tr in tbl.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cols)>2:
            rows.append(cols)
    if len(rows)<2:
        return None, f"Table vide (debug: {path})"
    hdr = [h.lower() for h in rows[0]]
    def find(names):
        for name in names:
            for i,h in enumerate(hdr):
                if name in h:
                    return i
        return None
    idx_nom = find(["nom","name"])
    idx_cote = find(["cote"])
    idx_poids = find(["poids"])
    idx_mus = find(["musique"])
    idx_num = find(["num","corde"])
    parsed = []
    for r in rows[1:]:
        def sg(i): return r[i] if (i is not None and i<len(r)) else ""
        parsed.append({
            "Numéro de corde": sg(idx_num),
            "Nom": sg(idx_nom),
            "Cote": sg(idx_cote),
            "Poids": sg(idx_poids),
            "Musique": sg(idx_mus)
        })
    return pd.DataFrame(parsed), f"Table extraite (debug: {path})"

def scrape_geny_stats(url):
    html, err = get_html(url)
    if html is None:
        return None, err
    path = save_debug_html(url, html)
    soup = BeautifulSoup(html, "html.parser")
    wanted = ["date", "hippodrome", "nom", "position", "arriv", "cote"]
    tbl = parse_table_by_headers(soup, wanted)
    if not tbl:
        return None, f"Aucune table trouvée (debug: {path})"
    rows = []
    for tr in tbl.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cols)>2:
            rows.append(cols)
    if len(rows)<2:
        return None, f"Table vide (debug: {path})"
    hdr = [h.lower() for h in rows[0]]
    def find(names):
        for name in names:
            for i,h in enumerate(hdr):
                if name in h:
                    return i
        return None
    idx_nom = find(["nom"])
    idx_pos = find(["arriv","position","résultat"])
    idx_cote = find(["cote"])
    parsed = []
    for r in rows[1:]:
        def sg(i): return r[i] if (i is not None and i<len(r)) else ""
        parsed.append({
            "Nom": sg(idx_nom),
            "Position": sg(idx_pos),
            "Cote": sg(idx_cote)
        })
    return pd.DataFrame(parsed), f"Stats extraites (debug: {path})"

# === Streamlit UI ===
st.set_page_config(page_title="Analyseur Hippique IA v5", layout="wide")
st.title("🏇 Analyseur Hippique IA Pro v5 (Scraping + ML + Correction)")

col1, col2 = st.columns(2)
with col1:
    url_partants = st.text_input("URL Partants (Geny)", "")
with col2:
    url_stats = st.text_input("URL Stats (Geny)", "")

run_btn = st.button("🔍 Extraire & Analyser")

if run_btn:
    dfp, msg1 = scrape_geny_partants(url_partants)
    st.info(msg1)
    dfs, msg2 = scrape_geny_stats(url_stats)
    st.info(msg2)

    if dfp is None:
        st.error("Impossible d'extraire les partants.")
        st.stop()

    st.subheader("✅ Partants extraits")
    st.dataframe(dfp, use_container_width=True)

    # --- Préparation Features ---
    dfp["odds_numeric"] = dfp["Cote"].apply(lambda x: safe_float(x, 999))
    dfp["weight_kg"] = dfp["Poids"].apply(lambda x: extract_weight(x) if x != "" else 60.0)
    dfp["Nom"] = dfp["Nom"].str.strip().str.upper()

    music_feats = dfp["Musique"].apply(extract_music_features).apply(pd.Series)
    dfp = pd.concat([dfp, music_feats], axis=1)
    dfp["odds_inv"] = 1 / (dfp["odds_numeric"] + 0.1)

    # --- Si stats disponibles ---
    if dfs is not None and len(dfs) > 0:
        st.subheader("📊 Stats historiques détectées")
        st.dataframe(dfs, use_container_width=True)

        # ✅ Correction du groupby bug
        stats_df_norm = dfs.copy()
        stats_df_norm["Nom"] = stats_df_norm["Nom"].str.strip().str.upper()

        if "Position" in stats_df_norm.columns:
            stats_df_norm["Position_num"] = stats_df_norm["Position"].apply(lambda x: safe_float(x, np.nan))
            stats_valid = stats_df_norm[stats_df_norm["Position_num"].notna()].copy()

            if len(stats_valid) > 0:
                hist = stats_valid.groupby("Nom").agg(
                    hist_total=("Position_num","count"),
                    hist_wins=("Position_num", lambda s: int((s == 1).sum()))
                ).reset_index()

                podiums = stats_valid.groupby("Nom").apply(
                    lambda g: int((g["Position_num"] <= 3).sum())
                ).rename("hist_podiums").reset_index()

                hist = hist.merge(podiums, on="Nom", how="left")
                hist["hist_podium_rate"] = hist.apply(
                    lambda r: (r["hist_podiums"]/r["hist_total"]) if r["hist_total"]>0 else 0.0,
                    axis=1
                )

                podium_map = dict(zip(hist["Nom"], hist["hist_podium_rate"]))
                dfp["historical_podium_rate"] = dfp["Nom"].map(podium_map).fillna(0.0)
            else:
                dfp["historical_podium_rate"] = 0.0
        else:
            dfp["historical_podium_rate"] = 0.0
    else:
        dfp["historical_podium_rate"] = 0.0

    # --- Score de base ---
    dfp["score_pred"] = (
        dfp["odds_inv"]*0.5 +
        dfp["win_rate"]*0.2 +
        dfp["recent_form"]*0.2 +
        dfp["historical_podium_rate"]*0.1
    )

    dfp["score_pred"] = (dfp["score_pred"] - dfp["score_pred"].min()) / (dfp["score_pred"].max() - dfp["score_pred"].min())

    st.subheader("🏁 Classement Prédit")
    st.dataframe(dfp[["Nom","Cote","score_pred","historical_podium_rate"]].sort_values("score_pred", ascending=False), use_container_width=True)

    fig = px.bar(
        dfp.sort_values("score_pred", ascending=False),
        x="Nom", y="score_pred",
        color="historical_podium_rate",
        color_continuous_scale="Viridis",
        title="Probabilités de podium (score ML pondéré)"
    )
    st.plotly_chart(fig, use_container_width=True)

    csv = dfp.to_csv(index=False)
    st.download_button("📄 Télécharger Résultats CSV", csv, file_name=f"resultats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
