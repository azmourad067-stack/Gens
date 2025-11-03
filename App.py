"""
🏇 Analyseur Hippique IA Pro v3 (Deep Learning + Calibration)
- Streamlit + TensorFlow Keras
- Sauvegarde du modèle et scaler (apprentissage continu possible)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import joblib

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# plotting
import plotly.express as px

st.set_page_config(page_title="🏇 Analyseur Hippique IA Pro v3 (DL)", layout="wide")

# -------------------------
# Helpers : feature extraction, weight parsing
# -------------------------
def extract_music_features(music):
    if pd.isna(music) or music == "":
        return {'win_rate': 0.0, 'place_rate': 0.0, 'recent_form': 0.0, 'total_races': 0}
    s = str(music)
    positions = [int(ch) for ch in s if ch.isdigit() and int(ch) > 0]
    if len(positions) == 0:
        return {'win_rate': 0.0, 'place_rate': 0.0, 'recent_form': 0.0, 'total_races': 0}
    total = len(positions)
    wins = positions.count(1)
    places = sum(1 for p in positions if p <= 3)
    recent = positions[:3]
    recent_form = sum(1/p for p in recent) / len(recent) if len(recent) > 0 else 0.0
    return {
        'win_rate': wins / total,
        'place_rate': places / total,
        'recent_form': recent_form,
        'total_races': total
    }

def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(',', '.'))
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return 60.0
    m = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(m.group(1).replace(',', '.')) if m else 60.0

# -------------------------
# File / model paths
# -------------------------
MODEL_PATH = "hippo_model.h5"
SCALER_PATH = "hippo_scaler.joblib"
CALIB_PATH = "hippo_calibrator.joblib"

# -------------------------
# UI : Load data
# -------------------------
st.title("🏇 Analyseur Hippique IA Pro v3 — Deep Learning + Calibration")
st.caption("But : prédire la probabilité de podium (arrivée ≤ 3) — modèle supervisé, calibré et sauvegardable.")

col_a, col_b = st.columns([2,1])
with col_a:
    uploaded = st.file_uploader("Charger un CSV (colonnes recommandées: Nom, Cote, Poids, Musique, Age, Sexe, Driver, Hippodrome, Distance, Terrain, Resultat)", type=["csv"])
with col_b:
    if st.button("Charger exemple"):
        uploaded = "EXAMPLE"

if uploaded is None:
    st.info("Charge un fichier ou clique sur 'Charger exemple' pour continuer.")
    st.stop()

# Example dataset
if uploaded == "EXAMPLE":
    df = pd.DataFrame({
        "Nom": ["Thunder Bolt", "Wind Walker", "Rain Dance", "Ocean Wave", "Storm King", "Fire Dancer", "Lightning Star"],
        "Cote": [3.2, 4.8, 7.5, 6.2, 9.1, 12.5, 15.0],
        "Poids": ['56.5','57.0','58.5','59.0','57.5','60.0','61.5'],
        "Musique": ['1a2a3a1a','2a1a4a3a','3a3a1a2a','1a4a2a1a','4a2a5a3a','5a3a6a4a','6a5a7a8a'],
        "Age": [4,5,3,6,4,5,4],
        "Sexe": ['H','M','F','H','M','H','F'],
        "Driver": ['Dupont','Martin','Dupont','Leclerc','Martin','Durand','Durand'],
        "Hippodrome": ['ParisLong','ParisLong','Lyon','Lyon','ParisLong','Nice','Nice'],
        "Distance": [1500,1500,1600,1500,1500,1600,1500],
        "Terrain": ['Bon','Bon','Souple','Bon','Bon','Souple','Bon'],
        "Resultat": [1,2,5,3,6,4,7]
    })
else:
    df = pd.read_csv(uploaded)

st.write(f"✅ {len(df)} lignes chargées")
st.dataframe(df.head(), use_container_width=True)

# -------------------------
# Feature engineering
# -------------------------
st.subheader("🔧 Préprocessing & Feature Engineering")

# Musique features
music_df = df['Musique'].apply(extract_music_features).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), music_df.reset_index(drop=True)], axis=1)

# Numeric conversions
df['odds'] = df['Cote'].apply(lambda x: safe_float(x, default=np.nan))
df['odds'].fillna(df['odds'].median(), inplace=True)
df['odds_inv'] = 1.0 / (df['odds'] + 0.1)

df['weight_kg'] = df.get('Poids', df.get('Poids', '')).apply(extract_weight) if 'Poids' in df.columns else 60.0
df['Age_num'] = df['Age'].apply(lambda x: safe_float(x, default=4)) if 'Age' in df.columns else 4.0

# Jockey/Driver stats (basic aggregation if present)
if 'Driver' in df.columns:
    driver_stats = df.groupby('Driver').agg(
        driver_races = ('Nom','count'),
        driver_wins = (lambda s: (df.loc[s.index,'Resultat'] == 1).sum()) if 'Resultat' in df.columns else ('Nom','count')
    )
    # compute driver win rate per driver and map
    driver_win_rate = {}
    for driver, group in df.groupby('Driver'):
        res = group.get('Resultat', pd.Series(dtype=int))
        wins = (res == 1).sum() if 'Resultat' in group else 0
        tot = len(group)
        driver_win_rate[driver] = wins / tot if tot>0 else 0.0
    df['driver_win_rate'] = df['Driver'].map(driver_win_rate).fillna(0.0)
else:
    df['driver_win_rate'] = 0.0

# Hippodrome stats (if historical included)
if 'Hippodrome' in df.columns and 'Resultat' in df.columns:
    hippo_win_rate = {}
    for hip, group in df.groupby('Hippodrome'):
        wins = (group['Resultat'] == 1).sum()
        tot = len(group)
        hippo_win_rate[hip] = wins / tot if tot>0 else 0.0
    df['hippo_win_rate'] = df['Hippodrome'].map(hippo_win_rate).fillna(0.0)
else:
    df['hippo_win_rate'] = 0.0

# Target
if 'Resultat' not in df.columns:
    st.error("La colonne 'Resultat' est requise pour l'entraînement (valeur entière du rang).")
    st.stop()
df['target'] = (df['Resultat'] <= 3).astype(int)

# Build feature set (extendable)
base_features = [
    'odds_inv', 'win_rate', 'place_rate', 'recent_form',
    'weight_kg', 'Age_num', 'driver_win_rate', 'hippo_win_rate', 'Distance'
]
# keep only features present
features = [f for f in base_features if f in df.columns]
st.write("Features utilisées :", features)

X = df[features].fillna(0.0)
y = df['target']

# -------------------------
# Correlation-based selection (simple automatic pruning)
# -------------------------
corr = pd.concat([X, y], axis=1).corr()['target'].drop('target').abs().sort_values(ascending=False)
st.subheader("📈 Corrélations absolues (avec target)")
st.dataframe(corr.to_frame("abs_corr"), use_container_width=True)

# keep features with abs(corr) > threshold or top-k
corr_thresh = st.slider("Seuil corrélation minimale (abs)", min_value=0.0, max_value=0.5, value=0.01, step=0.01)
selected_feats = corr[corr >= corr_thresh].index.tolist()
if len(selected_feats) == 0:
    selected_feats = corr.index.tolist()[:min(6, len(corr.index))]
st.info(f"Features retenues pour le modèle: {selected_feats}")
X = X[selected_feats]

# -------------------------
# Split / Standardize
# -------------------------
test_size = st.slider("Fraction test (%)", min_value=10, max_value=40, value=25, step=5)
X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=test_size/100.0, random_state=42, stratify=y.values)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# offer to load existing model
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
st.write("Modèle existant :", "Oui" if model_exists else "Non")
if model_exists:
    if st.button("Charger modèle existant"):
        try:
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            if os.path.exists(CALIB_PATH):
                calibrator = joblib.load(CALIB_PATH)
            else:
                calibrator = None
            st.success("Modèle & scaler chargés.")
        except Exception as e:
            st.error(f"Erreur chargement: {e}")
            model = None
            calibrator = None
else:
    model = None
    calibrator = None

# -------------------------
# Build / Train Keras model (if requested)
# -------------------------
st.subheader("⚙️ Modèle Deep Learning")

train_model_button = st.button("📡 Entraîner modèle DL maintenant")
if train_model_button:
    input_dim = X_train.shape[1]
    # Define model architecture (tunable)
    def build_model(input_dim, units1=64, units2=32, dropout=0.2):
        m = Sequential()
        m.add(Dense(units1, activation='relu', input_shape=(input_dim,)))
        m.add(BatchNormalization())
        m.add(Dropout(dropout))
        m.add(Dense(units2, activation='relu'))
        m.add(BatchNormalization())
        m.add(Dropout(dropout))
        m.add(Dense(1, activation='sigmoid'))
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
        return m

    model = build_model(input_dim=input_dim,
                        units1=st.slider("Units couche 1", 32, 256, 64, step=32),
                        units2=st.slider("Units couche 2", 16, 128, 32, step=16),
                        dropout=st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05))
    st.write(model.summary())

    # callbacks
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size= st.slider("Batch size", 8, 64, 16, step=8),
        callbacks=[early],
        verbose=1
    )

    # save model + scaler
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.success("✅ Modèle entraîné et sauvegardé (model.h5 & scaler.joblib).")

    # Evaluate on validation
    y_val_pred = model.predict(X_val).ravel()
    auc_val = roc_auc_score(y_val, y_val_pred) if len(np.unique(y_val))>1 else float('nan')
    logloss_val = log_loss(y_val, np.clip(y_val_pred, 1e-6, 1-1e-6))
    st.write(f"Validation AUC: {auc_val:.4f} | LogLoss: {logloss_val:.4f}")

    # Calibration (Platt scaling) : logistic regression mapping raw predicted prob -> calibrated prob
    do_cal = st.checkbox("Appliquer calibration Platt (recommandé)", value=True)
    calibrator = None
    if do_cal:
        # train calibrator on validation predictions
        lr = LogisticRegression(max_iter=500)
        lr.fit(y_val_pred.reshape(-1,1), y_val)
        joblib.dump(lr, CALIB_PATH)
        calibrator = lr
        st.success("✅ Calibrateur entraîné et sauvegardé (Platt).")
        # Evaluate calibrated
        y_val_cal = lr.predict_proba(y_val_pred.reshape(-1,1))[:,1]
        auc_cal = roc_auc_score(y_val, y_val_cal) if len(np.unique(y_val))>1 else float('nan')
        logloss_cal = log_loss(y_val, np.clip(y_val_cal,1e-6,1-1e-6))
        st.write(f"Après calibration — AUC: {auc_cal:.4f} | LogLoss: {logloss_cal:.4f}")

# -------------------------
# Predict & Score current data
# -------------------------
st.subheader("📊 Prédictions & Classement")

if model is None:
    st.info("Aucun modèle disponible — entraîne un modèle pour obtenir des prédictions.")
    st.stop()

# prepare entire X for predictions using saved scaler
X_all = X.values
X_all_scaled = scaler.transform(X_all)

raw_preds = model.predict(X_all_scaled).ravel()  # probabilities (before calibration)
if 'calibrator' in locals() and calibrator is not None:
    preds = calibrator.predict_proba(raw_preds.reshape(-1,1))[:,1]
else:
    # attempt to load calibrator from disk if exists
    if os.path.exists(CALIB_PATH):
        calibrator = joblib.load(CALIB_PATH)
        preds = calibrator.predict_proba(raw_preds.reshape(-1,1))[:,1]
    else:
        preds = raw_preds

df['score_pred'] = preds
df['rank_pred'] = df['score_pred'].rank(ascending=False)
df_sorted = df.sort_values('rank_pred').reset_index(drop=True)

st.dataframe(df_sorted[['Nom','Cote','score_pred','rank_pred']].round(3), use_container_width=True)

# Metrics on train/val if available
st.subheader("📈 Diagnostics rapides")
# Compute AUC on full dataset if possible
if len(df['target'].unique())>1:
    try:
        auc_all = roc_auc_score(df['target'], df['score_pred'])
        st.metric("AUC (sur dataset chargé)", f"{auc_all:.3f}")
    except Exception:
        st.info("Impossible de calculer l'AUC (peut nécessiter >=2 classes)")

# Permutation importance (model-agnostic using predictions)
st.markdown("**Importance approximative (permutation sur features retenues)**")
try:
    # use a small subset to compute permutation importance (expensive otherwise)
    sample_idx = np.random.choice(len(X_all_scaled), size=min(200, len(X_all_scaled)), replace=False)
    X_sample = X_all_scaled[sample_idx]
    y_sample = df['target'].values[sample_idx]
    # wrapper predict function mapping X->pred probability
    def predict_fn(x):
        return model.predict(x).ravel()
    # permutation via sklearn requires estimator API; we'll approximate by permuting columns and checking drop in AUC
    base_auc = roc_auc_score(y_sample, model.predict(X_sample).ravel()) if len(np.unique(y_sample))>1 else np.nan
    imp = []
    for i, feat in enumerate(selected_feats):
        X_perm = X_sample.copy()
        np.random.shuffle(X_perm[:, i])
        auc_perm = roc_auc_score(y_sample, model.predict(X_perm).ravel()) if len(np.unique(y_sample))>1 else np.nan
        drop = base_auc - auc_perm if (not np.isnan(base_auc) and not np.isnan(auc_perm)) else 0.0
        imp.append((feat, max(drop,0.0)))
    imp_df = pd.DataFrame(imp, columns=['feature','auc_drop']).sort_values('auc_drop', ascending=False)
    st.dataframe(imp_df, use_container_width=True)
except Exception as e:
    st.info(f"Permutation importance impossible: {e}")

# Top 5 display
st.subheader("🥇 Top 5 pronostics")
for i, r in df_sorted.head(5).iterrows():
    st.markdown(f"**{i+1}. {r['Nom']}** — Cote: {r.get('Cote', 'NA')} — Prob(podium): **{r['score_pred']:.3f}**")

# -------------------------
# Export
# -------------------------
st.subheader("💾 Exporter / Sauvegarder")
csv_output = df_sorted.to_csv(index=False)
st.download_button("Télécharger pronostics (CSV)", csv_output, file_name=f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if st.button("Sauvegarder modèle & scaler (explicit)"):
    try:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.success("Modèle et scaler sauvegardés.")
    except Exception as e:
        st.error(f"Erreur sauvegarde: {e}")

st.info("Fin du pipeline. Pour apprentissage continu: charger de nouvelles courses (CSV), réentraîner sur l'ensemble historique consolidé, puis sauvegarder le modèle mis à jour.")

# -------------------------
# Short explanation of approach & math
# -------------------------
st.markdown("---")
st.header("🔬 Notes méthodologiques (résumé)")
st.markdown("""
**Architecture** : réseau dense (2 couches cachées) → activation ReLU → dropout & batchnorm → sortie sigmoïde.

**Loss** : binary_crossentropy → minimise la log-vraisemblance négative, donc adapte probabilités.

**Calibration (Platt)** : le réseau produit p̂. On entraîne une régression logistique simple `sigma(a * p̂ + b)` sur un jeu de validation pour corriger biais de probabilité (sur/sous-confiance).  
Ceci donne `p_cal = logistic( a * p̂ + b )`.

**Pourquoi c'est meilleur** :
* Le réseau apprend automatiquement interactions non-linéaires (ex : poids × distance).
* La calibration corrige la tendance d'un modèle à être mal-calibré.
* La sélection par corrélation et la standardisation limitent le bruit et stabilisent l'entraînement.

**Recommandations** :
1. Entraîner sur un historique large (des milliers de courses) pour que le réseau généralise.
2. Ajouter features jockey/driver historiques et hippodrome (si disjointes par course, stocker historique séparé).
3. Vérifier l'équilibre classes (si peu de podiums, user stratified sampling ou sur-échantillonnage).
4. Logger chaque course et réentraîner périodiquement (ex: nightly) pour apprentissage continu.
""")
