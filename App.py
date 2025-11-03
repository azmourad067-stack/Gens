"""
🏇 Analyseur Hippique IA Pro v4 — scraping Geny + DL/logistic pipeline
- Streamlit app
- Ajoute : saisie d'URL Geny (partants) + URL stats (historique)
- Compatible Pydroid (dépendances : requests, beautifulsoup4, pandas, numpy, sklearn, tensorflow, joblib, plotly)
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

# HTTP / Scraping
import requests
from bs4 import BeautifulSoup

# ML / DL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Plotting
import plotly.express as px

# ----------------------------
# Configs: file paths
# ----------------------------
MODEL_PATH = "hippo_model.h5"
SCALER_PATH = "hippo_scaler.joblib"
CALIB_PATH = "hippo_calibrator.joblib"
SCRAPE_DEBUG_DIR = "scrape_debug"
os.makedirs(SCRAPE_DEBUG_DIR, exist_ok=True)

# ----------------------------
# Utils : parsing helpers
# ----------------------------
def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace(',', '.'))
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return np.nan
    m = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(m.group(1).replace(',', '.')) if m else np.nan

def extract_music_features(music):
    """Return dict from music string (ex: '1a2a3a1a' or '1-2-3')"""
    if pd.isna(music) or str(music).strip() == '':
        return {'wins':0,'places':0,'total_races':0,'win_rate':0.0,'place_rate':0.0,'recent_form':0.0,'best_pos':99,'avg_pos':np.nan}
    s = str(music)
    # keep digits groups (positions)
    positions = [int(d) for d in re.findall(r'(\d+)', s) if int(d)>0]
    if len(positions) == 0:
        return {'wins':0,'places':0,'total_races':0,'win_rate':0.0,'place_rate':0.0,'recent_form':0.0,'best_pos':99,'avg_pos':np.nan}
    total = len(positions)
    wins = sum(1 for p in positions if p==1)
    places = sum(1 for p in positions if p<=3)
    recent = positions[:3]
    recent_form = sum(1.0/p for p in recent)/len(recent) if len(recent)>0 else 0.0
    return {
        'wins': wins,
        'places': places,
        'total_races': total,
        'win_rate': wins/total,
        'place_rate': places/total,
        'recent_form': recent_form,
        'best_pos': min(positions),
        'avg_pos': float(np.mean(positions))
    }

# ----------------------------
# Scraper functions for Geny
# ----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def save_debug_html(url, html):
    fn = os.path.join(SCRAPE_DEBUG_DIR, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(f"<!-- URL: {url} -->\n")
        f.write(html)
    return fn

def get_html(url, timeout=10):
    """Get HTML with retries and polite delay. Returns (html, error)"""
    try:
        time.sleep(0.6)  # small delay to be polite
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.text, None
        else:
            return None, f"HTTP {resp.status_code}"
    except Exception as e:
        return None, str(e)

def parse_table_by_headers(soup, wanted_headers):
    """Generic: find a table where header texts contain the wanted headers (subset)."""
    tables = soup.find_all('table')
    for tbl in tables:
        ths = [th.get_text(strip=True).lower() for th in tbl.find_all(['th','td'])[:10]]
        # quick check: if any wanted header appears in first row text
        if any(any(h in th for th in ths) for h in wanted_headers):
            return tbl
    return None

def scrape_geny_partants(url):
    """
    Attempt to extract partants (participants) from a Geny 'partants' page.
    Returns (df, message).
    Expected columns: 'Numéro de corde' (or 'N°'), 'Nom', 'Cote', 'Poids', 'Musique', 'Âge/Sexe' (if present)
    """
    html, error = get_html(url)
    if html is None:
        return None, f"Erreur téléchargement: {error}"
    # save debug
    debug_path = save_debug_html(url, html)
    soup = BeautifulSoup(html, 'html.parser')

    # strategy 1: look for semantic JSON-LD or script with 'participants' or 'partants'
    # Many sites embed JSON in scripts; attempt to extract embedded JSON arrays
    scripts = soup.find_all('script', type=None)
    for sc in scripts:
        if sc.string and ('partant' in sc.string.lower() or 'participant' in sc.string.lower() or 'dataLayer' in sc.string):
            txt = sc.string
            # try to find a JSON array/object within
            json_matches = re.findall(r'(\{.*\}|\[.*\])', txt, flags=re.S)
            for jm in json_matches:
                try:
                    data = json.loads(jm)
                    # try to heuristically find participant list
                    if isinstance(data, dict):
                        for k,v in data.items():
                            if isinstance(v, list) and len(v)>0 and isinstance(v[0], dict):
                                # check for name/cote keys
                                if any(key in v[0].keys() for key in ['nom','name','cote','odds','starting_position']):
                                    # build df
                                    df = pd.DataFrame(v)
                                    return df, f"Extrait depuis JSON script (heuristique). Debug: {debug_path}"
                except:
                    continue

    # strategy 2: find obvious tables
    wanted_headers = ['nom', 'cote', 'poids', 'musique', 'num', 'numéro', 'corde', 'age']
    tbl = parse_table_by_headers(soup, wanted_headers)
    if tbl:
        # parse rows
        rows = []
        for tr in tbl.find_all('tr'):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
            if not cols:
                continue
            rows.append(cols)
        # heuristics: find header row to map fields
        if len(rows) >= 2:
            header_row = rows[0]
            # try to find indices
            hdrs = [h.lower() for h in header_row]
            # build mapping
            def find_idx(possible_names):
                for name in possible_names:
                    for i,h in enumerate(hdrs):
                        if name in h:
                            return i
                return None
            idx_num = find_idx(['num', 'n°', 'numéro', 'corde'])
            idx_nom = find_idx(['nom','name'])
            idx_cote = find_idx(['cote','odds'])
            idx_poids = find_idx(['poids','poids','weight'])
            idx_musique = find_idx(['musique','musique','music','musiq'])
            idx_age = find_idx(['âge','age','ans'])
            # build df from remaining rows (skip header)
            parsed = []
            for r in rows[1:]:
                # guard indexes
                def safe_get(i):
                    return r[i] if (i is not None and i < len(r)) else ''
                parsed.append({
                    'Numéro de corde': safe_get(idx_num) or '',
                    'Nom': safe_get(idx_nom) or safe_get(1) or '',
                    'Cote': safe_get(idx_cote) or '',
                    'Poids': safe_get(idx_poids) or '',
                    'Musique': safe_get(idx_musique) or '',
                    'Âge/Sexe': safe_get(idx_age) or ''
                })
            df = pd.DataFrame(parsed)
            return df, f"Extrait depuis tableau HTML. Debug: {debug_path}"

    # strategy 3: look for lists/divs of participants (some sites use repeated divs)
    # look for cards containing 'Cote' or '%'
    cards = soup.find_all(lambda tag: tag.name in ['div','li'] and tag.get_text().lower().count('cote')>0)
    if cards:
        parsed = []
        for c in cards:
            text = c.get_text(" ", strip=True)
            # heuristics to extract name, cote, weight
            name = ''
            cote = ''
            poids = ''
            musique = ''
            age_sex = ''
            # try to extract cote pattern like 4.2 or 4/1 or 4,2
            m_cote = re.search(r'(\d+[.,]?\d*(?:\/\d+)?)', text)
            if m_cote:
                cote = m_cote.group(1)
            # name: first words before cote
            parts = text.split()
            if parts:
                name = parts[0]
            parsed.append({'Numéro de corde':'', 'Nom': name, 'Cote': cote, 'Poids': poids, 'Musique': musique, 'Âge/Sexe': age_sex})
        df = pd.DataFrame(parsed)
        return df, f"Extrait depuis blocs/divs heuristiques. Debug: {debug_path}"

    return None, f"Aucun partant trouvé (debug saved: {debug_path}). Structure inattendue."

def scrape_geny_stats(url):
    """
    Attempt to extract race historical results from a Geny 'stats/ancien' page.
    Returns (df_results, message) where df_results contains historical rows with at least columns:
      'Date', 'Hippodrome', 'Nom', 'Position', 'Cote', 'Musique' (if available)
    """
    html, error = get_html(url)
    if html is None:
        return None, f"Erreur téléchargement: {error}"
    debug_path = save_debug_html(url, html)
    soup = BeautifulSoup(html, 'html.parser')

    # Strategy: find tables with headers like Date/Hippodrome/Arrivée/Nom/Cote
    wanted_headers = ['date', 'hippodrome', 'arriv', 'résultat', 'resultat', 'nom', 'cote', 'musique']
    tbl = parse_table_by_headers(soup, wanted_headers)
    if tbl:
        rows = []
        for tr in tbl.find_all('tr'):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
            if not cols:
                continue
            rows.append(cols)
        if len(rows) >= 2:
            header_row = rows[0]
            hdrs = [h.lower() for h in header_row]
            def find_idx(possible_names):
                for name in possible_names:
                    for i,h in enumerate(hdrs):
                        if name in h:
                            return i
                return None
            idx_date = find_idx(['date'])
            idx_hip = find_idx(['hippodrome','lieu'])
            idx_res = find_idx(['arriv','résultat','resultat','rang'])
            idx_nom = find_idx(['nom','cheval'])
            idx_cote = find_idx(['cote','odds'])
            idx_mus = find_idx(['musique'])
            parsed = []
            for r in rows[1:]:
                def safe_get(i):
                    return r[i] if (i is not None and i < len(r)) else ''
                parsed.append({
                    'Date': safe_get(idx_date),
                    'Hippodrome': safe_get(idx_hip),
                    'Nom': safe_get(idx_nom),
                    'Position': safe_get(idx_res),
                    'Cote': safe_get(idx_cote),
                    'Musique': safe_get(idx_mus)
                })
            df = pd.DataFrame(parsed)
            return df, f"Extrait historique depuis tableau HTML. Debug: {debug_path}"

    # Fallback: try to search for blocks with date + result info
    lines = soup.get_text("\n", strip=True).split("\n")
    hist_lines = [l for l in lines if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', l)]
    if hist_lines:
        parsed = []
        for line in hist_lines[:200]:
            # attempt to find date and a name + position
            date_m = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', line)
            if date_m:
                date = date_m.group(1)
            else:
                date = ''
            # find position digit like "1", "2"
            pos_m = re.search(r'\b([1-9]|1[0-9])\b', line)
            pos = pos_m.group(1) if pos_m else ''
            parsed.append({'Date': date, 'Hippodrome': '', 'Nom': line[:50], 'Position': pos, 'Cote': '', 'Musique': ''})
        df = pd.DataFrame(parsed)
        return df, f"Extrait historique heuristique du texte (debug: {debug_path})"

    return None, f"Aucun historique trouvé (debug saved: {debug_path})."

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Analyseur Hippique + Geny Scraper", layout="wide")
st.title("🏇 Analyseur Hippique IA Pro — Scraping Geny intégré")

st.markdown("""
Colle ici l'URL **partants** (liste des partants) et l'URL **stats** (historique de la course).
Si Geny bloque le scraping, tu peux télécharger la page HTML et l'importer (option 'Upload HTML').
""")

col1, col2 = st.columns(2)
with col1:
    url_partants = st.text_input("URL - Partants (Geny)", placeholder="https://www.geny.com/partants-pmu/...")
    uploaded_html_partants = st.file_uploader("Ou uploade le HTML de la page Partants", type=["html","htm"], key="p")
with col2:
    url_stats = st.text_input("URL - Stats/Historique (Geny)", placeholder="https://www.geny.com/stats-pmu?id_course=...")
    uploaded_html_stats = st.file_uploader("Ou uploade le HTML de la page Stats", type=["html","htm"], key="s")

analyse_btn = st.button("🔍 Extraire & Analyser (partants + stats)")

# If user uploaded HTMLs, write them to temp files and use parsing on soup directly
def parse_html_from_upload(uploaded_file):
    try:
        raw = uploaded_file.read().decode('utf-8', errors='ignore')
    except:
        raw = uploaded_file.read().decode('latin-1', errors='ignore')
    return raw

# result containers
partants_df = None
stats_df = None
messages = []

if analyse_btn:
    # PARTANTS
    if uploaded_html_partants is not None:
        html_content = parse_html_from_upload(uploaded_html_partants)
        debug_path = save_debug_html("uploaded_partants", html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        # try parse via same functions but passing soup -> hack: write soup to temp and call parsing logic expecting URL
        # quick hack: create a temporary function that uses parse_table_by_headers
        # We'll reuse scrape_geny_partants by saving HTML to a temp file and calling get_html? Simpler: call parsing logic inline:
        tbl = parse_table_by_headers(soup, ['nom','cote','poids','musique'])
        if tbl:
            rows = []
            for tr in tbl.find_all('tr'):
                cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
                if len(cols) < 2:
                    continue
                rows.append(cols)
            if len(rows) >= 2:
                header = [h.lower() for h in rows[0]]
                def idx_for(names):
                    for i,h in enumerate(header):
                        for n in names:
                            if n in h:
                                return i
                    return None
                inum = idx_for(['num','n°','corde'])
                inom = idx_for(['nom','name'])
                ico = idx_for(['cote'])
                ipoi = idx_for(['poids','weight'])
                imus = idx_for(['musique'])
                parsed = []
                for r in rows[1:]:
                    def sg(i):
                        return r[i] if (i is not None and i < len(r)) else ''
                    parsed.append({
                        'Numéro de corde': sg(inum),
                        'Nom': sg(inom),
                        'Cote': sg(ico),
                        'Poids': sg(ipoi),
                        'Musique': sg(imus),
                        'Âge/Sexe': ''
                    })
                partants_df = pd.DataFrame(parsed)
                messages.append(f"Partants extraits depuis HTML upload (table). Debug: {debug_path}")
            else:
                messages.append(f"HTML upload - table non structurée. Debug saved: {debug_path}")
        else:
            messages.append(f"HTML upload - aucun tableau détecté. Debug saved: {debug_path}")
    elif url_partants:
        partants_df, msg = scrape_geny_partants(url_partants)
        messages.append(msg)
    else:
        messages.append("Aucun URL / fichier Partants fourni.")

    # STATS
    if uploaded_html_stats is not None:
        html_content = parse_html_from_upload(uploaded_html_stats)
        debug_path = save_debug_html("uploaded_stats", html_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        tbl = parse_table_by_headers(soup, ['date','hippodrome','nom','cote','musique'])
        if tbl:
            rows = []
            for tr in tbl.find_all('tr'):
                cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
                if len(cols) < 2:
                    continue
                rows.append(cols)
            if len(rows) >= 2:
                header = [h.lower() for h in rows[0]]
                def idx_for(names):
                    for i,h in enumerate(header):
                        for n in names:
                            if n in h:
                                return i
                    return None
                idate = idx_for(['date'])
                ihip = idx_for(['hippodrome','lieu'])
                inom = idx_for(['nom'])
                ipos = idx_for(['arriv','résultat','rang'])
                ico = idx_for(['cote'])
                imus = idx_for(['musique'])
                parsed = []
                for r in rows[1:]:
                    def sg(i):
                        return r[i] if (i is not None and i < len(r)) else ''
                    parsed.append({
                        'Date': sg(idate),
                        'Hippodrome': sg(ihip),
                        'Nom': sg(inom),
                        'Position': sg(ipos),
                        'Cote': sg(ico),
                        'Musique': sg(imus)
                    })
                stats_df = pd.DataFrame(parsed)
                messages.append(f"Stats extraits depuis HTML upload (table). Debug: {debug_path}")
            else:
                messages.append(f"HTML upload stats - table non structurée. Debug saved: {debug_path}")
        else:
            messages.append(f"HTML upload stats - aucun tableau détecté. Debug saved: {debug_path}")
    elif url_stats:
        stats_df, msg = scrape_geny_stats(url_stats)
        messages.append(msg)
    else:
        messages.append("Aucun URL / fichier Stats fourni.")

    st.write("### Résumé extraction")
    for m in messages:
        st.info(m)

    # show extracted
    if partants_df is not None:
        st.subheader("Partants extraits")
        st.dataframe(partants_df.head(50), use_container_width=True)
    else:
        st.warning("Aucun partant extrait — fournir HTML ou vérifier l'URL.")

    if stats_df is not None:
        st.subheader("Historique / Stats extraits")
        st.dataframe(stats_df.head(50), use_container_width=True)
    else:
        st.info("Aucun historique extrait (optionnel).")

    # If we have partants, prepare dataframe for ML pipeline
    if partants_df is not None:
        st.subheader("Préparation automatique des features (à partir des partants et, si fournis, des stats)")
        dfp = partants_df.copy()

        # normalize keys that may come in different languages
        # heuristics: find columns matching expected names
        colmap = {}
        for c in dfp.columns:
            lc = c.lower()
            if 'nom' in lc or 'name' in lc:
                colmap[c] = 'Nom'
            elif 'cote' in lc or 'odds' in lc:
                colmap[c] = 'Cote'
            elif 'poids' in lc or 'weight' in lc:
                colmap[c] = 'Poids'
            elif 'musique' in lc or 'music' in lc:
                colmap[c] = 'Musique'
            elif 'age' in lc:
                colmap[c] = 'Âge/Sexe'
            elif 'corde' in lc or 'num' in lc:
                colmap[c] = 'Numéro de corde'
        dfp = dfp.rename(columns=colmap)

        # ensure required cols
        for req in ['Nom','Cote','Poids','Musique','Numéro de corde','Âge/Sexe']:
            if req not in dfp.columns:
                dfp[req] = ''

        # numerics
        dfp['odds_numeric'] = dfp['Cote'].apply(lambda x: safe_float(x, default=999))
        dfp['draw_numeric'] = dfp['Numéro de corde'].apply(lambda x: int(re.sub(r'\D','',str(x)) if re.sub(r'\D','',str(x)) else 1))
        dfp['weight_kg'] = dfp['Poids'].apply(lambda x: extract_weight(x) if x!='' else np.nan).fillna(60.0)

        # extract music features
        music_feats = dfp['Musique'].apply(extract_music_features).apply(pd.Series)
        dfp = pd.concat([dfp, music_feats], axis=1)

        # if stats_df provided, compute simple aggregated driver/hippo features (heuristic)
        # We won't attempt complex joins here unless explicit columns exist.
        if stats_df is not None:
            # compute horse-level historical win percent on this race id
            # We'll compute per-horse historic podium rate from stats_df
            stats_df_norm = stats_df.copy()
            if 'Nom' in stats_df_norm.columns and 'Position' in stats_df_norm.columns:
                stats_df_norm['Position_num'] = stats_df_norm['Position'].apply(lambda x: safe_float(x, default=np.nan))
                hist = stats_df_norm.groupby('Nom').agg(
                    hist_total = ('Position_num','count'),
                    hist_wins = (lambda s: (s==1).sum()) if 'Position_num' in stats_df_norm else ('Position_num','count'),
                )
                # simpler: compute podium rate
                # careful: stats_df may contain many races; do best effort
                podium_rate = {}
                for name, group in stats_df_norm.groupby('Nom'):
                    tot = len(group)
                    podium = ((group['Position_num']<=3).sum()) if 'Position_num' in group else 0
                    podium_rate[name] = podium / tot if tot>0 else 0.0
                dfp['historical_podium_rate'] = dfp['Nom'].map(podium_rate).fillna(0.0)
        else:
            dfp['historical_podium_rate'] = 0.0

        # show prepared df
        display_cols = ['Numéro de corde','Nom','Cote','odds_numeric','weight_kg','wins','places','total_races','win_rate','place_rate','recent_form','historical_podium_rate']
        for c in display_cols:
            if c not in dfp.columns:
                dfp[c] = np.nan
        st.dataframe(dfp[display_cols].head(50), use_container_width=True)

        # Ask user: run quick logistic or deep model?
        st.markdown("###  ▶️ Choix modèle et exécution rapide")
        colA, colB = st.columns(2)
        with colA:
            model_choice = st.selectbox("Type de modèle", ["Logistic (rapide)","Deep Learning (Keras)"])
        with colB:
            run_predict = st.button("Lancer prédiction (sur ces partants)")

        if run_predict:
            # build minimal features for model: odds_inv, recent_form, win_rate, weight, historical_podium_rate
            dfp['odds_inv'] = 1.0 / (dfp['odds_numeric'] + 0.1)
            # choose features present
            candidate_features = ['odds_inv','win_rate','place_rate','recent_form','weight_kg','historical_podium_rate']
            feats = [f for f in candidate_features if f in dfp.columns]
            X_new = dfp[feats].fillna(0.0)

            # If model choice logistic -> train quick logistic if no saved model
            if model_choice == "Logistic (rapide)":
                st.info("Entraînement d'une régression logistique local (nécessite historiques avec 'Resultat' pour labels).")
                # If stats_df contains Resultat rows for historical training, we can train a logistic quickly.
                # We'll attempt to build a tiny training set from stats_df if present.
                trained = False
                if stats_df is not None and 'Position' in stats_df.columns:
                    # Build small training dataset: each unique historical horse-row -> features (if we can parse Cote/Musique there)
                    st.info("Tentative d'utiliser la page 'stats' comme historique pour entraîner un modèle simple.")
                    # Build history_df with features if possible
                    hist = stats_df.copy()
                    # normalize column names
                    hist_cols = [c.lower() for c in hist.columns]
                    # try to extract odds if present
                    if 'Cote' in stats_df.columns or any('cote' in c.lower() for c in stats_df.columns):
                        # we may not have same features; as fallback, train on odds only
                        hist['Position_num'] = hist['Position'].apply(lambda x: safe_float(x, default=np.nan))
                        hist['target'] = (hist['Position_num'] <= 3).astype(int)
                        if 'Cote' in hist.columns:
                            hist['odds_numeric'] = hist['Cote'].apply(lambda x: safe_float(x, default=np.nan)).fillna(hist['Cote'].median() if 'Cote' in hist.columns else 5.0)
                        else:
                            hist['odds_numeric'] = 5.0
                        hist['odds_inv'] = 1.0 / (hist['odds_numeric'] + 0.1)
                        # Minimal training with odds_inv only
                        X_hist = hist[['odds_inv']].fillna(0.0)
                        y_hist = hist['target']
                        if len(y_hist.unique())>1 and len(X_hist)>10:
                            scaler_tmp = StandardScaler()
                            Xh = scaler_tmp.fit_transform(X_hist)
                            clf = LogisticRegression(max_iter=500)
                            clf.fit(Xh, y_hist)
                            # predict for current X_new (use same scaler)
                            X_new_scaled = scaler_tmp.transform(X_new[['odds_inv']]) if 'odds_inv' in X_new.columns else scaler_tmp.transform(X_new.values)
                            preds = clf.predict_proba(X_new_scaled)[:,1]
                            dfp['score_pred'] = preds
                            st.success("Prédictions (modèle log entraîné sur historique disponible).")
                            st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)
                            trained = True
                if not trained:
                    st.warning("Historique insuffisant pour entraîner un modèle fiable. On renvoie un score basique basé sur cotes.")
                    dfp['score_pred'] = (1.0 / (dfp['odds_numeric'] + 0.1))
                    # normalize
                    if dfp['score_pred'].max() != dfp['score_pred'].min():
                        dfp['score_pred'] = (dfp['score_pred'] - dfp['score_pred'].min()) / (dfp['score_pred'].max() - dfp['score_pred'].min())
                    st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)

            else:
                # Deep Learning option
                st.info("Option Deep Learning : on utilisera un modèle Keras si disponible (load) sinon entraînement local.")
                model_loaded = None
                scaler_loaded = None
                calibrator = None
                if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                    try:
                        model_loaded = load_model(MODEL_PATH)
                        scaler_loaded = joblib.load(SCALER_PATH)
                        if os.path.exists(CALIB_PATH):
                            calibrator = joblib.load(CALIB_PATH)
                        st.success("Modèle Keras & scaler chargés depuis disque.")
                    except Exception as e:
                        st.error(f"Erreur chargement modèle: {e}")
                        model_loaded = None
                if model_loaded is None:
                    st.info("Aucun modèle pré-entrainé disponible : entraînement rapide sur place (si historique suffisant).")
                    # Attempt to prepare a tiny training set from stats_df (if available)
                    if stats_df is not None and 'Position' in stats_df.columns:
                        hist = stats_df.copy()
                        # build features for hist (very approximate)
                        # require that hist contains Cote and/or Musique to extract features; otherwise bail out
                        if 'Cote' in hist.columns:
                            hist['odds_numeric'] = hist['Cote'].apply(lambda x: safe_float(x, default=np.nan)).fillna(hist['Cote'].median() if 'Cote' in hist.columns else 5.0)
                        else:
                            hist['odds_numeric'] = 5.0
                        # Extract music features if present
                        if 'Musique' in hist.columns:
                            mf = hist['Musique'].apply(extract_music_features).apply(pd.Series)
                            hist = pd.concat([hist.reset_index(drop=True), mf.reset_index(drop=True)], axis=1)
                        else:
                            hist['recent_form'] = 0.0
                            hist['win_rate'] = 0.0
                        hist['odds_inv'] = 1.0 / (hist['odds_numeric'] + 0.1)
                        hist['Position_num'] = hist['Position'].apply(lambda x: safe_float(x, default=np.nan))
                        hist['target'] = (hist['Position_num'] <= 3).astype(int)
                        feat_cols = [c for c in ['odds_inv','recent_form','win_rate'] if c in hist.columns]
                        if len(feat_cols)==0 or len(hist) < 30:
                            st.warning("Historique insuffisant pour entraînement DL local (besoin de >=30 lignes avec features). On renvoie score de base.")
                            dfp['score_pred'] = (1.0 / (dfp['odds_numeric'] + 0.1))
                            if dfp['score_pred'].max() != dfp['score_pred'].min():
                                dfp['score_pred'] = (dfp['score_pred'] - dfp['score_pred'].min()) / (dfp['score_pred'].max() - dfp['score_pred'].min())
                            st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)
                        else:
                            X_hist = hist[feat_cols].fillna(0.0).values
                            y_hist = hist['target'].values
                            # scale
                            scaler_tmp = StandardScaler()
                            X_hist_s = scaler_tmp.fit_transform(X_hist)
                            # build small keras
                            input_dim = X_hist_s.shape[1]
                            m = Sequential([
                                Dense(64, activation='relu', input_shape=(input_dim,)),
                                BatchNormalization(),
                                Dropout(0.2),
                                Dense(32, activation='relu'),
                                Dense(1, activation='sigmoid')
                            ])
                            m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
                            early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
                            m.fit(X_hist_s, y_hist, epochs=80, batch_size=16, validation_split=0.2, callbacks=[early], verbose=0)
                            # predict for current partants
                            X_new_sub = dfp[[c for c in feat_cols if c in dfp.columns]].fillna(0.0).values
                            X_new_s = scaler_tmp.transform(X_new_sub)
                            preds = m.predict(X_new_s).ravel()
                            dfp['score_pred'] = preds
                            st.success("Prédictions réalisées par entraînement DL local (historique disponible).")
                            st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)
                            # optionally save model/scaler for later
                            if st.checkbox("Sauvegarder modèle local (hippo_model.h5 + scaler)", value=False):
                                m.save(MODEL_PATH)
                                joblib.dump(scaler_tmp, SCALER_PATH)
                                st.success("Modèle et scaler sauvegardés.")
                    else:
                        st.warning("Aucun historique exploitable (colonne Position/Resultat manquante). On renvoie score de base.")
                        dfp['score_pred'] = (1.0 / (dfp['odds_numeric'] + 0.1))
                        if dfp['score_pred'].max() != dfp['score_pred'].min():
                            dfp['score_pred'] = (dfp['score_pred'] - dfp['score_pred'].min()) / (dfp['score_pred'].max() - dfp['score_pred'].min())
                        st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)
                else:
                    # model_loaded exists: use it
                    X_new_s = scaler_loaded.transform(X_new.values)
                    raw_preds = model_loaded.predict(X_new_s).ravel()
                    if calibrator is not None:
                        try:
                            preds = calibrator.predict_proba(raw_preds.reshape(-1,1))[:,1]
                        except:
                            preds = raw_preds
                    else:
                        preds = raw_preds
                    dfp['score_pred'] = preds
                    st.success("Prédictions réalisées avec modèle chargé.")
                    st.dataframe(dfp[['Numéro de corde','Nom','Cote','score_pred']].sort_values('score_pred',ascending=False), use_container_width=True)

        # export CSV
        csv_out = dfp.to_csv(index=False)
        st.download_button("Télécharger partants préparés (CSV)", csv_out, file_name=f"partants_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
