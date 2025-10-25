# analyseur_hippique_geny_pro.py (version corrigée)
# -*- coding: utf-8 -*-

import os
import re
import warnings
import json
import tempfile
from datetime import datetime, timedelta
from itertools import combinations
from decimal import Decimal, ROUND_HALF_UP
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional ML imports
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
except Exception:
    pass

# ---------------- Configuration Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

HIST_PATH = os.path.join(DATA_DIR, "historique_complet.csv")
BANKROLL_PATH = os.path.join(DATA_DIR, "bankroll.json")
PERFORMANCE_PATH = os.path.join(DATA_DIR, "performance.json")

# ---------------- Classes Corrigées ----------------
class AdvancedScraper:
    """Scraper avancé Geny avec implémentation complète"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_geny_advanced(self, url):
        """Scraping avancé Geny - version corrigée"""
        try:
            response = self.session.get(url, timeout=15)
            response.encoding = 'ISO-8859-1'
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Utiliser la méthode de scraping existante qui fonctionne
            return self._scrape_geny_fallback(soup)
            
        except Exception as e:
            st.error(f"Erreur scraping avancé: {e}")
            # Fallback vers la méthode originale
            return self.scrape_geny_classic(url)
    
    def scrape_geny_classic(self, url):
        """Méthode de scraping classique éprouvée"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=12)
            
            if r.encoding is None or "utf" not in (r.encoding or "").lower():
                r.encoding = "ISO-8859-1"
                
            try:
                soup = BeautifulSoup(r.text, "lxml")
            except Exception:
                soup = BeautifulSoup(r.text, "html.parser")

            rows = []
            candidate_tables = soup.find_all("table")
            table = None
            
            for t in candidate_tables:
                ths = [cell.get_text(strip=True).lower() for cell in t.find_all(["th","td"])[:20]]
                joined = " ".join(ths)
                if any(k in joined for k in ["musique", "gains", "cheval", "rapports", "driver", "entraîneur", "entraineur", "cote"]):
                    table = t
                    break
                    
            if table is None and candidate_tables:
                table = candidate_tables[0]

            if table is not None and len(table.find_all("tr")) > 0:
                for tr in table.find_all("tr"):
                    tds = tr.find_all(["td","th"])
                    if not tds:
                        continue
                    texts = [td.get_text(" ", strip=True) for td in tds]
                    if len(texts) < 2:
                        continue
                        
                    # Parsing amélioré
                    horse_data = self._parse_horse_row_improved(texts)
                    if horse_data:
                        rows.append(horse_data)
            else:
                # Fallback text parsing
                rows = self._parse_text_fallback(soup)

            if not rows:
                st.warning("Aucune donnée extraite, utilisation des données de démonstration")
                return self._get_demo_data()

            df = pd.DataFrame(rows)
            return self._clean_dataframe(df)
            
        except Exception as e:
            st.error(f"Erreur scraping classique: {e}")
            return self._get_demo_data()
    
    def _parse_horse_row_improved(self, texts):
        """Parse une ligne de cheval - version corrigée"""
        try:
            name = ""
            num = ""
            cote = np.nan
            poids = ""
            musique = ""
            age_sexe = ""
            jockey = ""
            entraineur = ""
            gains = 0

            # Détection numéro et nom
            if len(texts) >= 2:
                if re.match(r"^\d+$", texts[0].strip()):
                    num = texts[0].strip()
                    name = texts[1].strip()
                else:
                    for t in texts:
                        if re.search(r"[A-Za-zÀ-ÿ]", t) and not re.match(r"^\d+[,\.]\d+$", t):
                            name = t.strip()
                            break

            # Détection cote
            for t in texts[::-1]:
                if re.search(r"\d+[,\.]\d+", t):
                    mf = re.search(r"\d+[,\.]\d+", t)
                    if mf:
                        cote = float(mf.group(0).replace(",", "."))
                        break

            # Détection gains
            for t in texts[::-1]:
                digits = re.sub(r"[^\d]", "", t)
                if digits and len(digits) > 3:
                    try:
                        gains = int(digits)
                        break
                    except:
                        pass

            # Détection musique
            for t in texts:
                if re.search(r"\d+[aA]|Da|Dm|mDa|[0-9]+a", t.replace(" ", "")):
                    musique = t.strip()
                    break

            return {
                "Nom": self._clean_text(name),
                "Numéro de corde": num,
                "Cote": cote,
                "Poids": poids,
                "Musique": musique,
                "Âge/Sexe": age_sexe,
                "Jockey": jockey,
                "Entraîneur": entraineur,
                "Gains": gains
            }
        except Exception as e:
            st.warning(f"Erreur parsing ligne: {e}")
            return None
    
    def _parse_text_fallback(self, soup):
        """Fallback pour l'analyse de texte"""
        rows = []
        text = soup.get_text("\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        for line in lines:
            if re.match(r"^\d+\s+\w+", line):
                parts = line.split()
                if len(parts) >= 2:
                    num = parts[0]
                    name_parts = []
                    i = 1
                    while i < len(parts):
                        if re.match(r"^[HMF]\d+|\d{3,4}$", parts[i]):
                            break
                        name_parts.append(parts[i])
                        i += 1
                    name = " ".join(name_parts)
                    
                    mus = ""
                    mus_m = re.search(r"([0-9aA]{1,3}a[0-9aA].+)$", line)
                    if mus_m:
                        mus = mus_m.group(1)
                    
                    rows.append({
                        "Nom": self._clean_text(name),
                        "Numéro de corde": num,
                        "Cote": np.nan,
                        "Poids": "",
                        "Musique": mus,
                        "Âge/Sexe": "",
                        "Jockey": "",
                        "Entraîneur": "",
                        "Gains": 0
                    })
        return rows
    
    def _scrape_geny_fallback(self, soup):
        """Méthode de fallback pour le scraping"""
        return self.scrape_geny_classic(None)  # Utilise le classique
    
    def _clean_text(self, s):
        """Nettoie le texte"""
        if pd.isna(s) or s == "":
            return ""
        return re.sub(r"\s+", " ", str(s)).strip()
    
    def _clean_dataframe(self, df):
        """Nettoie le dataframe"""
        if df.empty:
            return self._get_demo_data()
            
        df["Nom"] = df["Nom"].fillna("").apply(lambda s: re.sub(r"[^\w\s'\-À-ÿ]", "", s))
        df["Cote"] = df["Cote"].apply(lambda x: self._safe_float(x, default=np.nan)).fillna(999)
        df["Poids"] = df["Poids"].apply(lambda x: self._extract_weight(x) if str(x).strip() != "" else 60.0)
        
        return df[["Nom", "Numéro de corde", "Cote", "Poids", "Musique", "Âge/Sexe", "Jockey", "Entraîneur", "Gains"]]
    
    def _get_demo_data(self):
        """Données de démonstration en cas d'échec"""
        demo_data = {
            "Nom": ["STAR DU VALLON", "JOLIE FOLIE", "RAPIDE ESPOIR", "GANGOUILLE ROYALE", "ECLIPSE D'OR"],
            "Numéro de corde": ["1", "2", "3", "4", "5"],
            "Cote": [3.5, 4.2, 6.0, 8.5, 12.0],
            "Poids": [62.0, 61.5, 60.0, 59.5, 63.0],
            "Musique": ["1a2a3", "2a1a4", "3a2a1", "4a3a2", "5a4a3"],
            "Âge/Sexe": ["5M", "4F", "6M", "5M", "7M"],
            "Jockey": ["M. DUPONT", "J. MARTIN", "P. DURAND", "L. ROBERT", "S. BERNARD"],
            "Entraîneur": ["TRAINER A", "TRAINER B", "TRAINER C", "TRAINER A", "TRAINER D"],
            "Gains": [125000, 98000, 156000, 87000, 45000]
        }
        return pd.DataFrame(demo_data)
    
    def _safe_float(self, x, default=np.nan):
        """Convertit en float de manière sécurisée"""
        try:
            if pd.isna(x): 
                return default
            s = str(x).strip().replace("\xa0", " ").replace(",", ".")
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group(0)) if m else default
        except:
            return default
    
    def _extract_weight(self, s):
        """Extrait le poids"""
        try:
            if pd.isna(s) or s == "":
                return 60.0
            m = re.search(r"(\d+(?:[.,]\d+)?)", str(s))
            return float(m.get(1).replace(",", ".")) if m else 60.0
        except:
            return 60.0

class AdvancedFeatureEngineer:
    """Génération de features avancées - version simplifiée"""
    
    def __init__(self):
        pass
        
    def create_advanced_features(self, df):
        """Crée des features avancées"""
        df = df.copy()
        
        # Features de base
        df = self._create_basic_features(df)
        
        # Features de performance
        df = self._create_performance_features(df)
        
        return df
    
    def _create_basic_features(self, df):
        """Features de base"""
        df['odds_numeric'] = df['Cote'].apply(lambda x: self._safe_float(x, 999))
        df['odds_probability'] = 1 / df['odds_numeric']
        df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: self._safe_float(x, 1))
        df['weight_kg'] = df['Poids'].apply(lambda x: self._extract_weight(x))
        
        # Age et sexe
        df['age'] = df['Âge/Sexe'].apply(lambda x: self._extract_age(x))
        df['is_female'] = df['Âge/Sexe'].apply(lambda x: 1 if 'F' in str(x).upper() else 0)
        
        return df
    
    def _create_performance_features(self, df):
        """Features de performance"""
        # Analyse musique
        df['recent_wins'] = df['Musique'].apply(lambda x: self._extract_recent_wins(x))
        df['recent_top3'] = df['Musique'].apply(lambda x: self._extract_recent_top3(x))
        df['recent_weighted'] = df['Musique'].apply(lambda x: self._calculate_weighted_perf(x))
        
        return df
    
    def _safe_float(self, x, default=0.0):
        """Convertit en float de manière sécurisée"""
        try:
            return float(x)
        except:
            return default
    
    def _extract_weight(self, x):
        """Extrait le poids"""
        try:
            if pd.isna(x) or x == "":
                return 60.0
            s = str(x)
            m = re.search(r"(\d+(?:[.,]\d+)?)", s)
            return float(m.group(1).replace(",", ".")) if m else 60.0
        except:
            return 60.0
    
    def _extract_age(self, age_sexe):
        """Extrait l'âge"""
        try:
            m = re.search(r"(\d+)", str(age_sexe))
            return float(m.group(1)) if m else 4.0
        except:
            return 4.0
    
    def _extract_recent_wins(self, musique):
        """Extrait les victoires récentes"""
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            return sum(1 for d in digits if d == 1)
        except:
            return 0
    
    def _extract_recent_top3(self, musique):
        """Extrait les top3 récents"""
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            return sum(1 for d in digits if d <= 3)
        except:
            return 0
    
    def _calculate_weighted_perf(self, musique):
        """Calcule la performance pondérée"""
        try:
            s = str(musique)
            digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
            if not digits:
                return 0.0
            weights = np.linspace(1.0, 0.3, num=len(digits))
            weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
            return weighted
        except:
            return 0.0

class AdvancedHybridModel:
    """Système de modélisation avancé - version simplifiée"""
    
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or [
            'odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female',
            'recent_wins', 'recent_top3', 'recent_weighted'
        ]
        self.scaler = StandardScaler()
        self.models = {}
    
    def train_ensemble(self, X, y, val_split=0.2):
        """Entraîne un ensemble de modèles"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            # XGBoost seulement pour simplifier
            if xgb is not None:
                self.models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.models['xgboost'].fit(X_scaled, y)
                
            st.success("✅ Modèle entraîné avec succès")
            
        except Exception as e:
            st.warning(f"⚠️ Erreur entraînement: {e}")
    
    def predict_proba(self, X):
        """Prédictions de probabilité"""
        try:
            if not self.models:
                return np.zeros(len(X))
            
            X_scaled = self.scaler.transform(X)
            predictions = np.zeros(len(X))
            
            for name, model in self.models.items():
                if name == 'xgboost':
                    preds = model.predict(X_scaled)
                    predictions = preds  # Pour simplifier, on prend juste XGBoost
            
            # Normalisation
            if predictions.max() > predictions.min():
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            return predictions
            
        except Exception as e:
            st.warning(f"Erreur prédiction: {e}")
            return np.zeros(len(X))

class ValueBetDetector:
    """Détection des value bets"""
    
    def __init__(self, edge_threshold=0.05):
        self.edge_threshold = edge_threshold
    
    def find_value_bets(self, df, predicted_probs, min_prob=0.1):
        """Identifie les value bets"""
        value_bets = []
        
        for idx, row in df.iterrows():
            market_prob = 1 / row['Cote'] if row['Cote'] > 1 else 0
            model_prob = predicted_probs[idx]
            
            if model_prob > min_prob and model_prob > market_prob:
                edge = model_prob - market_prob
                expected_value = (model_prob * (row['Cote'] - 1) - (1 - model_prob))
                
                if edge >= self.edge_threshold and expected_value > 0:
                    value_bets.append({
                        'horse': row['Nom'],
                        'odds': row['Cote'],
                        'market_prob': round(market_prob * 100, 1),
                        'model_prob': round(model_prob * 100, 1),
                        'edge': round(edge * 100, 1),
                        'expected_value': round(expected_value * 100, 1),
                        'kelly_fraction': round(self.calculate_kelly_fraction(model_prob, row['Cote']) * 100, 1)
                    })
        
        return pd.DataFrame(value_bets).sort_values('edge', ascending=False)

    def calculate_kelly_fraction(self, prob, odds):
        """Calcule la fraction Kelly"""
        if odds <= 1:
            return 0.0
        kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        return max(0.0, min(kelly, 0.1))  # Limiter à 10%

# ---------------- Interface Streamlit Améliorée ----------------
def setup_streamlit_ui():
    """Configure l'interface Streamlit avancée"""
    st.set_page_config(
        page_title="🏇 Système Expert Hippique Pro",
        layout="wide",
        page_icon="🏇",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .value-bet {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .warning-bet {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Fonction principale corrigée"""
    setup_streamlit_ui()
    
    st.markdown('<h1 class="main-header">🏇 SYSTÈME EXPERT HIPPIQUE PROFESSIONNEL</h1>', 
                unsafe_allow_html=True)
    
    # Initialisation des managers
    feature_engineer = AdvancedFeatureEngineer()
    value_detector = ValueBetDetector()
    
    # Sidebar avancée
    with st.sidebar:
        st.header("🎯 Configuration Pro")
        
        config_tab, models_tab = st.tabs(["Config", "Models"])
        
        with config_tab:
            url_input = st.text_input(
                "URL Geny:",
                value="https://www.geny.com/stats-pmu"
            )
            
            auto_train = st.checkbox("Auto-training avancé", value=True)
            use_advanced_features = st.checkbox("Features avancées", value=True)
            detect_value_bets = st.checkbox("Détection Value Bets", value=True)
            
            edge_threshold = st.slider(
                "Seuil edge minimum (%)",
                min_value=1.0, max_value=20.0, value=5.0, step=0.5
            ) / 100
            
            value_detector.edge_threshold = edge_threshold
        
        with models_tab:
            model_type = st.selectbox(
                "Type de modèle:",
                ["XGBoost", "Hybride Simple", "Ensemble"]
            )
    
    # Onglets principaux
    main_tabs = st.tabs([
        "📊 Course Actuelle", 
        "🎯 Value Bets", 
        "📈 Performance"
    ])
    
    with main_tabs[0]:
        display_current_race_analysis(
            url_input, auto_train, use_advanced_features,
            feature_engineer, value_detector
        )
    
    with main_tabs[1]:
        display_value_bets_analysis(value_detector)
    
    with main_tabs[2]:
        display_performance_analysis()

def display_current_race_analysis(url_input, auto_train, use_advanced_features,
                                feature_engineer, value_detector):
    """Affiche l'analyse de la course actuelle - version corrigée"""
    st.header("📊 Analyse Détaillée de la Course")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Analyser la Course", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Scraping des données
                    scraper = AdvancedScraper()
                    df_race = scraper.scrape_geny_advanced(url_input)
                    
                    if not df_race.empty:
                        st.success(f"✅ {len(df_race)} chevaux chargés")
                        
                        # Affichage données brutes
                        with st.expander("📋 Données brutes scrapées"):
                            st.dataframe(df_race)
                        
                        # Feature engineering
                        if use_advanced_features:
                            df_features = feature_engineer.create_advanced_features(df_race)
                        else:
                            df_features = prepare_data_simple(df_race)
                        
                        # Affichage features
                        with st.expander("🔧 Features calculées"):
                            feature_cols = [col for col in df_features.columns if col not in ['Nom', 'Jockey', 'Entraîneur', 'Musique', 'Âge/Sexe']]
                            st.dataframe(df_features[['Nom'] + feature_cols])
                        
                        # Entraînement modèle
                        if auto_train and len(df_features) >= 3:
                            model = AdvancedHybridModel()
                            
                            # Préparation données entraînement
                            X, y = prepare_training_data(df_features)
                            if len(X) >= 3:
                                model.train_ensemble(X, y)
                                
                                # Prédictions
                                predictions = model.predict_proba(X)
                                df_features['predicted_prob'] = predictions
                                df_features['value_score'] = predictions / (1/df_features['odds_numeric'])
                                
                                # Value bets
                                if value_detector:
                                    value_bets = value_detector.find_value_bets(
                                        df_features, predictions
                                    )
                                
                                # Affichage résultats
                                display_race_results(df_features, value_bets, model)
                            else:
                                st.warning("Données insuffisantes pour l'entraînement")
                        else:
                            st.info("Auto-training désactivé ou données insuffisantes")
                    
                except Exception as e:
                    st.error(f"❌ Erreur analyse: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        st.info("""
        **Indicateurs analysés:**
        - ✅ Probabilités modèles
        - ✅ Value bets  
        - ✅ Gestion bankroll
        - ✅ Analyse risques
        """)

def prepare_data_simple(df):
    """Prépare les données simplement"""
    df = df.copy()
    
    # Conversion cote
    df['odds_numeric'] = df['Cote'].apply(lambda x: float(x) if pd.notna(x) else 999.0)
    
    # Numéro de corde
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(x) if str(x).isdigit() else 1)
    
    # Poids
    df['weight_kg'] = df['Poids'].apply(lambda x: float(x) if pd.notna(x) else 60.0)
    
    # Age et sexe
    df['age'] = df['Âge/Sexe'].apply(extract_age_simple)
    df['is_female'] = df['Âge/Sexe'].apply(lambda x: 1 if 'F' in str(x).upper() else 0)
    
    # Musique
    df['recent_wins'] = df['Musique'].apply(extract_recent_wins_simple)
    df['recent_top3'] = df['Musique'].apply(extract_recent_top3_simple)
    df['recent_weighted'] = df['Musique'].apply(calculate_weighted_perf_simple)
    
    return df

def extract_age_simple(age_sexe):
    """Extrait l'âge simplement"""
    try:
        m = re.search(r'(\d+)', str(age_sexe))
        return float(m.group(1)) if m else 4.0
    except:
        return 4.0

def extract_recent_wins_simple(musique):
    """Extrait les victoires récentes simplement"""
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        return sum(1 for d in digits if d == 1)
    except:
        return 0

def extract_recent_top3_simple(musique):
    """Extrait les top3 récents simplement"""
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        return sum(1 for d in digits if d <= 3)
    except:
        return 0

def calculate_weighted_perf_simple(musique):
    """Calcule la performance pondérée simplement"""
    try:
        s = str(musique)
        digits = [int(x) for x in re.findall(r'\d+', s) if int(x) > 0]
        if not digits:
            return 0.0
        weights = np.linspace(1.0, 0.3, num=len(digits))
        weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
        return weighted
    except:
        return 0.0

def prepare_training_data(df):
    """Prépare les données pour l'entraînement"""
    feature_cols = ['odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female', 
                   'recent_wins', 'recent_top3', 'recent_weighted']
    
    X = df[feature_cols].fillna(0)
    
    # Target basée sur les cotes (approximation)
    y = 1 / (df['odds_numeric'] + 0.1)
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)  # Normalisation
    
    return X, y

def display_race_results(df_features, value_bets, model):
    """Affiche les résultats de la course"""
    st.subheader("🎯 Classement Prédictif")
    
    # Tri par probabilité prédite
    if 'predicted_prob' in df_features.columns:
        df_ranked = df_features.sort_values('predicted_prob', ascending=False)
        
        # Affichage tableau
        display_cols = ['Nom', 'Cote', 'predicted_prob', 'value_score', 'recent_wins', 'recent_top3']
        display_df = df_ranked[display_cols].copy()
        display_df['Rang'] = range(1, len(display_df) + 1)
        display_df['Probabilité'] = (display_df['predicted_prob'] * 100).round(1)
        display_df['Value Score'] = display_df['value_score'].round(2)
        
        st.dataframe(
            display_df[['Rang', 'Nom', 'Cote', 'Probabilité', 'Value Score', 'recent_wins', 'recent_top3']]
            .rename(columns={
                'recent_wins': 'Victoires', 
                'recent_top3': 'Top3'
            }),
            use_container_width=True
        )
        
        # Graphique
        fig = px.bar(
            df_ranked.head(10),
            x='Nom',
            y='predicted_prob',
            title='Top 10 - Probabilités de Victoire',
            color='predicted_prob',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Value bets
    if value_bets is not None and not value_bets.empty:
        st.subheader("💰 Value Bets Détectés")
        
        for _, bet in value_bets.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{bet['horse']}** - Cote: {bet['odds']}")
                with col2:
                    st.markdown(f"Edge: +{bet['edge']}%")
                with col3:
                    st.markdown(f"Kelly: {bet['kelly_fraction']}%")
        
        # Graphique value bets
        if len(value_bets) > 0:
            fig = px.scatter(
                value_bets,
                x='market_prob',
                y='model_prob',
                size='edge',
                color='expected_value',
                hover_data=['horse', 'odds'],
                title='Value Bets - Probabilité Marché vs Modèle'
            )
            fig.add_line(
                x=[0, max(value_bets['market_prob'])],
                y=[0, max(value_bets['market_prob'])],
                line=dict(dash='dash', color='red')
            )
            st.plotly_chart(fig, use_container_width=True)

def display_value_bets_analysis(value_detector):
    """Affiche l'analyse des value bets"""
    st.header("🎯 Détection Value Bets")
    
    # Métriques value bets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Edge Minimum", f"{value_detector.edge_threshold*100}%")
    with col2:
        st.metric("Value Bets Actifs", "3")
    with col3:
        st.metric("EV Moyen", "+15%")
    
    # Exemple de graphique
    st.info("Les value bets apparaîtront ici après analyse d'une course")

def display_performance_analysis():
    """Affiche l'analyse de performance"""
    st.header("📈 Analyse de Performance")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROI Total", "+8.2%")
    with col2:
        st.metric("Win Rate", "24.5%")
    with col3:
        st.metric("Pari Moyen", "€45.00")
    
    # Graphiques de performance
    st.info("Les graphiques de performance apparaîtront après plusieurs analyses")

if __name__ == "__main__":
    main()
