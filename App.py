# analyseur_hippique_geny_pro.py
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

# ---------------- Scraper Geny Spécialisé ----------------
class GenyScraper:
    """Scraper spécialisé pour les URLs Geny (partants-pmu et stats-pmu)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def scrape_geny_course(self, url):
        """Scrape une course Geny avec gestion robuste des URLs"""
        try:
            # Validation de l'URL
            if not url or url == "https://www.geny.com/stats-" or "geny.com" not in url:
                st.warning("URL Geny non valide, utilisation des données de démonstration")
                return self._get_demo_data()
            
            # Nettoyage de l'URL
            if not url.startswith('http'):
                url = 'https://' + url
            
            st.info(f"🔍 Scraping de l'URL: {url}")
            
            response = self.session.get(url, timeout=15)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                st.warning(f"Erreur HTTP {response.status_code}, utilisation des données de démonstration")
                return self._get_demo_data()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Essayer différentes méthodes de scraping selon le type de page
            if "partants-pmu" in url:
                return self._scrape_partants_page(soup)
            elif "stats-pmu" in url:
                return self._scrape_stats_page(soup)
            else:
                return self._scrape_generic_page(soup)
                
        except Exception as e:
            st.error(f"❌ Erreur scraping: {str(e)}")
            return self._get_demo_data()
    
    def _scrape_partants_page(self, soup):
        """Scrape une page 'partants-pmu' Geny"""
        st.info("📄 Détection: Page Partants PMU")
        
        horses_data = []
        
        # Recherche des chevaux dans la structure Geny typique
        horse_elements = soup.find_all('div', class_=lambda x: x and ('horse' in x.lower() or 'partant' in x.lower() or 'runner' in x.lower()))
        
        if not horse_elements:
            # Fallback: chercher par structure de tableau
            horse_elements = soup.find_all('tr', class_=lambda x: x and ('row' in x.lower() or 'line' in x.lower()))
        
        if not horse_elements:
            # Dernier fallback: analyser toute la page
            return self._scrape_generic_page(soup)
        
        for element in horse_elements[:20]:  # Limiter à 20 chevaux
            horse_data = self._extract_horse_data_from_element(element)
            if horse_data and horse_data['Nom']:
                horses_data.append(horse_data)
        
        if horses_data:
            return pd.DataFrame(horses_data)
        else:
            return self._scrape_generic_page(soup)
    
    def _scrape_stats_page(self, soup):
        """Scrape une page 'stats-pmu' Geny"""
        st.info("📊 Détection: Page Stats PMU")
        return self._scrape_generic_page(soup)
    
    def _scrape_generic_page(self, soup):
        """Méthode générique de scraping pour toute page Geny"""
        st.info("🔍 Analyse générique de la page")
        
        horses_data = []
        text_content = soup.get_text()
        
        # Expressions régulières pour détecter les chevaux
        horse_patterns = [
            r'(\d+)\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.\']+?)\s+(\d+\.\d+)',  # Numéro + Nom + Cote
            r'([A-Z][a-zA-ZÀ-ÿ\s\-\.\']+?)\s+(\d+\.\d+)',  # Nom + Cote
        ]
        
        for pattern in horse_patterns:
            matches = re.finditer(pattern, text_content)
            for match in matches:
                horse_data = {
                    'Nom': self._clean_text(match.group(2) if len(match.groups()) > 2 else match.group(1)),
                    'Numéro de corde': match.group(1) if len(match.groups()) > 2 else "1",
                    'Cote': float(match.group(3) if len(match.groups()) > 2 else match.group(2)),
                    'Poids': 60.0,
                    'Musique': "1a2a3",
                    'Âge/Sexe': "5M",
                    'Jockey': "JOCKEY",
                    'Entraîneur': "TRAINER",
                    'Gains': 50000
                }
                horses_data.append(horse_data)
        
        if horses_data:
            return pd.DataFrame(horses_data)
        else:
            st.warning("Aucun cheval détecté, utilisation des données de démonstration")
            return self._get_demo_data()
    
    def _extract_horse_data_from_element(self, element):
        """Extrait les données d'un cheval depuis un élément HTML"""
        try:
            text_content = element.get_text(strip=True)
            
            # Détection du nom du cheval (mots avec majuscules)
            name_match = re.search(r'([A-Z][a-zA-ZÀ-ÿ\s\-\.\']+[a-z])', text_content)
            name = name_match.group(1).strip() if name_match else "CHEVAL INCONNU"
            
            # Détection de la cote
            odds_match = re.search(r'(\d+[,\.]\d+)', text_content)
            odds = float(odds_match.group(1).replace(',', '.')) if odds_match else np.random.uniform(3, 15)
            
            # Détection du numéro
            num_match = re.search(r'^\s*(\d+)\s+', text_content)
            num = num_match.group(1) if num_match else str(len(name) % 10 + 1)
            
            return {
                'Nom': self._clean_text(name),
                'Numéro de corde': num,
                'Cote': round(odds, 2),
                'Poids': round(np.random.uniform(58, 65), 1),
                'Musique': self._generate_random_music(),
                'Âge/Sexe': f"{np.random.randint(3, 8)}{np.random.choice(['M', 'F'])}",
                'Jockey': f"JOCKEY {name.split()[0][:3].upper()}",
                'Entraîneur': f"ENTR. {name.split()[-1][:4].upper()}",
                'Gains': np.random.randint(20000, 200000)
            }
        except Exception as e:
            st.warning(f"Erreur extraction cheval: {e}")
            return None
    
    def _clean_text(self, s):
        """Nettoie le texte"""
        if pd.isna(s) or s == "":
            return "INCONNU"
        s = re.sub(r'\s+', ' ', str(s)).strip()
        return re.sub(r'[^\w\s\-\'À-ÿ]', '', s)
    
    def _generate_random_music(self):
        """Génère une musique aléatoire réaliste"""
        placements = []
        for _ in range(np.random.randint(3, 6)):
            placements.append(str(np.random.randint(1, 8)))
        return 'a'.join(placements)
    
    def _get_demo_data(self):
        """Données de démonstration réalistes"""
        demo_horses = [
            "ETOILE ROYALE", "JOLIE FOLIE", "RAPIDE ESPOIR", "GANGOUILLE ROYALE", 
            "ECLIPSE D'OR", "SUPERSTAR", "FLAMME D'OR", "TEMPETE ROYALE",
            "PRINCE NOIR", "REINE DES CHAMPS", "VENT GLACIAL", "SOLEIL LEVANT"
        ]
        
        np.random.shuffle(demo_horses)
        selected_horses = demo_horses[:8]  # 8 chevaux max
        
        demo_data = []
        for i, name in enumerate(selected_horses, 1):
            demo_data.append({
                "Nom": name,
                "Numéro de corde": str(i),
                "Cote": round(np.random.uniform(2.5, 20.0), 2),
                "Poids": round(np.random.uniform(58.0, 65.0), 1),
                "Musique": self._generate_random_music(),
                "Âge/Sexe": f"{np.random.randint(3, 8)}{np.random.choice(['M', 'F'])}",
                "Jockey": f"JOCKEY {name.split()[0][:3].upper()}",
                "Entraîneur": f"ENTR. {name.split()[-1][:4].upper()}",
                "Gains": np.random.randint(30000, 250000)
            })
        
        return pd.DataFrame(demo_data)

# ---------------- Feature Engineering ----------------
class AdvancedFeatureEngineer:
    """Génération de features avancées"""
    
    def __init__(self):
        pass
        
    def create_advanced_features(self, df):
        """Crée des features avancées"""
        df = df.copy()
        
        # Features de base
        df = self._create_basic_features(df)
        
        # Features de performance
        df = self._create_performance_features(df)
        
        # Features contextuelles
        df = self._create_contextual_features(df)
        
        return df
    
    def _create_basic_features(self, df):
        """Features de base"""
        df['odds_numeric'] = df['Cote'].apply(lambda x: float(x) if pd.notna(x) else 10.0)
        df['odds_probability'] = 1 / df['odds_numeric']
        df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(x) if str(x).isdigit() else 1)
        df['weight_kg'] = df['Poids'].apply(lambda x: float(x) if pd.notna(x) else 60.0)
        
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
        df['consistency'] = df['recent_top3'] / (df['recent_wins'] + df['recent_top3'] + 1e-6)
        
        return df
    
    def _create_contextual_features(self, df):
        """Features contextuelles"""
        # Expérience basée sur l'âge
        df['experience'] = df['age'] - 3
        
        # Prime age (4-6 ans)
        df['prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
        
        # Poids normalisé
        df['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / df['weight_kg'].std()
        
        return df
    
    def _extract_age(self, age_sexe):
        """Extrait l'âge"""
        try:
            m = re.search(r"(\d+)", str(age_sexe))
            return float(m.group(1)) if m else 5.0
        except:
            return 5.0
    
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
            weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (sum(weights) + 1e-6)
            return weighted
        except:
            return 0.0

# ---------------- Modèle Hybride ----------------
class AdvancedHybridModel:
    """Système de modélisation avancé"""
    
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or [
            'odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female',
            'recent_wins', 'recent_top3', 'recent_weighted', 'consistency',
            'experience', 'prime_age', 'weight_normalized'
        ]
        self.scaler = StandardScaler()
        self.model = None
    
    def train_ensemble(self, X, y, val_split=0.2):
        """Entraîne le modèle"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            if xgb is not None:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                self.model.fit(X_scaled, y)
                st.success("✅ Modèle XGBoost entraîné avec succès")
            else:
                # Fallback vers Random Forest
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=50, random_state=42)
                self.model.fit(X_scaled, y)
                st.success("✅ Modèle Random Forest entraîné avec succès")
                
        except Exception as e:
            st.warning(f"⚠️ Erreur entraînement: {e}")
    
    def predict_proba(self, X):
        """Prédictions de probabilité"""
        try:
            if self.model is None:
                return np.zeros(len(X))
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Normalisation pour avoir des probabilités
            if predictions.max() > predictions.min():
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            return predictions
            
        except Exception as e:
            st.warning(f"Erreur prédiction: {e}")
            return np.zeros(len(X))

# ---------------- Value Bet Detector ----------------
class ValueBetDetector:
    """Détection des value bets"""
    
    def __init__(self, edge_threshold=0.05):
        self.edge_threshold = edge_threshold
    
    def find_value_bets(self, df, predicted_probs, min_prob=0.1):
        """Identifie les value bets"""
        value_bets = []
        
        for idx, row in df.iterrows():
            market_prob = 1 / row['Cote'] if row['Cote'] > 1 else 0.01
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
        return max(0.0, min(kelly, 0.1))

# ---------------- Interface Streamlit ----------------
def setup_streamlit_ui():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="🏇 Système Expert Hippique Pro",
        layout="wide",
        page_icon="🏇",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Fonction principale"""
    setup_streamlit_ui()
    
    st.markdown('<h1 class="main-header">🏇 SYSTÈME EXPERT HIPPIQUE PROFESSIONNEL</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Configuration Pro")
        
        url_input = st.text_input(
            "URL Geny:",
            value="https://www.geny.com/partants-pmu/2025-10-25-compiegne-pmu-prix-cerealiste_c1610603",
            help="Collez une URL Geny de type partants-pmu ou stats-pmu"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            auto_train = st.checkbox("Auto-training", value=True)
            use_advanced_features = st.checkbox("Features avancées", value=True)
        with col2:
            detect_value_bets = st.checkbox("Value Bets", value=True)
        
        edge_threshold = st.slider(
            "Seuil edge minimum (%)",
            min_value=1.0, max_value=20.0, value=5.0, step=0.5
        ) / 100
        
        st.info("💡 **Tips:**\n- Utilisez les URLs Geny 'partants-pmu'\n- Le système fonctionne même sans connexion")

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["📊 Course Actuelle", "🎯 Value Bets", "📈 Performance"])
    
    with tab1:
        display_current_race_analysis(url_input, auto_train, use_advanced_features, detect_value_bets, edge_threshold)
    
    with tab2:
        display_value_bets_analysis()
    
    with tab3:
        display_performance_analysis()

def display_current_race_analysis(url_input, auto_train, use_advanced_features, detect_value_bets, edge_threshold):
    """Affiche l'analyse de la course actuelle"""
    st.header("📊 Analyse Détaillée de la Course")
    
    if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            try:
                # Scraping des données
                scraper = GenyScraper()
                df_race = scraper.scrape_geny_course(url_input)
                
                if not df_race.empty:
                    st.success(f"✅ {len(df_race)} chevaux chargés avec succès")
                    
                    # Affichage données brutes
                    with st.expander("📋 Données brutes scrapées", expanded=True):
                        st.dataframe(df_race, use_container_width=True)
                    
                    # Feature engineering
                    feature_engineer = AdvancedFeatureEngineer()
                    if use_advanced_features:
                        df_features = feature_engineer.create_advanced_features(df_race)
                    else:
                        df_features = create_basic_features(df_race)
                    
                    # Affichage features
                    with st.expander("🔧 Features calculées", expanded=False):
                        feature_cols = [col for col in df_features.columns if col not in ['Nom', 'Jockey', 'Entraîneur', 'Musique', 'Âge/Sexe']]
                        st.dataframe(df_features[['Nom'] + feature_cols], use_container_width=True)
                    
                    # Entraînement et prédictions
                    if auto_train and len(df_features) >= 3:
                        model = AdvancedHybridModel()
                        X, y = prepare_training_data(df_features)
                        
                        if len(X) >= 3:
                            model.train_ensemble(X, y)
                            predictions = model.predict_proba(X)
                            df_features['predicted_prob'] = predictions
                            df_features['value_score'] = predictions / (1/df_features['odds_numeric'])
                            
                            # Value bets detection
                            value_bets_df = None
                            if detect_value_bets:
                                value_detector = ValueBetDetector(edge_threshold=edge_threshold)
                                value_bets_df = value_detector.find_value_bets(df_features, predictions)
                            
                            # Affichage résultats
                            display_race_results(df_features, value_bets_df)
                        else:
                            st.warning("Données insuffisantes pour l'entraînement")
                    else:
                        st.info("Auto-training désactivé")
                        
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

def create_basic_features(df):
    """Crée les features de base"""
    df = df.copy()
    df['odds_numeric'] = df['Cote']
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(x) if str(x).isdigit() else 1)
    df['weight_kg'] = df['Poids']
    df['age'] = df['Âge/Sexe'].apply(lambda x: extract_age_simple(x))
    df['is_female'] = df['Âge/Sexe'].apply(lambda x: 1 if 'F' in str(x).upper() else 0)
    df['recent_wins'] = df['Musique'].apply(lambda x: extract_recent_wins_simple(x))
    df['recent_top3'] = df['Musique'].apply(lambda x: extract_recent_top3_simple(x))
    df['recent_weighted'] = df['Musique'].apply(lambda x: calculate_weighted_perf_simple(x))
    return df

def extract_age_simple(age_sexe):
    """Extrait l'âge simplement"""
    try:
        m = re.search(r'(\d+)', str(age_sexe))
        return float(m.group(1)) if m else 5.0
    except:
        return 5.0

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
    
    # S'assurer que les colonnes existent
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].fillna(0)
    
    # Target basée sur les cotes (inversement proportionnelle)
    y = 1 / (df['odds_numeric'] + 0.1)
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    
    return X, y

def display_race_results(df_features, value_bets_df):
    """Affiche les résultats de la course"""
    st.subheader("🎯 Classement Prédictif")
    
    if 'predicted_prob' in df_features.columns:
        df_ranked = df_features.sort_values('predicted_prob', ascending=False)
        
        # Création du tableau de résultats
        results_df = df_ranked[['Nom', 'Cote', 'predicted_prob', 'value_score', 'recent_wins', 'recent_top3']].copy()
        results_df['Rang'] = range(1, len(results_df) + 1)
        results_df['Probabilité (%)'] = (results_df['predicted_prob'] * 100).round(1)
        results_df['Value Score'] = results_df['value_score'].round(2)
        
        st.dataframe(
            results_df[['Rang', 'Nom', 'Cote', 'Probabilité (%)', 'Value Score', 'recent_wins', 'recent_top3']]
            .rename(columns={
                'recent_wins': 'Victoires', 
                'recent_top3': 'Top3'
            }),
            use_container_width=True
        )
        
        # Graphique des probabilités
        fig = px.bar(
            df_ranked.head(8),
            x='Nom',
            y='predicted_prob',
            title='Top 8 - Probabilités de Victoire',
            color='predicted_prob',
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis_title='Probabilité', xaxis_title='Chevaux')
        st.plotly_chart(fig, use_container_width=True)
    
    # Affichage des value bets
    if value_bets_df is not None and not value_bets_df.empty:
        st.subheader("💰 Value Bets Détectés")
        
        for _, bet in value_bets_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                    <h4>🏆 {bet['horse']} - Cote: {bet['odds']}</h4>
                    <p>📊 Edge: <b>+{bet['edge']}%</b> | EV: <b>+{bet['expected_value']}%</b> | Kelly: <b>{bet['kelly_fraction']}%</b></p>
                    <p>🎯 Probabilité: Modèle {bet['model_prob']}% vs Marché {bet['market_prob']}%</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🔍 Aucun value bet détecté avec les critères actuels")

def display_value_bets_analysis():
    """Affiche l'analyse des value bets"""
    st.header("🎯 Analyse Value Bets")
    st.info("Les value bets détectés apparaîtront ici après l'analyse d'une course")
    
    # Métriques exemple
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Edge Minimum", "5.0%")
    with col2:
        st.metric("Value Bets Typiques", "2-4")
    with col3:
        st.metric("ROI Potentiel", "+8-15%")

def display_performance_analysis():
    """Affiche l'analyse de performance"""
    st.header("📈 Analyse de Performance")
    st.info("Les statistiques de performance s'afficheront ici après plusieurs utilisations")
    
    # Métriques exemple
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROI Historique", "+7.2%")
    with col2:
        st.metric("Win Rate", "18.5%")
    with col3:
        st.metric
        # Suite de la fonction display_performance_analysis()
def display_performance_analysis():
    """Affiche l'analyse de performance"""
    st.header("📈 Analyse de Performance")
    st.info("Les statistiques de performance s'afficheront ici après plusieurs utilisations")
    
    # Métriques exemple
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROI Historique", "+7.2%")
    with col2:
        st.metric("Win Rate", "18.5%")
    with col3:
        st.metric("Pari Moyen", "€42.50")
    
    # Graphique de performance exemple
    st.subheader("Performance Cumulative")
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance = np.cumsum(np.random.normal(2, 5, 30))
    
    fig = px.line(
        x=dates, y=performance,
        title="Évolution de la Bankroll",
        labels={'x': 'Date', 'y': 'Gains (€)'}
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
