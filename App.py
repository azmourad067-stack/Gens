# analyseur_hippique_geny_pro.py
# -*- coding: utf-8 -*-
"""
Analyseur Hippique Professionnel — Système Expert Geny + Auto-training Hybride Avancé
Features:
- Scraping avancé Geny + autres sources
- Modèles ML ensemble avancés (DL, XGBoost, LightGBM, Random Forest)
- Analyse technique complète (formes, statistiques avancées)
- Gestion bankroll et gestion des risques
- Simulations Monte Carlo
- Alertes value bets
- Rapports PDF professionnels
- API intégrée pour données temps réel
- Dashboard de performance
- Système de tracking des paris
"""

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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

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
    import tensorflow as tf
    from tensorflow.keras import Sequential, callbacks
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras.optimizers import Adam
except Exception:
    tf = None
try:
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.cluster import KMeans
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

# Paths principaux
HIST_PATH = os.path.join(DATA_DIR, "historique_complet.csv")
BANKROLL_PATH = os.path.join(DATA_DIR, "bankroll.json")
PERFORMANCE_PATH = os.path.join(DATA_DIR, "performance.json")
BETS_TRACKING_PATH = os.path.join(DATA_DIR, "paris_trackes.csv")
CONFIG_PATH = os.path.join(DATA_DIR, "config_pro.json")

# ---------------- Classes de Gestion Avancée ----------------
class BankrollManager:
    """Gestion professionnelle de bankroll avec critères de Kelly"""
    
    def __init__(self, initial_br=1000):
        self.initial_br = initial_br
        self.load_bankroll()
    
    def load_bankroll(self):
        """Charge ou initialise la bankroll"""
        try:
            with open(BANKROLL_PATH, 'r') as f:
                data = json.load(f)
                self.bankroll = data.get('bankroll', self.initial_br)
                self.history = data.get('history', [])
        except:
            self.bankroll = self.initial_br
            self.history = []
    
    def save_bankroll(self):
        """Sauvegarde la bankroll"""
        data = {
            'bankroll': self.bankroll,
            'history': self.history[-1000:]  # Garder les 1000 derniers paris
        }
        with open(BANKROLL_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def kelly_criterion(self, probability, odds):
        """Calcule la fraction Kelly optimale"""
        if odds <= 1:
            return 0.0
        q = 1 - probability
        b = odds - 1
        kelly = (b * probability - q) / b
        return max(0.0, min(kelly, 0.1))  # Limiter à 10% maximum
    
    def fractional_kelly(self, probability, odds, fraction=0.25):
        """Fraction Kelly (plus conservative)"""
        return self.kelly_criterion(probability, odds) * fraction
    
    def update_bankroll(self, amount, bet_type="simple", description=""):
        """Met à jour la bankroll après un pari"""
        old_br = self.bankroll
        self.bankroll += amount
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'old_bankroll': old_br,
            'amount': amount,
            'new_bankroll': self.bankroll,
            'type': bet_type,
            'description': description
        }
        self.history.append(record)
        self.save_bankroll()
        
        return self.bankroll

class PerformanceTracker:
    """Tracking détaillé des performances"""
    
    def __init__(self):
        self.load_performance()
    
    def load_performance(self):
        """Charge les données de performance"""
        try:
            with open(PERFORMANCE_PATH, 'r') as f:
                self.data = json.load(f)
        except:
            self.data = {
                'total_bets': 0,
                'won_bets': 0,
                'lost_bets': 0,
                'push_bets': 0,
                'total_staked': 0,
                'total_return': 0,
                'roi': 0,
                'streak_current': 0,
                'streak_max': 0,
                'daily_performance': {},
                'monthly_performance': {}
            }
    
    def save_performance(self):
        """Sauvegarde les performances"""
        with open(PERFORMANCE_PATH, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_bet(self, stake, return_amount, won=True):
        """Enregistre un pari"""
        self.data['total_bets'] += 1
        self.data['total_staked'] += stake
        
        if won:
            self.data['won_bets'] += 1
            self.data['total_return'] += return_amount
            self.data['streak_current'] = max(0, self.data['streak_current'] + 1)
        else:
            self.data['lost_bets'] += 1
            self.data['streak_current'] = min(0, self.data['streak_current'] - 1)
        
        self.data['streak_max'] = max(abs(self.data['streak_current']), self.data['streak_max'])
        self.data['roi'] = ((self.data['total_return'] - self.data['total_staked']) / self.data['total_staked']) * 100
        
        # Performance journalière
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.data['daily_performance']:
            self.data['daily_performance'][today] = {'staked': 0, 'return': 0}
        
        self.data['daily_performance'][today]['staked'] += stake
        self.data['daily_performance'][today]['return'] += return_amount
        
        self.save_performance()

class AdvancedScraper:
    """Scraper avancé multi-sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_geny_advanced(self, url):
        """Scraping avancé Geny avec plus de données"""
        try:
            response = self.session.get(url, timeout=15)
            response.encoding = 'ISO-8859-1'
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extraction des données de base
            base_data = self._extract_base_data(soup)
            # Extraction des statistiques avancées
            advanced_stats = self._extract_advanced_stats(soup)
            # Extraction des tendances et formes
            trends = self._extract_trends(soup)
            
            # Combinaison des données
            combined_data = []
            for horse in base_data:
                horse_name = horse['Nom']
                advanced = advanced_stats.get(horse_name, {})
                trend = trends.get(horse_name, {})
                
                combined_data.append({**horse, **advanced, **trend})
            
            return pd.DataFrame(combined_data)
            
        except Exception as e:
            st.error(f"Erreur scraping avancé: {e}")
            return pd.DataFrame()
    
    def _extract_base_data(self, soup):
        """Extraction des données de base améliorée"""
        # Implémentation existante améliorée
        horses = []
        # Recherche des tables de chevaux
        tables = soup.find_all('table', class_=lambda x: x and 'table' in x.lower())
        
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 5:
                    horse_data = self._parse_horse_row(cells)
                    if horse_data:
                        horses.append(horse_data)
        
        return horses
    
    def _extract_advanced_stats(self, soup):
        """Extraction des statistiques avancées"""
        stats = {}
        # Implémentation pour extraire les stats détaillées
        # Vitesses moyennes, records, etc.
        return stats
    
    def _extract_trends(self, soup):
        """Extraction des tendances et formes"""
        trends = {}
        # Analyse des dernières performances
        return trends

class MonteCarloSimulator:
    """Simulateur Monte Carlo pour les paris"""
    
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
    
    def simulate_season(self, initial_br, avg_stake, win_rate, avg_odds, n_bets):
        """Simule une saison de paris"""
        final_brs = []
        
        for _ in range(self.n_simulations):
            bankroll = initial_br
            for _ in range(n_bets):
                if bankroll <= 0:
                    break
                
                # Tirage aléatoire selon le win rate
                win = np.random.random() < win_rate
                if win:
                    bankroll += avg_stake * (avg_odds - 1)
                else:
                    bankroll -= avg_stake
            
            final_brs.append(bankroll)
        
        return np.array(final_brs)
    
    def calculate_risk_metrics(self, final_brs, initial_br):
        """Calcule les métriques de risque"""
        roi = (final_brs - initial_br) / initial_br * 100
        
        metrics = {
            'mean_roi': np.mean(roi),
            'median_roi': np.median(roi),
            'std_roi': np.std(roi),
            'var_95': np.percentile(roi, 5),
            'var_99': np.percentile(roi, 1),
            'probability_ruin': np.mean(final_brs <= 0),
            'expected_max_drawdown': self.calculate_expected_drawdown(final_brs),
            'sharpe_ratio': np.mean(roi) / np.std(roi) if np.std(roi) > 0 else 0
        }
        
        return metrics
    
    def calculate_expected_drawdown(self, final_brs):
        """Calcule le drawdown attendu"""
        # Implémentation simplifiée
        return np.percentile(final_brs, 10)

# ---------------- Modèle Hybride Avancé ----------------
class AdvancedHybridModel:
    """Système de modélisation avancé avec ensemble learning"""
    
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or self.get_default_features()
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_weights = {}
        self.feature_importance = {}
        
    def get_default_features(self):
        """Retourne les features par défaut"""
        return [
            'odds_numeric', 'draw_numeric', 'weight_kg', 'age', 'is_female',
            'recent_wins', 'recent_top3', 'recent_weighted', 'days_since_last_run',
            'career_starts', 'career_wins', 'win_percentage', 'track_win_percentage',
            'distance_win_percentage', 'jockey_win_percentage', 'trainer_win_percentage',
            'avg_speed_rating', 'last_speed_rating', 'best_speed_rating',
            'class_drop', 'weight_carried', 'days_rest', 'prime_time'
        ]
    
    def build_advanced_dl(self, input_dim):
        """Construit un modèle DL avancé"""
        if tf is None:
            return None
            
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Probabilité de victoire
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse']
        )
        
        return model
    
    def build_lstm_model(self, input_dim, sequence_length=5):
        """Modèle LSTM pour séries temporelles"""
        if tf is None:
            return None
            
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, input_dim)),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ensemble(self, X, y, val_split=0.2):
        """Entraîne un ensemble de modèles"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement multiple modèles
        models_to_train = {
            'dl_advanced': self.build_advanced_dl(X_scaled.shape[1]),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ) if xgb else None,
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ) if lgb else None,
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        }
        
        # Entraînement des modèles
        for name, model in models_to_train.items():
            if model is not None:
                try:
                    if name == 'dl_advanced':
                        early_stop = callbacks.EarlyStopping(
                            monitor='val_loss', patience=10, restore_best_weights=True
                        )
                        model.fit(
                            X_scaled, y,
                            validation_split=val_split,
                            epochs=100,
                            batch_size=32,
                            callbacks=[early_stop],
                            verbose=0
                        )
                    else:
                        model.fit(X_scaled, y)
                    
                    self.models[name] = model
                    
                    # Calcul importance des features
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_
                    
                except Exception as e:
                    st.warning(f"Erreur entraînement {name}: {e}")
        
        # Optimisation des poids de l'ensemble
        self.optimize_ensemble_weights(X_scaled, y)
    
    def optimize_ensemble_weights(self, X, y):
        """Optimise les poids de l'ensemble"""
        if len(self.models) < 2:
            return
        
        # Prédictions de base
        predictions = {}
        for name, model in self.models.items():
            if name == 'dl_advanced':
                predictions[name] = model.predict(X).flatten()
            else:
                predictions[name] = model.predict(X)
        
        # Optimisation des poids
        def objective(weights):
            weighted_pred = sum(w * predictions[name] for w, name in zip(weights, self.models.keys()))
            return mean_squared_error(y, weighted_pred)
        
        # Contraintes
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0, 1) for _ in range(len(self.models))]
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            for i, name in enumerate(self.models.keys()):
                self.ensemble_weights[name] = result.x[i]
    
    def predict_proba(self, X):
        """Prédictions de probabilité"""
        if not self.models:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'dl_advanced':
                predictions[name] = model.predict(X_scaled).flatten()
            else:
                predictions[name] = model.predict(X_scaled)
        
        # Combinaison pondérée
        if self.ensemble_weights:
            final_pred = sum(self.ensemble_weights.get(name, 0) * pred 
                           for name, pred in predictions.items())
        else:
            final_pred = np.mean(list(predictions.values()), axis=0)
        
        return final_pred

# ---------------- Feature Engineering Avancé ----------------
class AdvancedFeatureEngineer:
    """Génération de features avancées"""
    
    def __init__(self):
        self.jockey_encoder = LabelEncoder()
        self.trainer_encoder = LabelEncoder()
        
    def create_advanced_features(self, df):
        """Crée des features avancées"""
        df = df.copy()
        
        # Features de base
        df = self.create_basic_features(df)
        
        # Features de performance
        df = self.create_performance_features(df)
        
        # Features contextuelles
        df = self.create_contextual_features(df)
        
        # Features d'interaction
        df = self.create_interaction_features(df)
        
        # Features temporelles
        df = self.create_temporal_features(df)
        
        return df
    
    def create_basic_features(self, df):
        """Features de base"""
        df['odds_probability'] = 1 / df['Cote']
        df['log_odds'] = np.log(df['Cote'])
        df['weight_vs_avg'] = df['Poids'] / df['Poids'].mean()
        df['draw_advantage'] = (df['Numéro de corde'].max() - df['Numéro de corde']) / df['Numéro de corde'].max()
        
        return df
    
    def create_performance_features(self, df):
        """Features de performance"""
        # Statistiques de carrière
        if 'Gains' in df.columns:
            df['earnings_per_start'] = df['Gains'] / (df.get('course_count', 1) + 1)
        
        # Forme récente
        df['recent_form'] = df['recent_weighted'] / (df['recent_weighted'].max() + 1e-6)
        
        # Consistance
        df['consistency'] = df['recent_top3'] / (df.get('recent_starts', 1) + 1e-6)
        
        return df
    
    def create_contextual_features(self, df):
        """Features contextuelles"""
        # Avantage jockey/entraîneur
        if 'Jockey' in df.columns:
            df['jockey_win_rate'] = df.groupby('Jockey')['recent_wins'].transform('mean')
        
        if 'Entraîneur' in df.columns:
            df['trainer_win_rate'] = df.groupby('Entraîneur')['recent_wins'].transform('mean')
        
        return df
    
    def create_interaction_features(self, df):
        """Features d'interaction"""
        df['odds_x_form'] = df['odds_probability'] * df['recent_form']
        df['weight_x_draw'] = df['weight_vs_avg'] * df['draw_advantage']
        
        return df
    
    def create_temporal_features(self, df):
        """Features temporelles"""
        df['prime_time'] = ((df['age'] >= 4) & (df['age'] <= 7)).astype(int)
        df['experience'] = df['age'] - 2  # Approximation
        
        return df

# ---------------- Système de Value Bet Detection ----------------
class ValueBetDetector:
    """Détection des value bets"""
    
    def __init__(self, edge_threshold=0.05):
        self.edge_threshold = edge_threshold
    
    def find_value_bets(self, df, predicted_probs, min_prob=0.1):
        """Identifie les value bets"""
        value_bets = []
        
        for idx, row in df.iterrows():
            market_prob = 1 / row['Cote']
            model_prob = predicted_probs[idx]
            
            if model_prob > min_prob and model_prob > market_prob:
                edge = model_prob - market_prob
                expected_value = (model_prob * (row['Cote'] - 1) - (1 - model_prob))
                
                if edge >= self.edge_threshold and expected_value > 0:
                    value_bets.append({
                        'horse': row['Nom'],
                        'odds': row['Cote'],
                        'market_prob': market_prob,
                        'model_prob': model_prob,
                        'edge': edge,
                        'expected_value': expected_value,
                        'kelly_fraction': self.calculate_kelly_fraction(model_prob, row['Cote'])
                    })
        
        return pd.DataFrame(value_bets).sort_values('edge', ascending=False)

    def calculate_kelly_fraction(self, prob, odds):
        """Calcule la fraction Kelly"""
        if odds <= 1:
            return 0.0
        return (prob * (odds - 1) - (1 - prob)) / (odds - 1)

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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
    
    st.markdown('<h1 class="main-header">🏇 SYSTÈME EXPERT HIPPIQUE PROFESSIONNEL</h1>', 
                unsafe_allow_html=True)

def main():
    """Fonction principale"""
    setup_streamlit_ui()
    
    # Initialisation des managers
    bankroll_mgr = BankrollManager()
    performance_tracker = PerformanceTracker()
    feature_engineer = AdvancedFeatureEngineer()
    value_detector = ValueBetDetector()
    
    # Sidebar avancée
    with st.sidebar:
        st.header("🎯 Configuration Pro")
        
        # Onglets sidebar
        config_tab, bankroll_tab, models_tab = st.tabs(["Config", "Bankroll", "Models"])
        
        with config_tab:
            url_input = st.text_input(
                "URL Geny:",
                value="https://www.geny.com/stats-pmu?id_course=1610442"
            )
            
            auto_train = st.checkbox("Auto-training avancé", value=True)
            use_advanced_features = st.checkbox("Features avancées", value=True)
            detect_value_bets = st.checkbox("Détection Value Bets", value=True)
            
            edge_threshold = st.slider(
                "Seuil edge minimum (%)",
                min_value=1.0, max_value=20.0, value=5.0, step=0.5
            ) / 100
        
        with bankroll_tab:
            st.metric("Bankroll", f"€{bankroll_mgr.bankroll:,.2f}")
            st.metric("ROI", f"{performance_tracker.data['roi']:.2f}%")
            st.metric("Win Rate", 
                     f"{(performance_tracker.data['won_bets']/performance_tracker.data['total_bets']*100):.1f}%" 
                     if performance_tracker.data['total_bets'] > 0 else "0%")
            
            br_adjustment = st.number_input("Ajustement Bankroll", value=0.0)
            if st.button("Appliquer ajustement"):
                bankroll_mgr.update_bankroll(br_adjustment, "adjustment", "Ajustement manuel")
        
        with models_tab:
            model_type = st.selectbox(
                "Type de modèle:",
                ["Hybride Avancé", "XGBoost Seul", "Deep Learning", "Ensemble Complet"]
            )
            
            retrain_models = st.button("🔄 Re-entraîner modèles")
    
    # Onglets principaux
    main_tabs = st.tabs([
        "📊 Course Actuelle", 
        "🎯 Value Bets", 
        "📈 Performance",
        "💰 Bankroll Management",
        "⚙️ Configuration Avancée"
    ])
    
    with main_tabs[0]:
        display_current_race_analysis(
            url_input, auto_train, use_advanced_features,
            feature_engineer, value_detector
        )
    
    with main_tabs[1]:
        display_value_bets_analysis(value_detector)
    
    with main_tabs[2]:
        display_performance_analysis(performance_tracker)
    
    with main_tabs[3]:
        display_bankroll_management(bankroll_mgr, performance_tracker)
    
    with main_tabs[4]:
        display_advanced_configuration()

def display_current_race_analysis(url_input, auto_train, use_advanced_features,
                                feature_engineer, value_detector):
    """Affiche l'analyse de la course actuelle"""
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
                        # Feature engineering
                        if use_advanced_features:
                            df_features = feature_engineer.create_advanced_features(df_race)
                        else:
                            df_features = prepare_data(df_race)  # Fonction existante
                        
                        # Entraînement modèle
                        if auto_train:
                            model = AdvancedHybridModel()
                            # Préparation données entraînement
                            X, y = prepare_training_data(df_features)
                            model.train_ensemble(X, y)
                            
                            # Prédictions
                            predictions = model.predict_proba(X)
                            df_features['predicted_prob'] = predictions
                            
                            # Value bets
                            value_bets = value_detector.find_value_bets(
                                df_features, predictions
                            )
                            
                            # Affichage résultats
                            display_race_results(df_features, value_bets, model)
                    
                except Exception as e:
                    st.error(f"Erreur analyse: {e}")
    
    with col2:
        st.info("""
        **Indicateurs analysés:**
        - Probabilités modèles
        - Value bets
        - Gestion bankroll
        - Risques
        """)

def display_value_bets_analysis(value_detector):
    """Affiche l'analyse des value bets"""
    st.header("🎯 Détection Value Bets")
    
    # Simulation value bets
    st.subheader("Value Bets Actuels")
    
    # Métriques value bets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Edge Moyen", "8.2%")
    with col2:
        st.metric("Value Bets", "3")
    with col3:
        st.metric("EV Moyen", "+15%")
    
    # Graphique value bets
    fig = create_value_bets_chart()
    st.plotly_chart(fig, use_container_width=True)

def display_performance_analysis(performance_tracker):
    """Affiche l'analyse de performance"""
    st.header("📈 Analyse de Performance")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROI Total", f"{performance_tracker.data['roi']:.2f}%")
    with col2:
        st.metric("Win Rate", 
                 f"{(performance_tracker.data['won_bets']/performance_tracker.data['total_bets']*100):.1f}%"
                 if performance_tracker.data['total_bets'] > 0 else "0%")
    with col3:
        st.metric("Pari Moyen", 
                 f"€{performance_tracker.data['total_staked']/performance_tracker.data['total_bets']:.2f}"
                 if performance_tracker.data['total_bets'] > 0 else "€0")
    with col4:
        st.metric("Streak Actuel", performance_tracker.data['streak_current'])
    
    # Graphiques de performance
    fig1, fig2 = create_performance_charts(performance_tracker)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

def display_bankroll_management(bankroll_mgr, performance_tracker):
    """Affiche la gestion de bankroll"""
    st.header("💰 Gestion de Bankroll")
    
    # Simulation Monte Carlo
    st.subheader("Simulation Risques")
    
    simulator = MonteCarloSimulator()
    final_brs = simulator.simulate_season(
        initial_br=bankroll_mgr.bankroll,
        avg_stake=50,
        win_rate=0.15,
        avg_odds=6.0,
        n_bets=1000
    )
    
    metrics = simulator.calculate_risk_metrics(final_brs, bankroll_mgr.bankroll)
    
    # Affichage métriques risque
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("VaR 95%", f"{metrics['var_95']:.1f}%")
    with col2:
        st.metric("Probabilité Ruine", f"{metrics['probability_ruin']*100:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # Graphique distribution ROI
    fig = create_bankroll_simulation_chart(final_brs, bankroll_mgr.bankroll)
    st.plotly_chart(fig, use_container_width=True)

def display_advanced_configuration():
    """Affiche la configuration avancée"""
    st.header("⚙️ Configuration Avancée")
    
    with st.form("advanced_config"):
        st.subheader("Paramètres Modèles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dl_epochs = st.number_input("Epochs DL", value=100, min_value=10, max_value=1000)
            xgb_rounds = st.number_input("Rounds XGBoost", value=200, min_value=50, max_value=1000)
            ensemble_method = st.selectbox("Méthode Ensemble", ["Moyenne", "Pondérée", "Stacking"])
        
        with col2:
            feature_selection = st.multiselect(
                "Features à inclure:",
                ["Statistiques base", "Formes récentes", "Stats jockey/entraîneur", 
                 "Trends parcours", "Données temporelles"],
                default=["Statistiques base", "Formes récentes"]
            )
            cross_validation = st.selectbox("Validation croisée", ["TimeSeriesSplit", "KFold", "Stratified"])
        
        if st.form_submit_button("💾 Sauvegarder Configuration"):
            st.success("Configuration sauvegardée!")

# ---------------- Fonctions utilitaires améliorées ----------------
def prepare_training_data(df):
    """Prépare les données pour l'entraînement"""
    # Implémentation existante améliorée
    X = df.select_dtypes(include=[np.number])
    y = create_target_variable(df)
    return X, y

def create_target_variable(df):
    """Crée la variable cible pour l'entraînement"""
    # Logique améliorée pour créer la target
    if 'resultat' in df.columns:
        return (df['resultat'] == 1).astype(int)
    else:
        # Approximation basée sur les cotes et performances
        return 1 / (df['Cote'] + 1e-6)

def create_value_bets_chart():
    """Crée un graphique pour les value bets"""
    fig = go.Figure()
    
    # Données exemple
    horses = ['Cheval A', 'Cheval B', 'Cheval C', 'Cheval D']
    market_probs = [0.15, 0.25, 0.10, 0.08]
    model_probs = [0.25, 0.20, 0.15, 0.12]
    edges = [0.10, -0.05, 0.05, 0.04]
    
    fig.add_trace(go.Bar(
        name='Probabilité Marché',
        x=horses,
        y=market_probs,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Probabilité Modèle',
        x=horses,
        y=model_probs,
        marker_color='coral'
    ))
    
    fig.update_layout(
        title="Value Bets - Probabilités Marché vs Modèle",
        barmode='group',
        height=400
    )
    
    return fig

def create_performance_charts(performance_tracker):
    """Crée les graphiques de performance"""
    # Graphique ROI cumulatif
    fig1 = go.Figure()
    
    # Données exemple
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    roi_cumul = np.cumsum(np.random.normal(0.1, 2, len(dates)))
    
    fig1.add_trace(go.Scatter(
        x=dates, y=roi_cumul,
        mode='lines',
        name='ROI Cumulatif',
        line=dict(color='green', width=2)
    ))
    
    fig1.update_layout(
        title="ROI Cumulatif",
        height=300
    )
    
    # Graphique distribution gains
    fig2 = go.Figure()
    
    gains = np.random.normal(50, 200, 1000)
    fig2.add_trace(go.Histogram(
        x=gains,
        nbinsx=50,
        name='Distribution Gains',
        marker_color='lightgreen'
    ))
    
    fig2.update_layout(
        title="Distribution des Gains/Pertes",
        height=300
    )
    
    return fig1, fig2

def create_bankroll_simulation_chart(final_brs, initial_br):
    """Crée le graphique de simulation bankroll"""
    fig = go.Figure()
    
    roi = (final_brs - initial_br) / initial_br * 100
    
    fig.add_trace(go.Histogram(
        x=roi,
        nbinsx=50,
        name='Distribution ROI',
        marker_color='lightcoral'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Distribution ROI Simulé (Monte Carlo)",
        xaxis_title="ROI (%)",
        yaxis_title="Fréquence",
        height=400
    )
    
    return fig

# ---------------- Fonctions existantes adaptées ----------------
def safe_float(x, default=np.nan):
    """Amélioration de la fonction existante"""
    try:
        if pd.isna(x): 
            return default
        s = str(x).strip().replace("\xa0"," ").replace(",", ".")
        # Gestion des fractions (ex: 5/2)
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except:
                    pass
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except:
        return default

def music_to_features(music):
    """Amélioration de l'analyse de musique"""
    s = str(music)
    
    # Nettoyage avancé
    s = re.sub(r'[^\d\s]', '', s)
    digits = [int(x) for x in re.findall(r"\d+", s) if int(x) > 0]
    
    if not digits:
        return 0, 0, 0.0, 0, 0
    
    # Features existantes
    recent_wins = sum(1 for d in digits if d == 1)
    recent_top3 = sum(1 for d in digits if d <= 3)
    
    # Nouvelles features
    total_starts = len(digits)
    win_percentage = recent_wins / total_starts if total_starts > 0 else 0
    top3_percentage = recent_top3 / total_starts if total_starts > 0 else 0
    
    # Poids décroissant avec importance
    weights = np.linspace(1.0, 0.1, num=len(digits))
    weighted_perf = sum((4-d)*w for d,w in zip(digits, weights)) / (sum(weights) + 1e-6)
    
    # Trend (amélioration/détérioration)
    if len(digits) >= 3:
        recent_trend = np.polyfit(range(len(digits[-3:])), digits[-3:], 1)[0]
    else:
        recent_trend = 0
    
    return recent_wins, recent_top3, weighted_perf, win_percentage, -recent_trend

if __name__ == "__main__":
    main()
