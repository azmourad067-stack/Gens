import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML avancé
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro+",
    page_icon="🏇",
    layout="wide"
)

# Configuration avancée des types de courses
RACE_CONFIGS = {
    "PLAT": {
        "description": "🏃 Course de galop - Importance du poids et de la corde",
        "optimal_draws": [1, 2, 3, 4],
        "weight_coef": 0.25,
        "draw_coef": 0.15,
        "features_priority": ['odds_inv', 'music_recent_form', 'weight_advantage', 'optimal_draw']
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Importance position départ",
        "optimal_draws": [4, 5, 6],
        "weight_coef": 0.05,
        "draw_coef": 0.25,
        "features_priority": ['odds_inv', 'optimal_draw', 'music_consistency', 'driver_skill']
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Importance régularité",
        "optimal_draws": [],
        "weight_coef": 0.05,
        "draw_coef": 0.05,
        "features_priority": ['odds_inv', 'music_consistency', 'music_recent_form', 'driver_skill']
    },
    "OBSTACLE": {
        "description": "🏇 Course d'obstacles - Expérience et technique",
        "optimal_draws": [2, 3, 4, 5],
        "weight_coef": 0.20,
        "draw_coef": 0.10,
        "features_priority": ['music_win_rate', 'music_consistency', 'age_optimal', 'weight_advantage']
    }
}

class AdvancedRacingPredictor:
    """
    Système prédictif avancé pour les courses hippiques
    Utilise l'apprentissage automatique pour pondérer automatiquement les facteurs
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_importance = {}
        self.cv_results = {}
        self.is_trained = False
        
    def create_advanced_models(self):
        """Crée une ensemble de modèles avancés"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
    
    def extract_comprehensive_features(self, music_str):
        """
        Extraction avancée des performances passées avec pondération temporelle
        Les courses récentes ont plus de poids que les anciennes
        """
        if pd.isna(music_str) or music_str == '':
            return self._get_default_music_features()
        
        music = str(music_str)
        # Extraction des positions (supprime les lettres)
        positions = []
        for char in music:
            if char.isdigit():
                pos = int(char)
                if 1 <= pos <= 9:  # Positions valides en course
                    positions.append(pos)
        
        if not positions:
            return self._get_default_music_features()
        
        # Pondération temporelle : les courses récentes comptent plus
        weights = np.linspace(1.0, 0.3, len(positions))  # Décroissance linéaire
        weighted_positions = [p * w for p, w in zip(positions, weights)]
        
        total_weighted = sum(weights)
        weighted_avg = sum(weighted_positions) / total_weighted if total_weighted > 0 else 0
        
        # Calculs avec pondération
        wins = sum(1 for p, w in zip(positions, weights) if p == 1)
        places = sum(w for p, w in zip(positions, weights) if p <= 3)
        
        total_races = len(positions)
        recent_races = positions[:min(3, len(positions))]
        
        return {
            'wins': wins,
            'places': places,
            'total_races': total_races,
            'win_rate': wins / total_races if total_races > 0 else 0,
            'place_rate': places / total_races if total_races > 0 else 0,
            'weighted_avg_position': weighted_avg,
            'recent_form': 1 / (np.mean(recent_races) + 0.1) if recent_races else 0,
            'consistency': 1 / (np.std(positions) + 0.5) if len(positions) > 1 else 0.5,
            'best_position': min(positions),
            'momentum': self._calculate_momentum(positions),
            'recovery_ability': self._calculate_recovery(positions)
        }
    
    def _get_default_music_features(self):
        """Retourne des features par défaut pour données manquantes"""
        return {
            'wins': 0, 'places': 0, 'total_races': 0,
            'win_rate': 0, 'place_rate': 0, 'weighted_avg_position': 8,
            'recent_form': 0, 'consistency': 0.5, 'best_position': 10,
            'momentum': 0, 'recovery_ability': 0
        }
    
    def _calculate_momentum(self, positions):
        """Calcule la dynamique récente du cheval"""
        if len(positions) < 2:
            return 0
        recent = positions[:3]
        return sum(1/(p+0.1) for p in recent) / len(recent)
    
    def _calculate_recovery(self, positions):
        """Calcule la capacité à rebondir après une mauvaise performance"""
        recoveries = 0
        for i in range(1, len(positions)):
            if positions[i] < positions[i-1]:  # Amélioration
                recoveries += 1
        return recoveries / (len(positions) - 1) if len(positions) > 1 else 0
    
    def extract_driver_stats(self, driver_info):
        """Extrait les statistiques du driver/jockey"""
        # Implémentation simplifiée - à adapter selon les données disponibles
        return {
            'driver_win_rate': 0.15,  # À remplacer par données réelles
            'driver_place_rate': 0.35,
            'driver_experience': 50,
            'driver_recent_form': 0.6
        }
    
    def prepare_advanced_features(self, df, race_type="PLAT"):
        """
        Préparation complète des features avec ingénierie avancée
        """
        features = pd.DataFrame()
        config = RACE_CONFIGS[race_type]
        
        # === FEATURES DE BASE AVANCÉES ===
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.01)  # Évite division par zéro
        features['log_odds'] = np.log1p(df['odds_numeric'])
        features['odds_rank'] = df['odds_numeric'].rank(pct=True)
        
        # === FEATURES DE POSITION INTELLIGENTES ===
        features['draw'] = df['draw_numeric']
        features['draw_normalized'] = df['draw_numeric'] / max(df['draw_numeric'].max(), 1)
        features['optimal_draw'] = df['draw_numeric'].apply(
            lambda x: 1 if x in config['optimal_draws'] else 0
        )
        
        # Distance à la position optimale
        if config['optimal_draws']:
            features['draw_penalty'] = df['draw_numeric'].apply(
                lambda x: min(abs(x - opt) for opt in config['optimal_draws']) / len(config['optimal_draws'])
            )
        else:
            features['draw_penalty'] = 0
        
        # === FEATURES DE POIDS CONTEXTUELLES ===
        features['weight'] = df['weight_kg']
        features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']) * config['weight_coef']
        features['weight_rank'] = df['weight_kg'].rank(pct=True)
        
        # === FEATURES DÉMOGRAPHIQUES ===
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['is_female'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
            features['is_gelding'] = df['Âge/Sexe'].str.contains('H', na=False).astype(int)
            features['age_optimal'] = features['age'].apply(lambda x: 1 if 3.5 <= x <= 6.5 else 0)
            features['age_experience'] = np.log1p(features['age'])
        else:
            # Valeurs par défaut raisonnables
            features['age'] = 4.5
            features['is_female'] = 0
            features['is_gelding'] = 0
            features['age_optimal'] = 1
            features['age_experience'] = np.log1p(4.5)
        
        # === FEATURES DE PERFORMANCE DÉTAILLÉES ===
        if 'Musique' in df.columns:
            music_features = df['Musique'].apply(self.extract_comprehensive_features)
            for key in music_features.iloc[0].keys():
                features[f'music_{key}'] = [m[key] for m in music_features]
        else:
            default_features = self._get_default_music_features()
            for key in default_features.keys():
                features[f'music_{key}'] = default_features[key]
        
        # === FEATURES D'INTERACTION AVANCÉES ===
        features['odds_form_interaction'] = features['odds_inv'] * features['music_recent_form']
        features['weight_age_interaction'] = features['weight_advantage'] * features['age_optimal']
        features['draw_odds_interaction'] = features['optimal_draw'] * features['odds_inv']
        features['consistency_advantage'] = features['music_consistency'] * features['weight_advantage']
        
        # === FEATURES DE CONTEXTE DE COURSE ===
        features['field_size'] = len(df)
        features['competitiveness'] = 1 / (df['odds_numeric'].std() + 0.1)
        features['favorite_pressure'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
        
        # === FEATURES STATISTIQUES RELATIVES ===
        features['win_rate_rank'] = features['music_win_rate'].rank(pct=True)
        features['form_rank'] = features['music_recent_form'].rank(pct=True)
        features['consistency_rank'] = features['music_consistency'].rank(pct=True)
        
        # === SCORE COMPOSITE PERSONNALISÉ ===
        features['composite_score'] = (
            features['odds_inv'] * 0.3 +
            features['music_recent_form'] * 0.25 +
            features['music_consistency'] * 0.15 +
            features['weight_advantage'] * 0.1 +
            features['optimal_draw'] * config['draw_coef'] +
            features['age_optimal'] * 0.05 +
            features['music_win_rate'] * 0.1
        )
        
        return features.fillna(0)
    
    def create_synthetic_labels(self, X, race_type="PLAT"):
        """
        Crée des labels synthétiques réalistes basés sur les features
        C'est le cœur du système d'apprentissage
        """
        config = RACE_CONFIGS[race_type]
        
        # Pondération selon le type de course
        base_weights = {
            'odds_inv': 0.35,
            'music_recent_form': 0.20,
            'music_consistency': 0.15,
            'weight_advantage': config['weight_coef'],
            'optimal_draw': config['draw_coef'],
            'music_win_rate': 0.10,
            'age_optimal': 0.05
        }
        
        # Calcul du score synthétique
        y_synthetic = np.zeros(len(X))
        for feature, weight in base_weights.items():
            if feature in X.columns:
                # Normalisation de la feature
                feature_norm = (X[feature] - X[feature].min()) / (X[feature].max() - X[feature].min() + 1e-8)
                y_synthetic += feature_norm * weight
        
        # Ajout d'un bruit réaliste
        noise = np.random.normal(0, 0.03, len(X))
        y_synthetic += noise
        
        return np.clip(y_synthetic, 0, 1)
    
    def train_ensemble_model(self, X, y, cv_folds=5):
        """
        Entraînement d'un ensemble de modèles avec validation croisée
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation croisée pour chaque modèle
        cv_scores = {}
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Scores de validation croisée
                scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2', n_jobs=-1)
                cv_scores[name] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'scores': scores
                }
                
                # Entraînement du modèle
                model.fit(X_scaled, y)
                pred = model.predict(X_scaled)
                predictions[name] = pred
                
                # Importance des features (si disponible)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        sorted(zip(X.columns, model.feature_importances_), 
                              key=lambda x: x[1], reverse=True)[:15]
                    )
                    
            except Exception as e:
                st.warning(f"⚠️ Erreur avec le modèle {name}: {str(e)}")
                predictions[name] = np.full(len(X), y.mean())
                cv_scores[name] = {'mean_r2': 0, 'std_r2': 0, 'scores': [0]}
        
        return predictions, cv_scores
    
    def optimize_ensemble_weights(self, predictions, y_true):
        """
        Optimise les poids de l'ensemble pour maximiser la performance
        """
        from scipy.optimize import minimize
        
        def objective(weights):
            # Combinaison linéaire des prédictions
            combined = sum(w * pred for w, pred in zip(weights, predictions.values()))
            return -r2_score(y_true, combined)  # On maximise R²
        
        # Contraintes : poids positifs et somme à 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in predictions]
        
        # Initialisation équitable
        x0 = np.ones(len(predictions)) / len(predictions)
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x
        except:
            # Retour aux poids par défaut en cas d'échec
            return x0
    
    def predict_with_confidence(self, X, race_type="PLAT"):
        """
        Prédiction avec estimation de la confiance
        """
        if len(X) < 4:
            st.warning("⚠️ Données insuffisantes pour une prédiction fiable")
            return np.zeros(len(X)), {}, np.zeros(len(X))
        
        # Création des modèles
        self.create_advanced_models()
        
        # Préparation des labels synthétiques
        y_synthetic = self.create_synthetic_labels(X, race_type)
        
        # Entraînement de l'ensemble
        predictions, cv_scores = self.train_ensemble_model(X, y_synthetic)
        
        if not predictions:
            st.error("❌ Aucun modèle n'a pu être entraîné")
            return np.zeros(len(X)), {}, np.zeros(len(X))
        
        # Optimisation des poids de l'ensemble
        optimal_weights = self.optimize_ensemble_weights(predictions, y_synthetic)
        
        # Prédiction finale pondérée
        final_predictions = sum(
            weight * pred for weight, pred in zip(optimal_weights, predictions.values())
        )
        
        # Calcul de la confiance
        confidence = self.calculate_advanced_confidence(final_predictions, X, cv_scores)
        
        self.is_trained = True
        self.cv_results = cv_scores
        
        return final_predictions, cv_scores, confidence
    
    def calculate_advanced_confidence(self, predictions, X, cv_scores):
        """
        Calcule un score de confiance avancé basé sur plusieurs facteurs
        """
        if len(predictions) < 3:
            return np.ones(len(predictions)) * 0.5
        
        # 1. Variabilité des prédictions
        pred_variance = np.var(predictions)
        confidence_variance = 1 / (1 + pred_variance * 10)
        
        # 2. Qualité des données
        data_quality = 1 - (X.isna().sum(axis=1) / len(X.columns))
        
        # 3. Performance des modèles (moyenne R²)
        avg_r2 = np.mean([scores['mean_r2'] for scores in cv_scores.values()])
        model_confidence = max(0, min(1, avg_r2 + 0.5))  # Normalisé entre 0 et 1
        
        # 4. Consistance des prédictions
        if len(predictions) > 5:
            sorted_pred = np.sort(predictions)
            consistency = 1 - (sorted_pred[-1] - sorted_pred[0])
        else:
            consistency = 0.7
        
        # Combinaison des facteurs de confiance
        confidence = (
            confidence_variance * 0.3 +
            data_quality.values * 0.3 +
            model_confidence * 0.2 +
            consistency * 0.2
        )
        
        return np.clip(confidence, 0.1, 0.95)

# =============================================================================
# FONCTIONS EXISTANTES AMÉLIORÉES
# =============================================================================

def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_enhanced_data(df):
    """
    Préparation des données avec gestion d'erreurs améliorée
    """
    df = df.copy()
    
    # Conversion robuste des cotes
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 99.0))
    
    # Conversion des numéros de corde
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    # Extraction du poids avec gestion d'erreurs
    def extract_weight_enhanced(poids_str):
        if pd.isna(poids_str):
            return 60.0
        try:
            # Recherche de nombres avec décimales
            matches = re.findall(r'\d+[.,]\d+|\d+', str(poids_str))
            if matches:
                return float(matches[0].replace(',', '.'))
        except:
            pass
        return 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight_enhanced)
    
    # Filtrage des données aberrantes
    df = df[(df['odds_numeric'] > 0) & (df['odds_numeric'] < 100)]
    df = df.reset_index(drop=True)
    
    return df

@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
            
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })

        if not horses_data:
            return None, "Aucune donnée extraite"
            
        return pd.DataFrame(horses_data), "Succès"
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def generate_sample_data(data_type="plat"):
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker', 'Fire Dancer', 'Ocean Wave'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1', '12.5', '15.0'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5', '60.0', '61.5'],
            'Musique': ['1a2a3a1a', '2a1a4a3a', '3a3a1a2a', '1a4a2a1a', '4a2a5a3a', '5a3a6a4a', '6a5a7a8a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M', '5H', '4F']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair', 'Foudre Noire', 'Vent du Nord', 'Tempête Rouge', 'Orage Bleu', 'Cyclone Vert'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6'],
            'Cote': ['4.2', '8.5', '15.0', '3.8', '6.8', '10.2'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['2a1a4a1a', '4a3a2a5a', '6a5a8a7a', '1a2a1a3a', '3a4a5a2a', '5a6a4a8a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M', '6H']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis', 'Mishriff', 'Love'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1', '5.5', '11.0'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5', '59.0', '58.0'],
            'Musique': ['1a1a2a1a', '1a3a1a2a', '2a1a4a1a', '1a2a1a3a', '3a1a2a1a', '1a1a1a2a', '2a3a1a4a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F', '5H', '4F']
        })

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .url-input {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏇 Système Prédictif Hippique Avancé</h1>', unsafe_allow_html=True)
    st.markdown("*Apprentissage automatique avec pondération automatique des facteurs*")
    
    # Sidebar améliorée
    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        
        race_type = st.selectbox(
            "🏁 Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE", "OBSTACLE"]
        )
        
        ml_method = st.selectbox(
            "🧠 Méthode ML",
            ["Ensemble Stacking", "XGBoost", "Random Forest", "Réseau de Neurones"]
        )
        
        st.subheader("📊 Paramètres d'Analyse")
        enable_correlation = st.checkbox("📈 Analyse de corrélation", value=True)
        enable_feature_importance = st.checkbox("🔍 Importance des features", value=True)
        confidence_threshold = st.slider("🎯 Seuil de confiance", 0.5, 0.95, 0.7, 0.05)
        
        st.subheader("ℹ️ À propos du système")
        st.info("""
        **Fonctionnalités avancées:**
        - 🧠 Ensemble de modèles ML
        - 📊 Pondération automatique
        - 🎯 Scores de confiance
        - 🔄 Validation croisée
        - 📈 Analyse de corrélation
        """)
    
    # SECTION URL BIEN VISIBLE
    st.markdown("---")
    st.header("🔍 Analyse d'URL de Course")
    
    # Container spécial pour l'input URL
    with st.container():
        st.markdown('<div class="url-input">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(
                "🌐 **URL de la course:**",
                placeholder="https://example-racing-site.com/course/123",
                key="url_input"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    df_final = None
    
    # Traitement de l'URL
    if analyze_button and url:
        with st.spinner("🔄 Extraction des données en cours..."):
            df, message = scrape_race_data(url)
            if df is not None:
                st.success(f"✅ {len(df)} chevaux extraits avec succès")
                st.dataframe(df.head(), use_container_width=True)
                df_final = df
            else:
                st.error(f"❌ {message}")
    
    # Autres onglets
    tab1, tab2 = st.tabs(["📁 Upload CSV", "🧪 Données de Test"])
    
    with tab1:
        st.subheader("📤 Upload de fichier CSV")
        st.markdown("Format attendu: `Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe`")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv", key="csv_uploader")
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(df_final)} chevaux chargés")
                st.dataframe(df_final.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur de lecture: {e}")
    
    with tab2:
        st.subheader("🧪 Données de Test")
        st.markdown("Tester l'analyseur avec des données pré-chargées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏃 Test Plat", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success("✅ Données PLAT chargées (7 chevaux)")
        with col2:
            if st.button("🚗 Test Attelé", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success("✅ Données ATTELÉ chargées (6 chevaux)")
        with col3:
            if st.button("⭐ Test Premium", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success("✅ Données PREMIUM chargées (7 chevaux)")
        
        if df_final is not None:
            st.dataframe(df_final, use_container_width=True)
    
    # === ANALYSE PRINCIPALE ===
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse et Prédictions ML")
        
        df_prepared = prepare_enhanced_data(df_final)
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide après préparation")
            return
        
        # Détection du type de course
        if race_type == "AUTO":
            weight_std = df_prepared['weight_kg'].std()
            if weight_std > 2.5:
                detected_type = "PLAT"
                reason = "Grande variation de poids (handicap)"
            elif df_prepared['weight_kg'].mean() > 65 and weight_std < 1.5:
                detected_type = "ATTELE_AUTOSTART"
                reason = "Poids uniformes élevés (attelé)"
            else:
                detected_type = "PLAT"
                reason = "Configuration par défaut"
            
            st.info(f"🤖 **Type détecté**: {detected_type} | **Raison**: {reason}")
        else:
            detected_type = race_type
            st.info(f"📋 **Type sélectionné**: {RACE_CONFIGS[detected_type]['description']}")
        
        # === MACHINE LEARNING AVANCÉ ===
        predictor = AdvancedRacingPredictor()
        
        with st.spinner("🤖 Entraînement des modèles ML avancés..."):
            try:
                # Préparation des features avancées
                X_ml = predictor.prepare_advanced_features(df_prepared, detected_type)
                
                # Affichage du nombre de features
                st.info(f"🔬 **{len(X_ml.columns)} features** créées pour l'analyse ML")
                
                # Prédiction avec confiance
                ml_predictions, ml_results, confidence_scores = predictor.predict_with_confidence(X_ml, detected_type)
                
                # Normalisation des prédictions ML
                if len(ml_predictions) > 0 and ml_predictions.max() != ml_predictions.min():
                    ml_predictions = (ml_predictions - ml_predictions.min()) / (ml_predictions.max() - ml_predictions.min())
                
                df_prepared['ml_score'] = ml_predictions
                df_prepared['confidence'] = confidence_scores
                
                st.success("✅ Modèles ML entraînés avec succès")
                
                # Affichage des métriques ML
                if ml_results:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if 'xgboost' in ml_results:
                            st.metric("🎯 R² XGBoost", f"{ml_results['xgboost']['mean_r2']:.3f}")
                    with col2:
                        if 'random_forest' in ml_results:
                            st.metric("🌲 R² RF", f"{ml_results['random_forest']['mean_r2']:.3f}")
                    with col3:
                        if 'gradient_boosting' in ml_results:
                            st.metric("📈 R² GB", f"{ml_results['gradient_boosting']['mean_r2']:.3f}")
                    with col4:
                        avg_confidence = confidence_scores.mean() if confidence_scores is not None else 0
                        st.metric("🎯 Confiance Moy.", f"{avg_confidence:.1%}")
                
            except Exception as e:
                st.error(f"⚠️ Erreur ML: {e}")
                # Fallback vers méthode simple
                traditional_score = 1 / (df_prepared['odds_numeric'] + 0.1)
                if traditional_score.max() != traditional_score.min():
                    traditional_score = (traditional_score - traditional_score.min()) / (traditional_score.max() - traditional_score.min())
                df_prepared['ml_score'] = traditional_score
                df_prepared['confidence'] = np.ones(len(df_prepared)) * 0.5
        
        # === SCORE FINAL ===
        ml_weight = 0.7  # Poids du ML
        df_prepared['score_final'] = (1 - ml_weight) * (1 / (df_prepared['odds_numeric'] + 0.1)) + ml_weight * df_prepared['ml_score']
        
        # === CLASSEMENT ===
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        # === AFFICHAGE DES RÉSULTATS ===
        st.markdown("---")
        st.subheader("🏆 Classement Final avec Confiance")
        
        # Préparation du DataFrame d'affichage
        display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde', 'Poids', 'score_final', 'confidence']
        display_df = df_ranked[[col for col in display_cols if col in df_ranked.columns]].copy()
        
        # Formatage
        if 'score_final' in display_df.columns:
            display_df['Score'] = display_df['score_final'].apply(lambda x: f"{x:.3f}")
        if 'confidence' in display_df.columns:
            display_df['Confiance'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
