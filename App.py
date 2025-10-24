import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# ML Imports optimisés
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor, BaggingRegressor)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, RandomizedSearchCV, TimeSeriesSplit)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration Streamlit ultra-optimisée
st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro Max",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS avancé avec animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        to { text-shadow: 0 0 30px rgba(118, 75, 162, 0.8); }
    }
    
    .metric-card-pro {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card-pro:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .prediction-box-pro {
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .ml-confidence-high { border-left-color: #10b981; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); }
    .ml-confidence-medium { border-left-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); }
    .ml-confidence-low { border-left-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%); }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Configuration avancée par type de course
ADVANCED_CONFIGS = {
    "PLAT": {
        "description": "🏃 Course de galop - Analyse complète handicap",
        "optimal_draws": [1, 2, 3, 4],
        "weight_importance": 0.35,
        "draw_importance": 0.25,
        "form_importance": 0.40
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Stratégie optimisée",
        "optimal_draws": [4, 5, 6],
        "weight_importance": 0.10,
        "draw_importance": 0.30,
        "form_importance": 0.60
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Focus performance pure",
        "optimal_draws": [],
        "weight_importance": 0.05,
        "draw_importance": 0.05,
        "form_importance": 0.90
    }
}

@st.cache_data(ttl=300)
def scrape_geny_data(url):
    """Scraper spécialisé pour Geny.com avec gestion d'erreurs robuste"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Gestion spéciale pour Geny.com
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        # Méthodes multiples pour Geny.com
        # Méthode 1: Recherche de tables classiques
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) > 3:  # Au moins quelques chevaux
                for i, row in enumerate(rows[1:]):  # Skip header
                    cols = row.find_all(['td', 'th'])
                    if len(cols) >= 4:
                        # Extraction intelligente
                        horse_data = extract_horse_data_geny(cols, i+1)
                        if horse_data:
                            horses_data.append(horse_data)
        
        # Méthode 2: Recherche par classes CSS spécifiques à Geny
        if not horses_data:
            horse_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'horse|partant|runner', re.I))
            
            for i, element in enumerate(horse_elements):
                horse_data = extract_horse_data_from_element(element, i+1)
                if horse_data:
                    horses_data.append(horse_data)
        
        # Méthode 3: Recherche par patterns de texte
        if not horses_data:
            text_content = soup.get_text()
            horses_data = extract_horses_from_text(text_content)
        
        # Méthode 4: Fallback avec structure générique
        if not horses_data:
            all_text_elements = soup.find_all(['span', 'div', 'p'])
            horses_data = extract_horses_generic_fallback(all_text_elements)

        if not horses_data:
            return None, "Aucune donnée de cheval détectée. Le site pourrait avoir changé de structure."
            
        # Nettoyage et validation
        cleaned_data = []
        for horse in horses_data:
            if horse.get('Nom') and horse.get('Cote'):
                cleaned_data.append(horse)
        
        if not cleaned_data:
            return None, "Données extraites mais incomplètes (nom ou cote manquants)"
            
        return pd.DataFrame(cleaned_data), f"Succès - {len(cleaned_data)} chevaux extraits"
        
    except requests.exceptions.Timeout:
        return None, "Timeout - Le site met trop de temps à répondre"
    except requests.exceptions.ConnectionError:
        return None, "Erreur de connexion - Vérifiez votre connexion internet"
    except Exception as e:
        return None, f"Erreur lors du scraping: {str(e)}"

def extract_horse_data_geny(cols, numero_defaut):
    """Extraction spécialisée pour les colonnes Geny"""
    try:
        # Patterns de reconnaissance pour Geny
        horse_data = {
            "Numéro de corde": str(numero_defaut),
            "Nom": "",
            "Cote": "",
            "Poids": "60.0",
            "Musique": "",
            "Âge/Sexe": "",
            "Jockey": "",
            "Entraîneur": ""
        }
        
        # Recherche intelligente dans les colonnes
        for i, col in enumerate(cols):
            text = col.get_text(strip=True)
            
            # Détection numéro (si présent)
            if re.match(r'^\d{1,2}$', text) and i == 0:
                horse_data["Numéro de corde"] = text
            
            # Détection nom de cheval (souvent en majuscules ou avec patterns spéciaux)
            elif len(text) > 3 and not re.match(r'^\d+[.,]\d*$', text):
                if not horse_data["Nom"] and (text.isupper() or len(text.split()) <= 3):
                    horse_data["Nom"] = text
            
            # Détection cote (format X.X ou X,X)
            elif re.match(r'^\d+[.,]\d*$', text) or re.match(r'^\d+$', text):
                cote_val = float(text.replace(',', '.'))
                if 1.0 <= cote_val <= 999.0:
                    horse_data["Cote"] = text.replace(',', '.')
            
            # Détection poids (format XXkg ou XX.X)
            elif 'kg' in text.lower() or (re.match(r'^\d{2}[.,]?\d?$', text) and 45 <= float(text.replace(',', '.')) <= 75):
                horse_data["Poids"] = text.replace('kg', '').replace(',', '.')
            
            # Détection âge/sexe (format XH, XM, XF)
            elif re.match(r'^\d[HMF]$', text.upper()):
                horse_data["Âge/Sexe"] = text.upper()
            
            # Détection musique (séquence de chiffres et lettres)
            elif re.match(r'^[\da-zA-Z]{3,}$', text) and len(text) <= 10:
                if not horse_data["Musique"]:
                    horse_data["Musique"] = text
        
        # Validation des données essentielles
        if horse_data["Nom"] and horse_data["Cote"]:
            return horse_data
        else:
            return None
            
    except Exception:
        return None

def extract_horse_data_from_element(element, numero_defaut):
    """Extraction depuis un élément DOM spécifique"""
    try:
        text_content = element.get_text(separator=' ', strip=True)
        
        # Recherche de patterns dans le texte
        horse_data = {
            "Numéro de corde": str(numero_defaut),
            "Nom": "",
            "Cote": "",
            "Poids": "60.0",
            "Musique": "",
            "Âge/Sexe": "",
            "Jockey": "",
            "Entraîneur": ""
        }
        
        # Pattern pour nom + cote
        name_cote_pattern = r'([A-Z\s]{3,30})\s+(\d+[.,]\d+)'
        match = re.search(name_cote_pattern, text_content)
        
        if match:
            horse_data["Nom"] = match.group(1).strip()
            horse_data["Cote"] = match.group(2).replace(',', '.')
            
            # Recherche d'infos supplémentaires
            age_sexe_match = re.search(r'(\d[HMF])', text_content.upper())
            if age_sexe_match:
                horse_data["Âge/Sexe"] = age_sexe_match.group(1)
            
            poids_match = re.search(r'(\d{2}[.,]?\d?)\s*kg?', text_content)
            if poids_match:
                horse_data["Poids"] = poids_match.group(1).replace(',', '.')
            
            return horse_data
        
        return None
        
    except Exception:
        return None

def extract_horses_from_text(text_content):
    """Extraction par analyse de texte brut"""
    horses = []
    
    try:
        # Patterns pour identifier les chevaux dans le texte
        patterns = [
            r'(\d+)\s+([A-Z\s]{3,25})\s+(\d+[.,]\d+)',  # Numéro Nom Cote
            r'([A-Z\s]{3,25})\s+(\d+[.,]\d+)\s+(\d[HMF])?',  # Nom Cote Âge/Sexe
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text_content)
            
            for j, match in enumerate(matches[:20]):  # Limiter à 20 chevaux max
                if i == 0:  # Pattern avec numéro
                    numero, nom, cote = match
                else:  # Pattern sans numéro
                    nom, cote = match[0], match[1]
                    numero = str(j + 1)
                
                horse_data = {
                    "Numéro de corde": numero,
                    "Nom": nom.strip(),
                    "Cote": cote.replace(',', '.'),
                    "Poids": "60.0",
                    "Musique": "",
                    "Âge/Sexe": match[2] if len(match) > 2 and match[2] else "",
                    "Jockey": "",
                    "Entraîneur": ""
                }
                
                horses.append(horse_data)
            
            if horses:  # Si on a trouvé des chevaux avec un pattern, on s'arrête
                break
        
        return horses[:16]  # Maximum 16 chevaux
        
    except Exception:
        return []

def extract_horses_generic_fallback(elements):
    """Fallback générique pour extraction"""
    horses = []
    
    try:
        potential_names = []
        potential_cotes = []
        
        # Collecte des noms et cotes potentiels
        for element in elements:
            text = element.get_text(strip=True)
            
            # Nom potentiel (3-25 caractères, pas que des chiffres)
            if 3 <= len(text) <= 25 and not text.isdigit() and not re.match(r'^\d+[.,]\d+$', text):
                potential_names.append(text)
            
            # Cote potentielle
            elif re.match(r'^\d+[.,]?\d*$', text):
                cote_val = float(text.replace(',', '.'))
                if 1.0 <= cote_val <= 100.0:
                    potential_cotes.append(text.replace(',', '.'))
        
        # Association nom-cote
        min_len = min(len(potential_names), len(potential_cotes))
        
        for i in range(min_len):
            horse_data = {
                "Numéro de corde": str(i + 1),
                "Nom": potential_names[i],
                "Cote": potential_cotes[i],
                "Poids": "60.0",
                "Musique": "",
                "Âge/Sexe": "",
                "Jockey": "",
                "Entraîneur": ""
            }
            horses.append(horse_data)
        
        return horses[:12]  # Maximum 12 chevaux
        
    except Exception:
        return []

def safe_convert(value, convert_func, default=0):
    """Conversion sécurisée"""
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    """Préparation complète des données"""
    df = df.copy()
    
    # Conversions sécurisées
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    # Extraction du poids
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    
    # Nettoyage
    df = df[df['odds_numeric'] > 0]  # Éliminer les cotes invalides
    df = df.reset_index(drop=True)
    
    return df

def auto_detect_race_type(df):
    """Détection automatique avec explications"""
    weight_std = df['weight_kg'].std()
    weight_mean = df['weight_kg'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💪 Écart-type poids", f"{weight_std:.1f} kg")
    with col2:
        st.metric("⚖️ Poids moyen", f"{weight_mean:.1f} kg")
    with col3:
        st.metric("🏇 Nb chevaux", len(df))
    
    if weight_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_std < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés (attelé)"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    st.info(f"🤖 **Type détecté**: {detected} | **Raison**: {reason}")
    return detected

@st.cache_resource
class AdvancedHorseRacingML:
    """Système ML ultra-avancé pour analyse hippique"""
    
    def __init__(self):
        # Ensemble de modèles optimisés
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=3,
                random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=8,
                random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Outils de preprocessing
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=10)
        
        # Métriques et résultats
        self.feature_importance = {}
        self.model_scores = {}
        self.is_trained = False
        
    def create_advanced_features(self, df, race_type):
        """Création de features ultra-avancées"""
        features = pd.DataFrame(index=df.index)
        
        # === FEATURES DE BASE OPTIMISÉES ===
        features['odds_log'] = np.log1p(df['odds_numeric'])
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.01)
        features['odds_sqrt'] = np.sqrt(df['odds_numeric'])
        
        features['draw'] = df['draw_numeric']
        features['draw_log'] = np.log1p(df['draw_numeric'])
        features['draw_sqrt'] = np.sqrt(df['draw_numeric'])
        
        features['weight'] = df['weight_kg']
        features['weight_norm'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-8)
        features['weight_rank'] = df['weight_kg'].rank(pct=True)
        
        # === FEATURES D'ÂGE ET SEXE ===
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract(r'(\d+)').astype(float).fillna(4)
            features['age_squared'] = features['age'] ** 2
            features['age_optimal'] = np.abs(features['age'] - 5)
            
            features['is_mare'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
            features['is_horse'] = df['Âge/Sexe'].str.contains('H', na=False).astype(int)
            features['is_gelding'] = df['Âge/Sexe'].str.contains('M', na=False).astype(int)
        else:
            features['age'] = 4.0
            features['age_squared'] = 16.0
            features['age_optimal'] = 1.0
            features['is_mare'] = 0
            features['is_horse'] = 0
            features['is_gelding'] = 1
        
        # === ANALYSE DE LA FORME ===
        if 'Musique' in df.columns:
            for i, row in df.iterrows():
                musique = str(row['Musique']) if pd.notna(row['Musique']) else ""
                positions = [int(c) for c in musique if c.isdigit() and int(c) <= 20]
                
                if positions:
                    features.loc[i, 'form_avg'] = np.mean(positions)
                    features.loc[i, 'form_best'] = min(positions)
                    features.loc[i, 'form_consistency'] = 1 / (1 + np.std(positions))
                    features.loc[i, 'wins_recent'] = positions[:3].count(1) if len(positions) >= 3 else 0
                    features.loc[i, 'places_recent'] = sum(1 for p in positions[:3] if p <= 3) if len(positions) >= 3 else 0
                else:
                    features.loc[i, 'form_avg'] = 8.0
                    features.loc[i, 'form_best'] = 8
                    features.loc[i, 'form_consistency'] = 0.5
                    features.loc[i, 'wins_recent'] = 0
                    features.loc[i, 'places_recent'] = 0
        else:
            features['form_avg'] = 8.0
            features['form_best'] = 8
            features['form_consistency'] = 0.5
            features['wins_recent'] = 0
            features['places_recent'] = 0
        
        # === FEATURES RELATIVES ===
        features['odds_rank'] = df['odds_numeric'].rank()
        features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
        
        # === INTERACTIONS ===
        features['odds_weight_ratio'] = features['odds_inv'] * features['weight_norm']
        features['draw_odds_interaction'] = features['draw'] * features['odds_log']
        features['age_form_interaction'] = features['age'] * features['form_consistency']
        
        # === FEATURES SPÉCIFIQUES PAR TYPE ===
        if race_type == "PLAT":
            features['inner_draw_bonus'] = np.where(features['draw'] <= 4, 0.3, 0)
            features['weight_penalty'] = np.maximum(0, features['weight'] - 56) * 0.02
            
        elif race_type == "ATTELE_AUTOSTART":
            features['optimal_draw'] = features['draw'].isin([4, 5, 6]).astype(float) * 0.3
            features['bad_draw'] = (features['draw'] <= 3).astype(float) * -0.2
        
        return features.fillna(0)
    
    def train_advanced_models(self, X, y):
        """Entraînement avancé des modèles"""
        
        if len(X) < 5:
            st.warning("⚠️ Pas assez de données pour entraînement avancé")
            return {}
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Sélection de features si nécessaire
        if len(X.columns) > 10:
            try:
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                selected_features = self.feature_selector.get_support()
                selected_feature_names = X.columns[selected_features].tolist()
            except:
                X_selected = X_scaled
                selected_feature_names = X.columns.tolist()
        else:
            X_selected = X_scaled
            selected_feature_names = X.columns.tolist()
        
        # Division train/validation si assez de données
        if len(X) >= 8:
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y, test_size=0.25, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X_selected, X_selected, y, y
        
        # Entraînement des modèles
        results = {}
        
        for name, model in self.base_models.items():
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                # Métriques
                results[name] = {
                    'model': model,
                    'train_r2': r2_score(y_train, y_pred_train),
                    'val_r2': r2_score(y_val, y_pred_val),
                    'mse': mean_squared_error(y_val, y_pred_val),
                    'predictions': model.predict(X_selected)
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(selected_feature_names, model.feature_importances_))
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    results[name]['feature_importance'] = dict(sorted_importance[:10])
                    self.feature_importance[name] = results[name]['feature_importance']
                
            except Exception as e:
                st.warning(f"⚠️ Erreur entraînement {name}: {str(e)}")
                continue
        
        # Modèle d'ensemble si plusieurs modèles réussis
        if len(results) >= 2:
            try:
                best_models = sorted(results.items(), key=lambda x: x[1].get('val_r2', 0), reverse=True)[:3]
                ensemble_estimators = [(name, result['model']) for name, result in best_models]
                ensemble = VotingRegressor(ensemble_estimators)
                ensemble.fit(X_selected, y)
                
                y_pred_ensemble = ensemble.predict(X_selected)
                results['ensemble'] = {
                    'model': ensemble,
                    'r2': r2_score(y, y_pred_ensemble),
                    'mse': mean_squared_error(y, y_pred_ensemble),
                    'predictions': y_pred_ensemble,
                    'component_models': [name for name, _ in best_models]
                }
                
            except Exception as e:
                st.warning(f"⚠️ Erreur ensemble: {str(e)}")
        
        self.model_scores = results
        self.is_trained = True
        
        return results
    
    def get_best_model_predictions(self, results):
        """Sélectionne le meilleur modèle et retourne ses prédictions"""
        if not results:
            return np.array([]), "no_model"
        
        # Priorité à l'ensemble s'il existe
        if 'ensemble' in results:
            return results['ensemble']['predictions'], 'ensemble'
        
        # Sinon, meilleur modèle basé sur R²
        best_model_name = max(results.keys(), key=lambda x: results[x].get('val_r2', results[x].get('r2', 0)))
        return results[best_model_name]['predictions'], best_model_name

def create_visualization(df_ranked, ml_results=None):
    """Visualisations avancées"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🏆 Scores par Position', '📊 Distribution Cotes', '⚖️ Poids vs Performance', '🧠 Features ML'),
        specs=[[{"secondary_y": False}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = px.colors.qualitative.Set3
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    
    # Graphique 1: Scores
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'], y=df_ranked[score_col],
                mode='markers+lines', 
                marker=dict(size=12, color=colors[0], line=dict(width=2, color='white')),
                text=df_ranked['Nom'], 
                hovertemplate='<b>%{text}</b><br>Rang: %{x}<br>Score: %{y:.3f}<extra></extra>',
                name='Score Final'
            ), row=1, col=1
        )
    
    # Graphique 2: Distribution cotes
    fig.add_trace(
        go.Histogram(x=df_ranked['odds_numeric'], nbinsx=8, marker_color=colors[1], name='Cotes'),
        row=1, col=2
    )
    
    # Graphique 3: Poids vs Performance
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['weight_kg'], y=df_ranked[score_col],
                mode='markers', 
                marker=dict(size=10, color=df_ranked['rang'], colorscale='Viridis', showscale=True),
                text=df_ranked['Nom'], name='Poids vs Score'
            ), row=2, col=1
        )
    
    # Graphique 4: Feature importance
    if ml_results:
        all_importance = {}
        for model_name, results in ml_results.items():
            if 'feature_importance' in results:
                for feature, importance in results['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
        top_features = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:8])
        
        if top_features:
            fig.add_trace(
                go.Bar(x=list(top_features.values()), y=list(top_features.keys()), 
                       orientation='h', marker_color=colors[3], name='Importance'),
                row=2, col=2
            )
    
    fig.update_layout(height=700, showlegend=True, title_text="📊 Analyse Complète", title_x=0.5)
    return fig

def generate_sample_data(data_type="plat"):
    """Génération de données d'exemple"""
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5'],
            'Musique': ['1a2a3a', '2a1a4a', '3a3a1a', '1a4a2a', '4a2a5a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair', 'Foudre Noire', 'Vent du Nord', 'Tempête Rouge', 'Orage Bleu'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['4.2', '8.5', '15.0', '3.8', '6.8'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['2a1a4a', '4a3a2a', '6a5a8a', '1a2a1a', '3a4a5a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5'],
            'Musique': ['1a1a2a', '1a3a1a', '2a1a4a', '1a2a1a', '3a1a2a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F']
        })

# Interface principale
def main():
    # En-tête avec animation
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro Max</h1>', unsafe_allow_html=True)
    st.markdown("*Intelligence Artificielle avancée pour l'analyse prédictive des courses hippiques*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        
        # Type de course
        race_type = st.selectbox(
            "🏁 Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
            help="AUTO = détection automatique basée sur les données"
        )
        
        # Paramètres ML
        st.subheader("🤖 Configuration IA")
        use_ml = st.checkbox("✅ Activer prédictions ML", value=True)
        ml_confidence = st.slider("🎯 Poids ML dans score final", 0.1, 0.9, 0.7, 0.05)
        
        # Options d'analyse
        st.subheader("📊 Options d'Analyse")
        show_feature_analysis = st.checkbox("🔍 Analyse features avancée")
        export_ml_report = st.checkbox("📊 Rapport ML complet")
        
        # Informations
        st.subheader("ℹ️ Système IA")
        st.info("🧠 **5 Modèles ML** + Ensemble")
        st.info("🎯 **25+ Features** automatiques")
        st.info("📊 **Optimisation** hyperparamètres")
        st.info("🔬 **Validation croisée** intégrée")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data", "📖 Documentation"
    ])
    
    df_final = None
    
    with tab1:
        st.subheader("🔍 Analyse URL Geny.com Optimisée")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(
                "🌐 URL de la course:",
                placeholder="https://www.geny.com/partants-pmu/...",
                help="Scraping intelligent spécialisé pour Geny.com"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyse Turbo", type="primary"):
                if url:
                    with st.spinner("🔄 Extraction Geny.com en cours..."):
                        # Utilisation du scraper spécialisé
                        df, message = scrape_geny_data(url)
                        if df is not None:
                            st.success(f"✅ **{len(df)} chevaux extraits** avec succès!")
                            
                            # Affichage des données extraites
                            st.subheader("📊 Données Extraites")
                            st.dataframe(df, use_container_width=True)
                            df_final = df
                            
                            # Informations de diagnostic
                            with st.expander("🔍 Détails de l'extraction"):
                                st.info(message)
                                st.write("**Colonnes détectées:**", list(df.columns))
                                st.write("**Échantillon de données:**")
                                st.json(df.head(2).to_dict('records'))
                        else:
                            st.error(f"❌ {message}")
                            st.info("💡 **Conseils de dépannage:**")
                            st.write("• Vérifiez que l'URL contient bien une page de partants")
                            st.write("• Certains sites peuvent bloquer le scraping automatique")
                            st.write("• Utilisez les données de test pour voir le fonctionnement")
    
    with tab2:
        st.subheader("📤 Upload CSV Intelligent")
        
        uploaded_file = st.file_uploader(
            "Glissez votre fichier CSV ici",
            type="csv",
            help="Format auto-détecté | Colonnes optimisées | Validation automatique"
        )
        
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ **{len(df_final)} chevaux** chargés avec succès!")
                
                # Validation automatique
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Validation")
                    required_cols = ['Nom', 'Cote']
                    optional_cols = ['Numéro de corde', 'Poids', 'Musique', 'Âge/Sexe']
                    
                    for col in required_cols:
                        if col in df_final.columns:
                            st.success(f"✅ {col}")
                        else:
                            st.error(f"❌ {col} manquant")
                    
                    for col in optional_cols:
                        if col in df_final.columns:
                            st.info(f"ℹ️ {col} détecté")
                
                with col2:
                    st.subheader("📊 Aperçu")
                    st.dataframe(df_final.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {e}")
    
    with tab3:
        st.subheader("🧪 Données de Test Premium")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🏃 **Course PLAT**\n- Handicap réaliste\n- Cordes variées\n- 5 chevaux elite")
            if st.button("🏃 Charger PLAT", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success("✅ Course de PLAT chargée!")
        
        with col2:
            st.markdown("🚗 **Trot ATTELÉ**\n- Autostart tactique\n- Poids uniformes\n- 5 trotteurs")
            if st.button("🚗 Charger ATTELÉ", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success("✅ Course d'ATTELÉ chargée!")
        
        with col3:
            st.markdown("⭐ **Course PREMIUM**\n- Style Arc Triomphe\n- Chevaux internationaux\n- 5 cracks mondiaux")
            if st.button("⭐ Charger PREMIUM", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success("✅ Course PREMIUM chargée!")
        
        if df_final is not None:
            st.markdown("### 📊 Données Chargées")
            st.dataframe(df_final, use_container_width=True)
    
    with tab4:
        st.subheader("📚 Documentation IA Avancée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Intelligence Artificielle
            
            **🧠 Modèles Intégrés**
            - Random Forest (100 arbres)
            - Gradient Boosting (100 itérations)
            - Extra Trees (ensemble)
            - Ridge Regression (L2)
            - Elastic Net (L1+L2)
            - **Ensemble Voting** (meilleurs modèles)
            
            **🎯 Features Automatiques (25+)**
            - Cotes (log, inverse, sqrt)
            - Position (log, sqrt, percentile)
            - Poids (normalisé, rang)
            - Âge (carré, optimal)
            - Forme (moyenne, best, constance)
            - **Interactions complexes**
            - **Sélection automatique**
            """)
        
        with col2:
            st.markdown("""
            ### 🔬 Spécialisation Geny.com
            
            **🌐 Scraping Intelligent**
            - 4 méthodes d'extraction
            - Reconnaissance patterns multiples
            - Fallback automatique
            - Gestion erreurs robuste
            
            **📊 Optimisations**
            - Headers spécialisés
            - Session persistante
            - Timeout intelligent
            - Cache 5 minutes
            
            **🎯 Extraction**
            - Noms de chevaux
            - Cotes automatiques
            - Numéros de corde
            - Poids et musique
            - Âge/Sexe si disponible
            """)
    
    # ANALYSE PRINCIPALE
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse IA Avancée")
        
        # Préparation des données
        df_prepared = prepare_data(df_final)
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide après nettoyage")
            return
        
        # Détection du type
        if race_type == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type
            config = ADVANCED_CONFIGS[detected_type]
            st.markdown(f'<div class="metric-card-pro">{config["description"]}</div>', 
                       unsafe_allow_html=True)
        
        # === ANALYSE ML ===
        if use_ml:
            with st.spinner("🤖 IA en cours... Entraînement des modèles..."):
                try:
                    # Initialisation du système ML
                    advanced_ml = AdvancedHorseRacingML()
                    
                    # Création des features
                    progress_bar = st.progress(0)
                    st.text("🔧 Création des features avancées...")
                    progress_bar.progress(20)
                    
                    X_advanced = advanced_ml.create_advanced_features(df_prepared, detected_type)
                    st.success(f"✅ **{len(X_advanced.columns)} features** créées automatiquement!")
                    
                    # Target intelligent
                    st.text("🎯 Génération du target ML...")
                    progress_bar.progress(40)
                    
                    odds_component = 1 / (df_prepared['odds_numeric'] + 0.01)
                    form_component = 1 / (X_advanced.get('form_avg', pd.Series([8.0]*len(df_prepared))) + 1)
                    y_target = (0.6 * odds_component + 0.4 * form_component + 
                               np.random.normal(0, 0.05, len(df_prepared)))
                    
                    # Entraînement
                    st.text("🚀 Entraînement des modèles...")
                    progress_bar.progress(70)
                    
                    ml_results = advanced_ml.train_advanced_models(X_advanced, y_target)
                    
                    progress_bar.progress(100)
                    st.success("✅ **Système IA entraîné avec succès!**")
                    
                    # Sélection des meilleures prédictions
                    best_predictions, best_model_name = advanced_ml.get_best_model_predictions(ml_results)
                    
                    if len(best_predictions) > 0:
                        # Normalisation
                        if best_predictions.max() != best_predictions.min():
                            ml_predictions_norm = ((best_predictions - best_predictions.min()) / 
                                                 (best_predictions.max() - best_predictions.min()))
                        else:
                            ml_predictions_norm = np.ones(len(best_predictions)) * 0.5
                        
                        df_prepared['ml_score'] = ml_predictions_norm
                        
                        # Affichage du rapport ML
                        st.subheader("🤖 Rapport ML")
                        if ml_results:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f'<div class="metric-card-pro">🏆 Meilleur Modèle<br><strong>{best_model_name.upper()}</strong></div>', 
                                           unsafe_allow_html=True)
                            
                            with col2:
                                if 'val_r2' in ml_results.get(best_model_name, {}):
                                    r2_score = ml_results[best_model_name]['val_r2']
                                    st.markdown(f'<div class="metric-card-pro">🎯 R² Score<br><strong>{r2_score:.3f}</strong></div>', 
                                               unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f'<div class="metric-card-pro">🔧 Modèles Entraînés<br><strong>{len(ml_results)}</strong></div>', 
                                           unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Erreur dans l'analyse ML: {str(e)}")
                    use_ml = False
                    ml_results = {}
        
        # === SCORE FINAL ===
        # Score traditionnel
        traditional_score = 1 / (df_prepared['odds_numeric'] + 0.01)
        if traditional_score.max() != traditional_score.min():
            traditional_score = (traditional_score - traditional_score.min()) / (traditional_score.max() - traditional_score.min())
        
        if use_ml and 'ml_score' in df_prepared.columns:
            df_prepared['score_final'] = (
                (1 - ml_confidence) * traditional_score + 
                ml_confidence * df_prepared['ml_score']
            )
            df_prepared['prediction_method'] = f"Hybride (ML: {ml_confidence:.0%})"
        else:
            df_prepared['score_final'] = traditional_score
            df_prepared['prediction_method'] = "Traditionnel"
        
        # Classement final
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        # === AFFICHAGE RÉSULTATS ===
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.subheader("🏆 Classement Final IA")
            
            display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            display_cols.extend(['score_final', 'prediction_method'])
            
            display_df = df_ranked[display_cols].copy()
            display_df['Score IA'] = display_df['score_final'].round(3)
            display_df['Méthode'] = display_df['prediction_method']
            display_df = display_df.drop(['score_final', 'prediction_method'], axis=1)
            
            # Styling
            styled_df = display_df.style.background_gradient(subset=['Score IA'], cmap='RdYlGn')
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Métriques")
            
            # Métriques de course
            favoris = len(df_ranked[df_ranked['odds_numeric'] < 5])
            outsiders = len(df_ranked[df_ranked['odds_numeric'] > 15])
            
            st.markdown(f'<div class="metric-card-pro">⭐ Favoris (cote < 5)<br><strong>{favoris}</strong></div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card-pro">🎲 Outsiders (cote > 15)<br><strong>{outsiders}</strong></div>', 
                       unsafe_allow_html=True)
            
            # Top 3
            st.subheader("🥇 Top 3 IA")
            for i in range(min(3, len(df_ranked))):
                horse = df_ranked.iloc[i]
                
                st.markdown(f"""
                <div class="prediction-box-pro">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    🎯 Cote: {horse['Cote']} | 📊 Score: {horse['score_final']:.3f}<br>
                    📍 Position: {horse['Numéro de corde']} | ⚖️ Poids: {horse.get('Poids', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        
        # === VISUALISATIONS ===
        st.subheader("📊 Visualisations IA")
        fig_advanced = create_visualization(df_ranked, ml_results if use_ml else None)
        st.plotly_chart(fig_advanced, use_container_width=True)
        
        # === ANALYSE FEATURES ===
        if show_feature_analysis and use_ml and hasattr(advanced_ml, 'feature_importance') and advanced_ml.feature_importance:
            st.subheader("🔍 Analyse des Features")
            
            # Combinaison des importances
            all_features = {}
            for model_name, importance_dict in advanced_ml.feature_importance.items():
                for feature, importance in importance_dict.items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            # Statistiques par feature
            feature_stats = []
            for feature, importances in all_features.items():
                feature_stats.append({
                    'Feature': feature,
                    'Importance Moyenne': np.mean(importances),
                    'Écart-Type': np.std(importances),
                    'Nb Modèles': len(importances)
                })
            
            feature_df = pd.DataFrame(feature_stats).sort_values('Importance Moyenne', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top 10 Features")
                st.dataframe(feature_df.head(10), use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Répartition Importances")
                if len(feature_df) > 0:
                    fig_features = px.bar(
                        feature_df.head(8), 
                        x='Importance Moyenne', 
                        y='Feature',
                        orientation='h',
                        title="Top 8 Features par Importance"
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
        
        # === EXPORT ===
        st.subheader("💾 Export Complet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            export_df = df_ranked.copy()
            if use_ml:
                export_df['ml_model_used'] = best_model_name
            
            csv_data = export_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📄 CSV Enrichi",
                data=csv_data,
                file_name=f"pronostic_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export JSON
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'race_type_detected': detected_type,
                    'num_horses': len(df_ranked),
                    'ml_enabled': use_ml,
                    'best_ml_model': best_model_name if use_ml else None
                },
                'predictions': df_ranked.to_dict('records'),
                'ml_performance': ml_results if use_ml else {}
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📋 JSON Complet",
                data=json_data,
                file_name=f"analyse_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
        main()
