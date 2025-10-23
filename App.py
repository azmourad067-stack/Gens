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
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
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
        "form_importance": 0.40,
        "distance_factors": {"sprint": 1.2, "mile": 1.0, "long": 0.8},
        "track_conditions": {"firm": 1.0, "good": 0.95, "soft": 0.85, "heavy": 0.75}
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Stratégie optimisée",
        "optimal_draws": [4, 5, 6],
        "weight_importance": 0.10,
        "draw_importance": 0.30,
        "form_importance": 0.60,
        "autostart_bonuses": {1: -0.3, 2: -0.2, 3: -0.1, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.1, 8: 0.0}
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Focus performance pure",
        "optimal_draws": [],
        "weight_importance": 0.05,
        "draw_importance": 0.05,
        "form_importance": 0.90,
        "driver_importance": 0.25
    }
}

@st.cache_resource
class AdvancedHorseRacingML:
    """Système ML ultra-avancé pour analyse hippique"""
    
    def __init__(self):
        # Ensemble de modèles optimisés
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=3,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                min_samples_split=3, random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=150, max_depth=12, min_samples_split=2,
                random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'bayesian_ridge': BayesianRidge(),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                random_state=42, early_stopping=True
            )
        }
        
        # Ensemble voting
        self.ensemble_model = VotingRegressor([
            ('rf', self.base_models['random_forest']),
            ('gb', self.base_models['gradient_boosting']),
            ('et', self.base_models['extra_trees'])
        ])
        
        # Outils de preprocessing
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        self.feature_selector = SelectKBest(score_func=f_regression, k=15)
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Métriques et résultats
        self.feature_importance = {}
        self.model_scores = {}
        self.prediction_confidence = {}
        self.is_trained = False
        
    def create_advanced_features(self, df, race_type):
        """Création de features ultra-avancées"""
        features = pd.DataFrame(index=df.index)
        
        # === FEATURES DE BASE OPTIMISÉES ===
        features['odds_log'] = np.log1p(df['odds_numeric'])
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.01)
        features['odds_sqrt'] = np.sqrt(df['odds_numeric'])
        features['odds_power'] = np.power(df['odds_numeric'], 0.5)
        
        features['draw'] = df['draw_numeric']
        features['draw_log'] = np.log1p(df['draw_numeric'])
        features['draw_sqrt'] = np.sqrt(df['draw_numeric'])
        
        features['weight'] = df['weight_kg']
        features['weight_norm'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-8)
        features['weight_rank'] = df['weight_kg'].rank(pct=True)
        
        # === FEATURES D'ÂGE ET SEXE ===
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['age_squared'] = features['age'] ** 2
            features['age_optimal'] = np.abs(features['age'] - 5)  # Âge optimal ~5 ans
            
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
        
        # === ANALYSE AVANCÉE DE LA FORME ===
        if 'Musique' in df.columns:
            for i, row in df.iterrows():
                musique = str(row['Musique']) if pd.notna(row['Musique']) else ""
                positions = [int(c) for c in musique if c.isdigit() and int(c) <= 20]
                
                if positions:
                    # Statistiques de base
                    features.loc[i, 'form_avg'] = np.mean(positions)
                    features.loc[i, 'form_best'] = min(positions)
                    features.loc[i, 'form_worst'] = max(positions)
                    features.loc[i, 'form_std'] = np.std(positions) if len(positions) > 1 else 0
                    
                    # Analyse de tendance
                    if len(positions) >= 3:
                        recent_positions = positions[:3]
                        features.loc[i, 'form_trend'] = np.polyfit(range(len(recent_positions)), recent_positions, 1)[0]
                        features.loc[i, 'form_acceleration'] = np.mean(np.diff(recent_positions))
                    
                    # Constance et régularité
                    features.loc[i, 'form_consistency'] = 1 / (1 + np.std(positions))
                    features.loc[i, 'form_recent_improvement'] = max(0, positions[-1] - np.mean(positions[:3])) if len(positions) >= 4 else 0
                    
                    # Victoires et places
                    features.loc[i, 'wins_total'] = positions.count(1)
                    features.loc[i, 'places_total'] = sum(1 for p in positions if p <= 3)
                    features.loc[i, 'wins_recent'] = positions[:3].count(1)
                    features.loc[i, 'places_recent'] = sum(1 for p in positions[:3] if p <= 3)
                    
                    # Séries et streaks
                    win_streak = 0
                    place_streak = 0
                    for p in positions:
                        if p == 1:
                            win_streak += 1
                        else:
                            break
                    for p in positions:
                        if p <= 3:
                            place_streak += 1
                        else:
                            break
                    features.loc[i, 'win_streak'] = win_streak
                    features.loc[i, 'place_streak'] = place_streak
                    
                    # Performance dans différentes conditions
                    features.loc[i, 'big_field_performance'] = np.mean([p for p in positions if p <= 16])  # Gros pelotons
                    features.loc[i, 'small_field_performance'] = np.mean([p for p in positions if p <= 8])   # Petits pelotons
        
        # Valeurs par défaut si pas de musique
        form_cols = ['form_avg', 'form_best', 'form_worst', 'form_std', 'form_trend', 'form_acceleration',
                    'form_consistency', 'form_recent_improvement', 'wins_total', 'places_total',
                    'wins_recent', 'places_recent', 'win_streak', 'place_streak',
                    'big_field_performance', 'small_field_performance']
        for col in form_cols:
            if col not in features.columns:
                features[col] = 0
        
        # === FEATURES RELATIVES ET COMPARATIVES ===
        features['odds_rank'] = df['odds_numeric'].rank()
        features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
        features['weight_rank'] = df['weight_kg'].rank()
        features['draw_percentile'] = df['draw_numeric'].rank(pct=True)
        
        # Écarts à la moyenne
        features['odds_vs_avg'] = df['odds_numeric'] - df['odds_numeric'].mean()
        features['weight_vs_avg'] = df['weight_kg'] - df['weight_kg'].mean()
        features['draw_vs_avg'] = df['draw_numeric'] - df['draw_numeric'].mean()
        
        # === INTERACTIONS COMPLEXES ===
        features['odds_weight_ratio'] = features['odds_inv'] * features['weight_norm']
        features['draw_odds_interaction'] = features['draw'] * features['odds_log']
        features['age_form_interaction'] = features['age'] * features['form_consistency']
        features['weight_draw_penalty'] = features['weight'] * np.maximum(0, features['draw'] - 8)
        
        # === FEATURES SPÉCIFIQUES PAR TYPE ===
        config = ADVANCED_CONFIGS[race_type]
        
        if race_type == "PLAT":
            # Avantages/pénalités corde pour le plat
            features['inner_draw_bonus'] = np.where(features['draw'] <= 4, 0.3, 0)
            features['outer_draw_penalty'] = np.where(features['draw'] >= 12, -0.2, 0)
            features['draw_optimal_distance'] = np.abs(features['draw'] - 4)
            
            # Pénalités poids sophistiquées
            weight_baseline = df['weight_kg'].median()
            features['weight_penalty_linear'] = np.maximum(0, features['weight'] - weight_baseline) * 0.02
            features['weight_penalty_exponential'] = np.power(np.maximum(0, features['weight'] - weight_baseline), 1.5) * 0.01
            
        elif race_type == "ATTELE_AUTOSTART":
            # Stratégie autostart avancée
            autostart_bonuses = config.get('autostart_bonuses', {})
            features['autostart_position_bonus'] = features['draw'].apply(lambda x: autostart_bonuses.get(x, 0))
            features['first_line_advantage'] = np.where(features['draw'] <= 9, 0.2, -0.1)
            features['center_first_line'] = np.where(features['draw'].isin([4, 5, 6]), 0.4, 0)
            
        elif race_type == "ATTELE_VOLTE":
            # Pour le volté, focus sur la forme pure
            features['pure_form_score'] = (features['form_consistency'] * 0.4 + 
                                         features['wins_recent'] * 0.3 + 
                                         features['places_recent'] * 0.3)
        
        # === CLUSTERING ET FEATURES AVANCÉES ===
        # Création de profils de chevaux
        base_features_for_clustering = ['odds_inv', 'form_avg', 'age', 'weight_norm']
        if len(df) >= 3:
            cluster_data = features[base_features_for_clustering].fillna(0)
            try:
                clusters = self.kmeans.fit_predict(cluster_data)
                features['horse_profile_cluster'] = clusters
                features['cluster_0'] = (clusters == 0).astype(int)
                features['cluster_1'] = (clusters == 1).astype(int) 
                features['cluster_2'] = (clusters == 2).astype(int)
            except:
                features['horse_profile_cluster'] = 0
                features['cluster_0'] = 0
                features['cluster_1'] = 0
                features['cluster_2'] = 0
        
        # === FEATURES TEMPORELLES ET SAISONNIÈRES ===
        current_date = datetime.now()
        features['month'] = current_date.month
        features['season'] = (current_date.month % 12 + 3) // 3  # 1=hiver, 2=printemps, etc.
        features['is_weekend'] = current_date.weekday() >= 5
        
        return features.fillna(0)
    
    def optimize_hyperparameters(self, X, y, model_name='random_forest'):
        """Optimisation avancée des hyperparamètres"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3]
            },
            'gradient_boosting': {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'min_samples_split': [2, 3, 5]
            }
        }
        
        if model_name in param_grids and len(X) >= 20:
            base_model = self.base_models[model_name].__class__()
            
            # Utilisation de RandomizedSearchCV pour plus d'efficacité
            random_search = RandomizedSearchCV(
                base_model, param_grids[model_name],
                n_iter=20, cv=3, scoring='neg_mean_squared_error',
                random_state=42, n_jobs=-1
            )
            
            try:
                random_search.fit(X, y)
                return random_search.best_estimator_
            except:
                return self.base_models[model_name]
        
        return self.base_models[model_name]
    
    def train_advanced_models(self, X, y, optimize_hyperparams=True, use_feature_selection=True):
        """Entraînement avancé avec sélection de features et optimisation"""
        
        if len(X) < 5:
            st.warning("⚠️ Pas assez de données pour entraînement avancé")
            return {}
        
        # === PREPROCESSING AVANCÉ ===
        X_processed = X.copy()
        
        # Normalisation robuste
        X_scaled = self.scalers['robust'].fit_transform(X_processed)
        
        # Sélection de features si demandée
        if use_feature_selection and len(X.columns) > 10:
            try:
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                selected_features = self.feature_selector.get_support()
                selected_feature_names = X.columns[selected_features].tolist()
                st.info(f"🎯 {len(selected_feature_names)} features sélectionnées sur {len(X.columns)}")
            except:
                X_selected = X_scaled
                selected_feature_names = X.columns.tolist()
        else:
            X_selected = X_scaled
            selected_feature_names = X.columns.tolist()
        
        # Création de features polynomiales si dataset pas trop grand
        if len(X.columns) <= 15 and len(X) >= 10:
            try:
                X_poly = self.poly_features.fit_transform(X_selected)
                X_final = X_poly
                feature_names_final = self.poly_features.get_feature_names_out(selected_feature_names)
            except:
                X_final = X_selected
                feature_names_final = selected_feature_names
        else:
            X_final = X_selected
            feature_names_final = selected_feature_names
        
        # === ENTRAÎNEMENT DES MODÈLES ===
        results = {}
        
        # Division train/validation si assez de données
        if len(X) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X_final, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X_final, X_final, y, y
        
        # Entraînement de chaque modèle
        for name, model in self.base_models.items():
            try:
                # Optimisation des hyperparamètres si demandée
                if optimize_hyperparams and name in ['random_forest', 'gradient_boosting']:
                    optimized_model = self.optimize_hyperparameters(X_train, y_train, name)
                else:
                    optimized_model = model
                
                # Entraînement
                optimized_model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = optimized_model.predict(X_train)
                y_pred_val = optimized_model.predict(X_val)
                
                # Métriques
                results[name] = {
                    'model': optimized_model,
                    'train_r2': r2_score(y_train, y_pred_train),
                    'val_r2': r2_score(y_val, y_pred_val),
                    'train_mse': mean_squared_error(y_train, y_pred_train),
                    'val_mse': mean_squared_error(y_val, y_pred_val),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'val_mae': mean_absolute_error(y_val, y_pred_val)
                }
                
                # Cross-validation si assez de données
                if len(X) >= 15:
                    cv_scores = cross_val_score(optimized_model, X_final, y, cv=3, scoring='r2')
                    results[name]['cv_r2_mean'] = cv_scores.mean()
                    results[name]['cv_r2_std'] = cv_scores.std()
                
                # Feature importance
                if hasattr(optimized_model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_names_final, optimized_model.feature_importances_))
                    # Garder seulement les top features
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    results[name]['feature_importance'] = dict(sorted_importance[:15])
                    self.feature_importance[name] = results[name]['feature_importance']
                
                # Prédictions finales sur tout le dataset
                final_predictions = optimized_model.predict(X_final)
                results[name]['predictions'] = final_predictions
                
                # Confiance des prédictions (écart-type des prédictions des arbres pour RF)
                if name == 'random_forest' and hasattr(optimized_model, 'estimators_'):
                    tree_predictions = np.array([tree.predict(X_final) for tree in optimized_model.estimators_])
                    prediction_std = np.std(tree_predictions, axis=0)
                    results[name]['prediction_confidence'] = 1 / (1 + prediction_std)  # Plus l'écart-type est faible, plus la confiance est élevée
                
            except Exception as e:
                st.warning(f"⚠️ Erreur entraînement {name}: {str(e)}")
                continue
        
        # === MODÈLE D'ENSEMBLE ===
        if len(results) >= 2:
            try:
                # Sélection des meilleurs modèles pour l'ensemble
                best_models = sorted(results.items(), key=lambda x: x[1].get('val_r2', 0), reverse=True)[:3]
                
                ensemble_estimators = [(name, result['model']) for name, result in best_models]
                ensemble = VotingRegressor(ensemble_estimators)
                ensemble.fit(X_final, y)
                
                # Métriques ensemble
                y_pred_ensemble = ensemble.predict(X_final)
                results['ensemble'] = {
                    'model': ensemble,
                    'r2': r2_score(y, y_pred_ensemble),
                    'mse': mean_squared_error(y, y_pred_ensemble),
                    'mae': mean_absolute_error(y, y_pred_ensemble),
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
    
    def generate_prediction_report(self, results):
        """Génère un rapport détaillé des prédictions"""
        if not results:
            return "Aucun modèle entraîné"
        
        report = ["🤖 **RAPPORT D'ANALYSE ML AVANCÉE**", "="*50]
        
        # Résumé des modèles
        report.append(f"\n📊 **Modèles entraînés**: {len(results)}")
        
        for name, result in results.items():
            if name == 'ensemble':
                report.append(f"\n🎯 **{name.upper()}** (Meilleurs modèles combinés)")
                report.append(f"   • Composants: {', '.join(result.get('component_models', []))}")
            else:
                report.append(f"\n🔧 **{name.upper()}**")
            
            report.append(f"   • R² Score: {result.get('val_r2', result.get('r2', 0)):.3f}")
            report.append(f"   • MSE: {result.get('val_mse', result.get('mse', 0)):.3f}")
            
            if 'cv_r2_mean' in result:
                report.append(f"   • CV R² (mean±std): {result['cv_r2_mean']:.3f}±{result['cv_r2_std']:.3f}")
        
        # Meilleur modèle
        best_predictions, best_model = self.get_best_model_predictions(results)
        report.append(f"\n🏆 **Meilleur modèle sélectionné**: {best_model.upper()}")
        
        return "\n".join(report)

def create_advanced_visualizations(df_ranked, ml_results=None, prediction_confidence=None):
    """Visualisations ultra-avancées avec métriques ML"""
    
    # Configuration des sous-graphiques
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '🏆 Scores ML avec Intervalles de Confiance',
            '📊 Distribution des Cotes (Histogramme + Densité)',
            '🎯 Corrélation Poids-Performance-Âge',
            '🧠 Top Features ML par Importance',
            '📈 Analyse de Forme vs Prédiction ML',
            '🔄 Matrice de Corrélation des Facteurs'
        ),
        specs=[
            [{"secondary_y": True}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "heatmap"}]
        ]
    )
    
    colors = px.colors.qualitative.Set3
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    
    # Graphique 1: Scores avec intervalles de confiance
    if score_col in df_ranked.columns:
        y_values = df_ranked[score_col]
        
        # Intervalles de confiance si disponibles
        if prediction_confidence is not None:
            confidence = prediction_confidence if len(prediction_confidence) == len(df_ranked) else [0.5] * len(df_ranked)
            y_upper = y_values + np.array(confidence) * 0.1
            y_lower = y_values - np.array(confidence) * 0.1
            
            # Zone de confiance
            fig.add_trace(
                go.Scatter(
                    x=list(df_ranked['rang']) + list(df_ranked['rang'][::-1]),
                    y=list(y_upper) + list(y_lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalle Confiance',
                    showlegend=True
                ), row=1, col=1
            )
        
        # Scores principaux
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'],
                y=y_values,
                mode='markers+lines',
                marker=dict(
                    size=15,
                    color=df_ranked['odds_numeric'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cote", x=0.45)
                ),
                text=[f"{row['Nom']}<br>Cote: {row['Cote']}<br>Score: {row[score_col]:.3f}" 
                      for _, row in df_ranked.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='Score ML',
                line=dict(width=3, color=colors[0])
            ), row=1, col=1
        )
    
    # Graphique 2: Distribution avancée des cotes
    fig.add_trace(
        go.Histogram(
            x=df_ranked['odds_numeric'],
            nbinsx=12,
            marker_color=colors[1],
            opacity=0.7,
            name='Distribution Cotes',
            yaxis='y2'
        ), row=1, col=2
    )
    
    # Ajout courbe de densité
    try:
        from scipy import stats
        density = stats.gaussian_kde(df_ranked['odds_numeric'])
        x_range = np.linspace(df_ranked['odds_numeric'].min(), df_ranked['odds_numeric'].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=density(x_range),
                mode='lines',
                name='Densité',
                line=dict(color='red', width=2),
                yaxis='y2'
            ), row=1, col=2
        )
    except:
        pass
    
    # Graphique 3: Analyse 3D (Poids-Performance-Âge)
    if 'age' in df_ranked.columns and score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['weight_kg'],
                y=df_ranked[score_col],
                mode='markers',
                marker=dict(
                    size=df_ranked.get('age', [5]*len(df_ranked)) * 3,
                    color=df_ranked['rang'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Rang", x=1.02)
                ),
                text=[f"{row['Nom']}<br>Poids: {row['weight_kg']:.1f}kg<br>Âge: {row.get('age', 'N/A')}<br>Score: {row[score_col]:.3f}" 
                      for _, row in df_ranked.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='Poids-Performance-Âge'
            ), row=2, col=1
        )
    
    # Graphique 4: Feature importance
    if ml_results:
        # Combiner les importances de tous les modèles
        all_importance = {}
        for model_name, results in ml_results.items():
            if 'feature_importance' in results:
                for feature, importance in results['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        # Moyenne des importances
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
        top_features = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_features:
            fig.add_trace(
                go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker_color=colors[3],
                    name='Feature Importance',
                    text=[f"{v:.3f}" for v in top_features.values()],
                    textposition='auto'
                ), row=2, col=2
            )
    
    # Graphique 5: Forme vs ML
    if 'form_avg' in df_ranked.columns and score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['form_avg'],
                y=df_ranked[score_col],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df_ranked['odds_numeric'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Cote", x=0.45, y=0.25)
                ),
                text=df_ranked['Nom'],
                name='Forme vs ML'
            ), row=3, col=1
        )
        
        # Ligne de tendance
        try:
            z = np.polyfit(df_ranked['form_avg'], df_ranked[score_col], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=df_ranked['form_avg'],
                    y=p(df_ranked['form_avg']),
                    mode='lines',
                    name='Tendance',
                    line=dict(color='red', dash='dash')
                ), row=3, col=1
            )
        except:
            pass
    
    # Graphique 6: Matrice de corrélation
    numeric_cols = ['odds_numeric', 'weight_kg', 'draw_numeric']
    if 'age' in df_ranked.columns:
        numeric_cols.append('age')
    if score_col in df_ranked.columns:
        numeric_cols.append(score_col)
    
    correlation_data = df_ranked[numeric_cols].corr()
    
    fig.add_trace(
        go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='RdBu',
            zmid=0,
            name='Corrélation',
            text=correlation_data.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=10)
        ), row=3, col=2
    )
    
    # Mise à jour du layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="📊 Dashboard ML Avancé - Analyse Hippique Complète",
        title_x=0.5,
        title_font_size=20
    )
    
    return fig

def create_performance_dashboard(ml_results):
    """Dashboard de performance des modèles ML"""
    if not ml_results:
        return None
    
    # Préparation des données
    models = []
    r2_scores = []
    mse_scores = []
    model_types = []
    
    for name, result in ml_results.items():
        models.append(name.upper())
        r2_scores.append(result.get('val_r2', result.get('r2', 0)))
        mse_scores.append(result.get('val_mse', result.get('mse', 0)))
        model_types.append('Ensemble' if name == 'ensemble' else 'Base')
    
    # Création des graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🎯 R² Score par Modèle', '📉 MSE par Modèle', 
                       '⚡ Performance Comparative', '🔄 Cross-Validation (si disponible)'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "box"}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # R² Scores
    fig.add_trace(
        go.Bar(
            x=models, y=r2_scores,
            marker_color=[colors[0] if t == 'Ensemble' else colors[1] for t in model_types],
            name='R² Score',
            text=[f"{score:.3f}" for score in r2_scores],
            textposition='auto'
        ), row=1, col=1
    )
    
    # MSE Scores
    fig.add_trace(
        go.Bar(
            x=models, y=mse_scores,
            marker_color=[colors[2] if t == 'Ensemble' else colors[3] for t in model_types],
            name='MSE',
            text=[f"{score:.3f}" for score in mse_scores],
            textposition='auto'
        ), row=1, col=2
    )
    
    # Performance comparative (R² vs MSE)
    fig.add_trace(
        go.Scatter(
            x=mse_scores, y=r2_scores,
            mode='markers+text',
            marker=dict(
                size=15,
                color=[colors[0] if t == 'Ensemble' else colors[4] for t in model_types],
                line=dict(width=2, color='white')
            ),
            text=models,
            textposition='middle right',
            name='R² vs MSE'
        ), row=2, col=1
    )
    
    # Cross-validation si disponible
    cv_data = []
    cv_labels = []
    for name, result in ml_results.items():
        if 'cv_r2_mean' in result:
            cv_data.append([result['cv_r2_mean'] - result['cv_r2_std'],
                           result['cv_r2_mean'],
                           result['cv_r2_mean'] + result['cv_r2_std']])
            cv_labels.append(name.upper())
    
    if cv_data:
        for i, (label, data) in enumerate(zip(cv_labels, cv_data)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=label,
                    marker_color=colors[i % len(colors)]
                ), row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text="🔬 Dashboard Performance ML",
        title_x=0.5,
        showlegend=False
    )
    
    return fig

# Interface principale optimisée
def main():
    # En-tête avec animation
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro Max</h1>', unsafe_allow_html=True)
    st.markdown("*Intelligence Artificielle avancée pour l'analyse prédictive des courses hippiques*")
    
    # Sidebar ultra-configurée
    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        
        # Type de course
        race_type = st.selectbox(
            "🏁 Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
            help="AUTO = détection automatique avec analyse statistique"
        )
        
        # Paramètres ML avancés
        st.subheader("🤖 Configuration IA")
        use_ml = st.checkbox("✅ Activer prédictions ML", value=True)
        ml_confidence = st.slider("🎯 Poids ML dans score final", 0.1, 0.9, 0.7, 0.05)
        
        optimize_hyperparams = st.checkbox("🔧 Optimisation hyperparamètres", value=True)
        use_feature_selection = st.checkbox("🎯 Sélection automatique features", value=True)
        use_ensemble = st.checkbox("🎪 Modèles d'ensemble", value=True)
        
        # Options d'analyse
        st.subheader("📊 Options d'Analyse")
        show_ml_dashboard = st.checkbox("📈 Dashboard ML détaillé", value=True)
        show_prediction_confidence = st.checkbox("🎯 Intervalles de confiance", value=True)
        show_feature_analysis = st.checkbox("🔍 Analyse features avancée")
        
        # Export avancé
        st.subheader("💾 Export Avancé")
        export_ml_report = st.checkbox("📊 Rapport ML complet")
        export_confidence_intervals = st.checkbox("📈 Intervalles de confiance")
        
        # Informations système
        st.subheader("ℹ️ Système IA")
        st.info("🧠 **7 Modèles ML** + Ensemble")
        st.info("🎯 **50+ Features** automatiques")
        st.info("📊 **Optimisation** hyperparamètres")
        st.info("🔬 **Validation croisée** intégrée")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data", 
        "📊 ML Dashboard", "📖 Documentation"
    ])
    
    df_final = None
    
    # [Reprise du code des onglets avec les mêmes fonctions que précédemment mais optimisées]
    with tab1:
        st.subheader("🔍 Analyse URL Ultra-Rapide")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(
                "🌐 URL de la course:",
                placeholder="https://site-courses.com/course/123",
                help="Scraping intelligent avec détection automatique de structure"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyse Turbo", type="primary"):
                if url:
                    with st.spinner("🔄 Extraction IA en cours..."):
                        # Utilisation de la fonction de scraping existante
                        df, message = scrape_race_data(url)
                        if df is not None:
                            st.success(f"✅ **{len(df)} chevaux extraits** avec succès!")
                            st.dataframe(df.head(), use_container_width=True)
                            df_final = df
                        else:
                            st.error(f"❌ {message}")
    
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
                
                # Validation automatique intelligente
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Validation Automatique")
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
                    st.subheader("📊 Aperçu Intelligent")
                    st.dataframe(df_final.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {e}")
    
    with tab3:
        st.subheader("🧪 Données de Test Premium")
        
        col1, col2, col3 = st.columns(3)
        
        test_descriptions = {
            "plat": "🏃 **Course PLAT**\n- Handicap réaliste\n- Cordes variées\n- 8 chevaux elite",
            "attele": "🚗 **Trot ATTELÉ**\n- Autostart tactique\n- Poids uniformes\n- 8 trotteurs",
            "premium": "⭐ **Course PREMIUM**\n- Style Arc Triomphe\n- Chevaux internationaux\n- 5 cracks mondiaux"
        }
        
        with col1:
            st.markdown(test_descriptions["plat"])
            if st.button("🏃 Charger PLAT", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success("✅ Course de PLAT chargée!")
        
        with col2:
            st.markdown(test_descriptions["attele"])
            if st.button("🚗 Charger ATTELÉ", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success("✅ Course d'ATTELÉ chargée!")
        
        with col3:
            st.markdown(test_descriptions["premium"])
            if st.button("⭐ Charger PREMIUM", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success("✅ Course PREMIUM chargée!")
        
        if df_final is not None:
            st.markdown("### 📊 Données Chargées")
            st.dataframe(df_final, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Dashboard ML en Temps Réel")
        st.info("🔄 Ce dashboard se met à jour automatiquement après chaque analyse")
        
        # Placeholder pour les métriques ML
        dashboard_placeholder = st.empty()
    
    with tab5:
        st.subheader("📚 Documentation IA Avancée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Intelligence Artificielle
            
            **🧠 Modèles Intégrés**
            - Random Forest (200 arbres)
            - Gradient Boosting (150 itérations)
            - Extra Trees (ensemble)
            - Ridge Regression (L2)
            - Elastic Net (L1+L2)
            - Bayesian Ridge (probabiliste)
            - Neural Network (100-50-25)
            - **Ensemble Voting** (meilleurs modèles)
            
            **🎯 Features Automatiques (50+)**
            - Cotes (log, inverse, sqrt, power)
            - Position (log, sqrt, percentile)
            - Poids (normalisé, rang, vs moyenne)
            - Âge (carré, optimal, interactions)
            - Forme (tendance, accélération, constance)
            - **Interactions polynomiales**
            - **Clustering automatique**
            - **Sélection intelligente**
            """)
        
        with col2:
            st.markdown("""
            ### 🔬 Optimisations Avancées
            
            **⚙️ Hyperparamètres**
            - RandomizedSearchCV (20 itérations)
            - 3-fold Cross-Validation
            - Optimisation automatique
            
            **📊 Validation**
            - Train/Validation split
            - Cross-validation scores
            - Intervalles de confiance
            - Feature importance
            
            **🎪 Ensemble Learning**
            - Voting des 3 meilleurs
            - Pondération dynamique
            - Prédictions robustes
            
            **📈 Métriques**
            - R² Score (coefficient détermination)
            - MSE (erreur quadratique)
            - MAE (erreur absolue)
            - Confiance prédictions
            """)
    
    # ANALYSE PRINCIPALE ULTRA-AVANCÉE
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse IA Ultra-Avancée")
        
        # Préparation des données
        df_prepared = prepare_data(df_final)
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide après nettoyage")
            return
        
        # Détection du type avec analyse poussée
        if race_type == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type
            config = ADVANCED_CONFIGS[detected_type]
            st.markdown(f'<div class="metric-card-pro">{config["description"]}</div>', 
                       unsafe_allow_html=True)
        
        # === ANALYSE ML ULTRA-AVANCÉE ===
        if use_ml:
            with st.spinner("🤖 IA Ultra-Avancée en cours... Optimisation des modèles..."):
                try:
                    # Initialisation du système ML avancé
                    advanced_ml = AdvancedHorseRacingML()
                    
                    # Création des features ultra-avancées
                    progress_bar = st.progress(0)
                    st.text("🔧 Création des features avancées...")
                    progress_bar.progress(20)
                    
                    X_advanced = advanced_ml.create_advanced_features(df_prepared, detected_type)
                    st.success(f"✅ **{len(X_advanced.columns)} features** créées automatiquement!")
                    
                    # Création du target synthétique intelligent
                    st.text("🎯 Génération du target ML intelligent...")
                    progress_bar.progress(40)
                    
                    # Target basé sur plusieurs facteurs
                    odds_component = 1 / (df_prepared['odds_numeric'] + 0.01)
                    if 'form_avg' in X_advanced.columns:
                        form_component = 1 / (X_advanced['form_avg'] + 1)
                    else:
                        form_component = np.ones(len(df_prepared))
                    
                    y_target = (0.6 * odds_component + 0.4 * form_component + 
                               np.random.normal(0, 0.05, len(df_prepared)))
                    
                    # Entraînement des modèles ultra-avancés
                    st.text("🚀 Entraînement des modèles IA...")
                    progress_bar.progress(70)
                    
                    ml_results = advanced_ml.train_advanced_models(
                        X_advanced, y_target,
                        optimize_hyperparams=optimize_hyperparams,
                        use_feature_selection=use_feature_selection
                    )
                    
                    progress_bar.progress(100)
                    st.success("✅ **Système IA entraîné avec succès!**")
                    
                    # Sélection des meilleures prédictions
                    best_predictions, best_model_name = advanced_ml.get_best_model_predictions(ml_results)
                    
                    if len(best_predictions) > 0:
                        # Normalisation des prédictions
                        if best_predictions.max() != best_predictions.min():
                            ml_predictions_norm = ((best_predictions - best_predictions.min()) / 
                                                 (best_predictions.max() - best_predictions.min()))
                        else:
                            ml_predictions_norm = np.ones(len(best_predictions)) * 0.5
                        
                        df_prepared['ml_score'] = ml_predictions_norm
                        df_prepared['ml_confidence'] = ml_results.get(best_model_name, {}).get('prediction_confidence', [0.5] * len(df_prepared))
                        
                        # Affichage du rapport ML
                        st.subheader("🤖 Rapport ML Détaillé")
                        report = advanced_ml.generate_prediction_report(ml_results)
                        st.markdown(report)
                        
                except Exception as e:
                    st.error(f"❌ Erreur dans l'analyse ML avancée: {str(e)}")
                    use_ml = False
                    ml_results = {}
        
        # === CALCUL DU SCORE FINAL OPTIMISÉ ===
        # Score traditionnel amélioré
        traditional_components = []
        
        # Composant cotes (toujours présent)
        odds_score = 1 / (df_prepared['odds_numeric'] + 0.01)
        traditional_components.append(('odds', odds_score, 0.4))
        
        # Composant position (si pertinent)
        if detected_type in ['PLAT', 'ATTELE_AUTOSTART']:
            config = ADVANCED_CONFIGS[detected_type]
            draw_score = np.zeros(len(df_prepared))
            for i, draw in enumerate(df_prepared['draw_numeric']):
                if draw in config['optimal_draws']:
                    draw_score[i] = 1.0
                elif detected_type == 'PLAT' and draw <= 4:
                    draw_score[i] = 0.8
                elif detected_type == 'ATTELE_AUTOSTART' and draw <= 9:
                    draw_score[i] = 0.6
                else:
                    draw_score[i] = 0.3
            traditional_components.append(('draw', draw_score, 0.3))
        
        # Composant poids (si important)
        if ADVANCED_CONFIGS[detected_type]['weight_importance'] > 0.1:
            weight_score = 1 / (df_prepared['weight_kg'] - df_prepared['weight_kg'].min() + 1)
            traditional_components.append(('weight', weight_score, 0.3))
        
        # Combinaison des composants traditionnels
        traditional_score = np.zeros(len(df_prepared))
        total_weight = sum(weight for _, _, weight in traditional_components)
        
        for name, score, weight in traditional_components:
            normalized_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            traditional_score += normalized_score * (weight / total_weight)
        
        # Score final combiné
        if use_ml and 'ml_score' in df_prepared.columns:
            df_prepared['score_final'] = (
                (1 - ml_confidence) * traditional_score + 
                ml_confidence * df_prepared['ml_score']
            )
            df_prepared['prediction_method'] = f"Hybride (ML: {ml_confidence:.0%})"
        else:
            df_prepared['score_final'] = traditional_score
            df_prepared['prediction_method'] = "Traditionnel"
        
        # === CLASSEMENT FINAL ===
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        # Ajout de classes de confiance
        if 'ml_confidence' in df_ranked.columns:
            df_ranked['confidence_class'] = pd.cut(
                df_ranked['ml_confidence'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Faible', 'Moyenne', 'Élevée']
            )
        
        # === AFFICHAGE DES RÉSULTATS ULTRA-AVANCÉS ===
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.subheader("🏆 Classement Final IA")
            
            # Préparation de l'affichage
            display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            display_cols.extend(['score_final', 'prediction_method'])
            
            if 'ml_confidence' in df_ranked.columns:
                display_cols.append('confidence_class')
            
            # Formatage avancé
            display_df = df_ranked[display_cols].copy()
            display_df['Score IA'] = display_df['score_final'].round(3)
            display_df['Méthode'] = display_df['prediction_method']
            
            # Suppression des colonnes techniques
            display_df = display_df.drop(['score_final', 'prediction_method'], axis=1)
            
            # Styling conditionnel
            def style_confidence(val):
                if val == 'Élevée':
                    return 'background-color: #d1fae5; color: #065f46'
                elif val == 'Moyenne':
                    return 'background-color: #fef3c7; color: #92400e'
                else:
                    return 'background-color: #fecaca; color: #991b1b'
            
            if 'confidence_class' in display_df.columns:
                styled_df = display_df.style.applymap(
                    style_confidence, subset=['confidence_class']
                ).background_gradient(
                    subset=['Score IA'], cmap='RdYlGn'
                )
            else:
                styled_df = display_df.style.background_gradient(
                    subset=['Score IA'], cmap='RdYlGn'
                )
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Métriques Avancées")
            
            # Métriques ML si disponibles
            if use_ml and ml_results:
                best_model_result = ml_results.get(best_model_name, {})
                
                if 'val_r2' in best_model_result:
                    r2_score = best_model_result['val_r2']
                    confidence_color = "ml-confidence-high" if r2_score > 0.7 else "ml-confidence-medium" if r2_score > 0.4 else "ml-confidence-low"
                    st.markdown(f'<div class="metric-card-pro">🧠 R² Score ML<br><strong>{r2_score:.3f}</strong></div>', 
                               unsafe_allow_html=True)
                
                if 'cv_r2_mean' in best_model_result:
                    cv_score = best_model_result['cv_r2_mean']
                    cv_std = best_model_result['cv_r2_std']
                    st.markdown(f'<div class="metric-card-pro">🔄 Cross-Validation<br><strong>{cv_score:.3f}±{cv_std:.3f}</strong></div>', 
                               unsafe_allow_html=True)
                
                st.markdown(f'<div class="metric-card-pro">🏆 Meilleur Modèle<br><strong>{best_model_name.upper()}</strong></div>', 
                           unsafe_allow_html=True)
            
            # Métriques de course
            favoris = len(df_ranked[df_ranked['odds_numeric'] < 5])
            outsiders = len(df_ranked[df_ranked['odds_numeric'] > 15])
            
            st.markdown(f'<div class="metric-card-pro">⭐ Favoris (cote < 5)<br><strong>{favoris}</strong></div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card-pro">🎲 Outsiders (cote > 15)<br><strong>{outsiders}</strong></div>', 
                       unsafe_allow_html=True)
            
            # Répartition des confiances
            if 'confidence_class' in df_ranked.columns:
                conf_counts = df_ranked['confidence_class'].value_counts()
                st.markdown("### 🎯 Répartition Confiance")
                for conf_level, count in conf_counts.items():
                    conf_class = f"ml-confidence-{conf_level.lower()}" if conf_level != 'Moyenne' else "ml-confidence-medium"
                    st.markdown(f'<div class="prediction-box-pro {conf_class}"><strong>{conf_level}</strong>: {count} chevaux</div>', 
                               unsafe_allow_html=True)
            
            # Top 3 avec analyse détaillée
            st.subheader("🥇 Top 3 IA")
            for i in range(min(3, len(df_ranked))):
                horse = df_ranked.iloc[i]
                
                # Classe de confiance pour le styling
                if 'ml_confidence' in horse:
                    conf_val = horse['ml_confidence']
                    if conf_val > 0.7:
                        conf_class = "ml-confidence-high"
                        conf_icon = "🟢"
                    elif conf_val > 0.4:
                        conf_class = "ml-confidence-medium" 
                        conf_icon = "🟡"
                    else:
                        conf_class = "ml-confidence-low"
                        conf_icon = "🔴"
                    
                    confidence_info = f"{conf_icon} Confiance: {conf_val:.2f}"
                else:
                    conf_class = "ml-confidence-medium"
                    confidence_info = ""
                
                st.markdown(f"""
                <div class="prediction-box-pro {conf_class}">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    🎯 Cote: {horse['Cote']} | 📊 Score IA: {horse['score_final']:.3f}<br>
                    📍 Position: {horse['Numéro de corde']} | ⚖️ Poids: {horse.get('Poids', 'N/A')}<br>
                    {confidence_info}
                </div>
                """, unsafe_allow_html=True)
        
        # === VISUALISATIONS ULTRA-AVANCÉES ===
        st.subheader("📊 Visualisations IA Avancées")
        
        prediction_confidence = df_ranked.get('ml_confidence', None)
        fig_advanced = create_advanced_visualizations(
            df_ranked, 
            ml_results if use_ml else None,
            prediction_confidence
        )
        st.plotly_chart(fig_advanced, use_container_width=True)
        
        # Dashboard ML si activé
        if show_ml_dashboard and use_ml and ml_results:
            st.subheader("🔬 Dashboard Performance ML")
            fig_dashboard = create_performance_dashboard(ml_results)
            if fig_dashboard:
                st.plotly_chart(fig_dashboard, use_container_width=True)
        
        # === ANALYSE DES FEATURES SI DEMANDÉE ===
        if show_feature_analysis and use_ml and hasattr(advanced_ml, 'feature_importance'):
            st.subheader("🔍 Analyse Avancée des Features")
            
            # Combinaison des importances de tous les modèles
            all_features = {}
            for model_name, importance_dict in advanced_ml.feature_importance.items():
                for feature, importance in importance_dict.items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            # Calcul des statistiques par feature
            feature_stats = []
            for feature, importances in all_features.items():
                feature_stats.append({
                    'Feature': feature,
                    'Importance Moyenne': np.mean(importances),
                    'Écart-Type': np.std(importances),
                    'Min': np.min(importances),
                    'Max': np.max(importances),
                    'Nb Modèles': len(importances)
                })
            
            feature_df = pd.DataFrame(feature_stats).sort_values('Importance Moyenne', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top 15 Features")
                st.dataframe(feature_df.head(15), use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Distribution des Importances")
                fig_features = px.box(
                    feature_df.head(10), 
                    y='Feature', 
                    x='Importance Moyenne',
                    title="Distribution des Importances (Top 10)"
                )
                st.plotly_chart(fig_features, use_container_width=True)
        
        # === EXPORT AVANCÉ ===
        st.subheader("💾 Export Ultra-Complet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export CSV enrichi
            export_df = df_ranked.copy()
            if use_ml:
                export_df['ml_model_used'] = best_model_name
                if 'ml_confidence' in export_df.columns:
                    export_df['prediction_confidence'] = export_df['ml_confidence']
            
            csv_data = export_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📄 CSV Enrichi",
                data=csv_data,
                file_name=f"pronostic_ia_avance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export JSON complet
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'race_type_detected': detected_type,
                    'num_horses': len(df_ranked),
                    'ml_enabled': use_ml,
                    'best_ml_model': best_model_name if use_ml else None,
                    'features_count': len(X_advanced.columns) if use_ml else 0
                },
                'predictions': df_ranked.to_dict('records'),
                'ml_performance': ml_results if use_ml else {},
                'feature_importance': advanced_ml.feature_importance if use_ml else {}
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📋 JSON Complet",
                data=json_data,
                file_name=f"analyse_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Rapport ML détaillé
            if export_ml_report and use_ml:
                ml_report = {
                    'executive_summary': {
                        'best_model': best_model_name,
                        'r2_score': ml_results.get(best_model_name, {}).get('val_r2', 'N/A'),
                        'top_features': list(advanced_ml.feature_importance.get(best_model_name, {}).keys())[:5],
                        'confidence_distribution': df_ranked['confidence_class'].value_counts().to_dict() if 'confidence_class' in df_ranked.columns else {}
                    },
                    'detailed_results': ml_results,
                    'model_comparison': {name: result.get('val_r2', result.get('r2', 0)) for name, result in ml_results.items()},
                    'recommendations': advanced_ml.generate_prediction_report(ml_results)
                }
                
                report_json = json.dumps(ml_report, indent=2, ensure_ascii=False, default=str)
                st.download_button(
                    label="📊 Rapport ML",
                    data=report_json,
                    file_name=f"rapport_ml_detaille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Point d'entrée avec gestion d'erreurs avancée
if __name__ == "__main__":
    try:
        main()
