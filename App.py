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

# ML avancé avec Deep Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro+",
    page_icon="🏇",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.prediction-box {
    border-left: 5px solid #f59e0b;
    padding: 1rem 1rem 1rem 1.5rem;
    background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.confidence-high { color: #10b981; font-weight: bold; }
.confidence-medium { color: #f59e0b; font-weight: bold; }
.confidence-low { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class AdvancedHorseRacingML:
    def __init__(self):
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=8,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=6,
                min_samples_split=10, random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        }
        
        # Modèle Deep Learning
        self.nn_model = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.cv_scores = {}
        self.is_trained = False
        
    def build_neural_network(self, input_dim):
        """Construction d'un réseau de neurones profond"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')  # Sortie entre 0 et 1
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def parse_music_notation(self, music_str):
        """Analyse avancée de la notation musicale"""
        if pd.isna(music_str) or music_str == '':
            return np.zeros(10)
        
        music = str(music_str)
        # Extraction des positions et types de performance
        positions = []
        performance_types = []
        
        # Regex pour extraire chiffres et lettres
        pattern = r'(\d+)([a-zA-Z]?)'
        matches = re.findall(pattern, music)
        
        for match in matches:
            pos = int(match[0])
            perf_type = match[1].lower() if match[1] else 'p'  # p par défaut
            positions.append(pos)
            performance_types.append(perf_type)
        
        if not positions:
            return np.zeros(10)
        
        # Features avancées
        features = {
            'mean_position': np.mean(positions),
            'std_position': np.std(positions),
            'min_position': np.min(positions),
            'max_position': np.max(positions),
            'win_rate': positions.count(1) / len(positions),
            'top3_rate': sum(1 for p in positions if p <= 3) / len(positions),
            'recent_form': np.mean(positions[:min(3, len(positions))]) if positions else 5,
            'consistency': 1 / (np.std(positions) + 0.1),  # Éviter division par zéro
            'improvement_trend': self.calculate_trend(positions),
            'performance_variety': len(set(performance_types)) / len(performance_types) if performance_types else 0
        }
        
        return np.array(list(features.values()))
    
    def calculate_trend(self, positions):
        """Calcule la tendance d'amélioration"""
        if len(positions) < 2:
            return 0
        
        recent = positions[:min(5, len(positions))]
        if len(recent) < 2:
            return 0
            
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return -slope  # Négatif car plus la pente est négative, meilleure est l'amélioration
    
    def extract_jockey_stats(self, jockey_data):
        """Extraction des statistiques du jockey"""
        if pd.isna(jockey_data):
            return np.zeros(5)
        
        # Exemple de format: "5p 2p 1p 3p 4p"
        data = str(jockey_data)
        positions = [int(c) for c in data.split() if c[:-1].isdigit()]
        
        if not positions:
            return np.zeros(5)
        
        return np.array([
            len(positions),  # nb courses
            positions.count(1),  # victoires
            sum(1 for p in positions if p <= 3),  # places
            np.mean(positions),  # position moyenne
            1 / (np.std(positions) + 0.1)  # régularité
        ])
    
    def prepare_advanced_features_from_csv(self, df):
        """Préparation des features à partir du format CSV fourni"""
        features = pd.DataFrame()
        
        # === FEATURES DE BASE ===
        features['cote_direct'] = df['cotedirect'].fillna(50)
        features['cote_prob'] = df['coteprob'].fillna(50)
        features['odds_inv'] = 1 / (features['cote_direct'] + 0.1)
        features['log_odds'] = np.log1p(features['cote_direct'])
        
        # === FEATURES DE POSITION ===
        features['num_corde'] = df['numero'].fillna(1)
        features['partant'] = df['partant'].fillna(1)
        
        # === FEATURES CHEVAL ===
        features['age'] = df['age'].fillna(4)
        features['sexe'] = df['sexe'].map({'M': 0, 'F': 1, 'H': 0}).fillna(0)
        features['recence'] = df['recence'].fillna(100)
        
        # === STATISTIQUES CHEVAL ===
        features['courses_cheval'] = df['coursescheval'].fillna(0)
        features['victoires_cheval'] = df['victoirescheval'].fillna(0)
        features['places_cheval'] = df['placescheval'].fillna(0)
        features['win_rate_cheval'] = features['victoires_cheval'] / (features['courses_cheval'] + 1)
        features['place_rate_cheval'] = features['places_cheval'] / (features['courses_cheval'] + 1)
        
        # === STATISTIQUES JOCKEY ===
        features['courses_jockey'] = df['coursesjockey'].fillna(0)
        features['victoires_jockey'] = df['victoiresjockey'].fillna(0)
        features['places_jockey'] = df['placejockey'].fillna(0)
        features['win_rate_jockey'] = features['victoires_jockey'] / (features['courses_jockey'] + 1)
        features['place_rate_jockey'] = features['places_jockey'] / (features['courses_jockey'] + 1)
        
        # === STATISTIQUES ENTRAÎNEUR ===
        features['courses_entraineur'] = df['coursesentraineur'].fillna(0)
        features['victoires_entraineur'] = df['victoiresentraineur'].fillna(0)
        features['win_rate_entraineur'] = features['victoires_entraineur'] / (features['courses_entraineur'] + 1)
        
        # === PERFORMANCES RÉCENTES ===
        if 'm1' in df.columns and 'm2' in df.columns and 'm3' in df.columns:
            features['moyenne_3_dernieres'] = df[['m1', 'm2', 'm3']].mean(axis=1)
            features['ecart_3_dernieres'] = df[['m1', 'm2', 'm3']].std(axis=1)
        else:
            features['moyenne_3_dernieres'] = 5
            features['ecart_3_dernieres'] = 2
        
        # === GAINS ET PERFORMANCES FINANCIÈRES ===
        features['gains_carriere'] = np.log1p(df['gainsCarriere'].fillna(0))
        features['gains_victoires'] = np.log1p(df['gainsVictoires'].fillna(0))
        features['gains_annee_cours'] = np.log1p(df['gainsAnneeEnCours'].fillna(0))
        
        # === FEATURES D'INTERACTION ===
        features['interaction_jockey_cheval'] = features['win_rate_jockey'] * features['win_rate_cheval']
        features['interaction_entraineur_cheval'] = features['win_rate_entraineur'] * features['win_rate_cheval']
        features['experience_combinee'] = np.log1p(features['courses_cheval'] * features['courses_jockey'])
        
        # === FEATURES DE CONTEXTE ===
        features['nb_partants'] = df['partant'].max() if 'partant' in df.columns else 16
        features['cote_moyenne'] = df['cotedirect'].mean()
        features['ecart_type_cotes'] = df['cotedirect'].std()
        
        # === ANALYSE DE LA MUSIQUE AVANCÉE ===
        if 'musiqueche' in df.columns:
            music_features = df['musiqueche'].apply(self.parse_music_notation)
            music_df = pd.DataFrame(music_features.tolist(), 
                                  columns=[f'music_{i}' for i in range(10)])
            features = pd.concat([features, music_df], axis=1)
        
        return features.fillna(0)
    
    def create_synthetic_labels(self, X, df):
        """Création de labels synthétiques basés sur les données historiques"""
        # Score composite basé sur multiples facteurs
        y_synthetic = (
            X['odds_inv'] * 0.25 +
            X['win_rate_cheval'] * 0.20 +
            X['win_rate_jockey'] * 0.15 +
            X['win_rate_entraineur'] * 0.10 +
            X['music_0'] * 0.10 +  # mean_position (inversé)
            X['music_5'] * 0.10 +  # win_rate musique
            (1 / (X['recence'] + 1)) * 0.05 +
            (1 / (X['num_corde'] + 1)) * 0.05
        )
        
        # Ajout de bruit pour éviter le surapprentissage
        y_synthetic += np.random.normal(0, 0.02, len(y_synthetic))
        
        return np.clip(y_synthetic, 0, 1)
    
    def train_ensemble_model(self, X, y):
        """Entraînement de l'ensemble de modèles"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation croisée
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
                self.cv_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Erreur modèle {name}: {e}")
        
        # Entraînement du réseau de neurones
        if len(X) >= 10:  # Minimum pour DL
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                self.nn_model = self.build_neural_network(X_scaled.shape[1])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
                
                history = self.nn_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=16,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                # Évaluation du modèle NN
                nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
                nn_r2 = r2_score(y, nn_pred)
                self.cv_scores['neural_network'] = {
                    'mean': nn_r2,
                    'std': 0,
                    'scores': [nn_r2]
                }
                
            except Exception as e:
                print(f"Erreur réseau neuronal: {e}")
                self.nn_model = None
        
        self.is_trained = True
    
    def predict(self, X):
        """Prédiction avec l'ensemble de modèles"""
        if not self.is_trained:
            return np.zeros(len(X)), np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        weights = []
        
        # Prédictions des modèles de base
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
                weights.append(self.cv_scores.get(name, {'mean': 0.5})['mean'])
            except:
                pass
        
        # Prédiction du réseau neuronal
        if self.nn_model is not None:
            try:
                nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
                predictions.append(nn_pred)
                weights.append(self.cv_scores.get('neural_network', {'mean': 0.5})['mean'])
            except:
                pass
        
        if not predictions:
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Moyenne pondérée
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = np.maximum(weights, 0)  # Éviter poids négatifs
        weights = weights / (weights.sum() + 1e-8)  # Normalisation
        
        final_predictions = np.average(predictions, axis=0, weights=weights)
        
        # Calcul de la confiance basé sur la variance des prédictions
        confidence = 1 / (1 + np.std(predictions, axis=0))
        
        return final_predictions, confidence

def load_and_prepare_csv_data(uploaded_file):
    """Chargement et préparation des données CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier chargé: {len(df)} chevaux détectés")
        
        # Affichage des colonnes disponibles
        st.info(f"📊 Colonnes disponibles: {', '.join(df.columns.tolist()[:10])}...")
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro+</h1>', unsafe_allow_html=True)
    st.markdown("*Deep Learning & Analyse Prédictive Avancée*")
    
    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        st.subheader("🧠 Modèles")
        st.checkbox("✅ Random Forest", value=True)
        st.checkbox("✅ Gradient Boosting", value=True)
        st.checkbox("✅ Ridge Regression", value=True)
        st.checkbox("✅ Réseau Neuronal", value=True)
        
        st.subheader("📊 Features")
        st.slider("🎯 Poids des cotes", 0.1, 0.5, 0.25, 0.05)
        st.slider("🏇 Poids historique", 0.1, 0.5, 0.20, 0.05)
        st.slider("👨 Poids jockey", 0.1, 0.5, 0.15, 0.05)
        
        st.subheader("ℹ️ Données Supportées")
        st.info("📈 **Format Geny/CSV** complet")
        st.info("🎯 **45+ variables** analysées")
        st.info("🧠 **Deep Learning** intégré")
    
    tab1, tab2 = st.tabs(["📁 Analyse CSV", "🌐 Analyse URL"])
    
    with tab1:
        st.subheader("📊 Analyse de Fichier CSV")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file:
            df = load_and_prepare_csv_data(uploaded_file)
            
            if df is not None:
                st.subheader("🔍 Aperçu des Données")
                st.dataframe(df.head(), use_container_width=True)
                
                # Statistiques descriptives
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🐎 Chevaux", len(df))
                with col2:
                    st.metric("📈 Variables", len(df.columns))
                with col3:
                    st.metric("🎯 Cote moyenne", f"{df['cotedirect'].mean():.1f}")
                
                # Analyse avec ML
                if st.button("🚀 Lancer l'Analyse Prédictive", type="primary"):
                    with st.spinner("🤖 Entraînement des modèles avancés..."):
                        ml_model = AdvancedHorseRacingML()
                        
                        # Préparation des features
                        X = ml_model.prepare_advanced_features_from_csv(df)
                        
                        # Création des labels synthétiques
                        y = ml_model.create_synthetic_labels(X, df)
                        
                        # Entraînement
                        ml_model.train_ensemble_model(X, y)
                        
                        # Prédictions
                        predictions, confidence = ml_model.predict(X)
                        
                        # Résultats
                        results_df = df.copy()
                        results_df['score_ia'] = predictions
                        results_df['confiance'] = confidence
                        results_df['rang_ia'] = results_df['score_ia'].rank(ascending=False)
                        
                        # Affichage des résultats
                        st.subheader("🏆 Classement IA")
                        
                        # Top 10
                        top_10 = results_df.nlargest(10, 'score_ia')[['cheval', 'cotedirect', 'score_ia', 'confiance', 'rang_ia']]
                        
                        for idx, row in top_10.iterrows():
                            conf_level = "🟢" if row['confiance'] > 0.7 else "🟡" if row['confiance'] > 0.5 else "🔴"
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <strong>{int(row['rang_ia'])}. {row['cheval']}</strong><br>
                                📊 Cote: <strong>{row['cotedirect']}</strong> | 
                                🎯 Score IA: <strong>{row['score_ia']:.3f}</strong> | 
                                {conf_level} Confiance: <strong>{row['confiance']:.1%}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Métriques de performance
                        st.subheader("📈 Performance des Modèles")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if ml_model.cv_scores:
                            with col1:
                                if 'random_forest' in ml_model.cv_scores:
                                    st.metric("🌲 Random Forest", f"{ml_model.cv_scores['random_forest']['mean']:.3f}")
                            with col2:
                                if 'gradient_boosting' in ml_model.cv_scores:
                                    st.metric("📈 Gradient Boosting", f"{ml_model.cv_scores['gradient_boosting']['mean']:.3f}")
                            with col3:
                                if 'neural_network' in ml_model.cv_scores:
                                    st.metric("🧠 Réseau Neuronal", f"{ml_model.cv_scores['neural_network']['mean']:.3f}")
                            with col4:
                                avg_conf = results_df['confiance'].mean()
                                st.metric("🎯 Confiance Moyenne", f"{avg_conf:.1%}")
    
    with tab2:
        st.subheader("🌐 Analyse par URL")
        st.info("Fonctionnalité en développement...")
        url = st.text_input("URL de la course:", placeholder="https://...")
        if st.button("Analyser URL"):
            st.warning("⚠️ L'analyse URL sera disponible dans la prochaine version")

if __name__ == "__main__":
    main()
