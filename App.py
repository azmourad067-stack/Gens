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
.param-active { background-color: #10b981; color: white; padding: 2px 8px; border-radius: 4px; }
.param-inactive { background-color: #6b7280; color: white; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

class AdvancedHorseRacingML:
    def __init__(self, params):
        self.params = params  # Stocker les paramètres
        self.base_models = {}
        self._initialize_models()
        
        # Modèle Deep Learning
        self.nn_model = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.cv_scores = {}
        self.is_trained = False
    
    def _initialize_models(self):
        """Initialise les modèles selon les paramètres sélectionnés"""
        if self.params['use_rf']:
            self.base_models['random_forest'] = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=8,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )
        
        if self.params['use_gb']:
            self.base_models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=6,
                min_samples_split=10, random_state=42
            )
        
        if self.params['use_ridge']:
            self.base_models['ridge'] = Ridge(alpha=1.0, random_state=42)
        
        if self.params['use_elastic']:
            self.base_models['elastic'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    
    def build_neural_network(self, input_dim):
        """Construction d'un réseau de neurones profond"""
        if not self.params['use_nn']:
            return None
            
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
            
            Dense(1, activation='sigmoid')
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
        positions = []
        
        # Extraction des positions
        for char in music.split():
            if char.endswith('p') and char[:-1].isdigit():
                positions.append(int(char[:-1]))
        
        if not positions:
            # Fallback: chercher tous les nombres
            numbers = re.findall(r'\d+', music)
            positions = [int(n) for n in numbers if int(n) > 0]
        
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
            'consistency': 1 / (np.std(positions) + 0.1),
            'improvement_trend': self.calculate_trend(positions),
            'performance_count': len(positions)
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
        return -slope
    
    def prepare_advanced_features_from_csv(self, df):
        """Préparation des features à partir du format CSV"""
        features = pd.DataFrame()
        
        # === FEATURES DE BASE AVEC PARAMÈTRES ===
        if self.params['weight_odds'] > 0:
            features['cote_direct'] = df['cotedirect'].fillna(50)
            features['cote_prob'] = df['coteprob'].fillna(50)
            features['odds_inv'] = 1 / (features['cote_direct'] + 0.1)
            features['log_odds'] = np.log1p(features['cote_direct'])
        
        # === FEATURES HISTORIQUES AVEC PARAMÈTRES ===
        if self.params['weight_historical'] > 0:
            features['age'] = df['age'].fillna(4)
            features['sexe'] = df['sexe'].map({'M': 0, 'F': 1, 'H': 0}).fillna(0)
            features['recence'] = df['recence'].fillna(100)
            
            # Statistiques cheval
            features['courses_cheval'] = df['coursescheval'].fillna(0)
            features['victoires_cheval'] = df['victoirescheval'].fillna(0)
            features['places_cheval'] = df['placescheval'].fillna(0)
            features['win_rate_cheval'] = features['victoires_cheval'] / (features['courses_cheval'] + 1)
            features['place_rate_cheval'] = features['places_cheval'] / (features['courses_cheval'] + 1)
            
            # Analyse musicale
            if 'musiqueche' in df.columns:
                music_features = df['musiqueche'].apply(self.parse_music_notation)
                music_df = pd.DataFrame(music_features.tolist(), 
                                      columns=[f'music_{i}' for i in range(10)])
                features = pd.concat([features, music_df], axis=1)
        
        # === FEATURES JOCKEY/ENTRAÎNEUR AVEC PARAMÈTRES ===
        if self.params['weight_jockey'] > 0:
            features['courses_jockey'] = df['coursesjockey'].fillna(0)
            features['victoires_jockey'] = df['victoiresjockey'].fillna(0)
            features['places_jockey'] = df['placejockey'].fillna(0)
            features['win_rate_jockey'] = features['victoires_jockey'] / (features['courses_jockey'] + 1)
            features['place_rate_jockey'] = features['places_jockey'] / (features['courses_jockey'] + 1)
            
            features['courses_entraineur'] = df['coursesentraineur'].fillna(0)
            features['victoires_entraineur'] = df['victoiresentraineur'].fillna(0)
            features['win_rate_entraineur'] = features['victoires_entraineur'] / (features['courses_entraineur'] + 1)
        
        # === FEATURES DE POSITION (toujours incluses) ===
        features['num_corde'] = df['numero'].fillna(1)
        features['partant'] = df['partant'].fillna(1)
        
        # === FEATURES D'INTERACTION ===
        if self.params['weight_historical'] > 0 and self.params['weight_jockey'] > 0:
            features['interaction_jockey_cheval'] = features.get('win_rate_jockey', 0) * features.get('win_rate_cheval', 0)
            features['interaction_entraineur_cheval'] = features.get('win_rate_entraineur', 0) * features.get('win_rate_cheval', 0)
        
        # === GAINS (toujours inclus) ===
        features['gains_carriere'] = np.log1p(df['gainsCarriere'].fillna(0))
        features['gains_victoires'] = np.log1p(df['gainsVictoires'].fillna(0))
        features['gains_annee_cours'] = np.log1p(df['gainsAnneeEnCours'].fillna(0))
        
        # === CONTEXTE DE COURSE ===
        features['nb_partants'] = df['partant'].max() if 'partant' in df.columns else 16
        
        return features.fillna(0)
    
    def create_synthetic_labels(self, X, df):
        """Création de labels synthétiques BASÉS SUR LES PARAMÈTRES"""
        y_synthetic = np.zeros(len(X))
        total_weight = 0
        
        # Cotes (si activé)
        if self.params['weight_odds'] > 0 and 'odds_inv' in X.columns:
            y_synthetic += X['odds_inv'] * self.params['weight_odds']
            total_weight += self.params['weight_odds']
        
        # Historique (si activé)
        if self.params['weight_historical'] > 0:
            historical_components = 0
            historical_count = 0
            
            if 'win_rate_cheval' in X.columns:
                historical_components += X['win_rate_cheval']
                historical_count += 1
            
            if 'music_5' in X.columns:  # win_rate musique
                historical_components += X['music_5']
                historical_count += 1
            
            if 'recence' in X.columns:
                historical_components += (1 / (X['recence'] + 1))
                historical_count += 1
            
            if historical_count > 0:
                y_synthetic += (historical_components / historical_count) * self.params['weight_historical']
                total_weight += self.params['weight_historical']
        
        # Jockey/Entraîneur (si activé)
        if self.params['weight_jockey'] > 0:
            jockey_components = 0
            jockey_count = 0
            
            if 'win_rate_jockey' in X.columns:
                jockey_components += X['win_rate_jockey']
                jockey_count += 1
            
            if 'win_rate_entraineur' in X.columns:
                jockey_components += X['win_rate_entraineur']
                jockey_count += 1
            
            if jockey_count > 0:
                y_synthetic += (jockey_components / jockey_count) * self.params['weight_jockey']
                total_weight += self.params['weight_jockey']
        
        # Position (toujours inclus - poids résiduel)
        position_weight = 1.0 - total_weight
        if position_weight > 0 and 'num_corde' in X.columns:
            y_synthetic += (1 / (X['num_corde'] + 1)) * position_weight
        
        # Normalisation
        if y_synthetic.max() > y_synthetic.min():
            y_synthetic = (y_synthetic - y_synthetic.min()) / (y_synthetic.max() - y_synthetic.min())
        
        # Bruit pour éviter le surapprentissage
        y_synthetic += np.random.normal(0, 0.02, len(y_synthetic))
        
        return np.clip(y_synthetic, 0, 1)
    
    def train_ensemble_model(self, X, y):
        """Entraînement de l'ensemble de modèles"""
        if len(self.base_models) == 0 and not self.params['use_nn']:
            st.error("❌ Aucun modèle sélectionné !")
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Validation croisée seulement si assez de données
        if len(X) >= 10:
            kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
            
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
                    st.warning(f"⚠️ Erreur modèle {name}: {e}")
        
        # Entraînement direct si pas assez de données pour CV
        else:
            for name, model in self.base_models.items():
                try:
                    model.fit(X_scaled, y)
                    # Prédiction pour calcul R²
                    pred = model.predict(X_scaled)
                    r2 = r2_score(y, pred)
                    self.cv_scores[name] = {
                        'mean': r2,
                        'std': 0,
                        'scores': [r2]
                    }
                except Exception as e:
                    st.warning(f"⚠️ Erreur modèle {name}: {e}")
        
        # Réseau neuronal
        if self.params['use_nn'] and len(X) >= 8:
            try:
                self.nn_model = self.build_neural_network(X_scaled.shape[1])
                if self.nn_model:
                    # Pour petites datasets, on utilise tous les données
                    if len(X) < 20:
                        history = self.nn_model.fit(
                            X_scaled, y,
                            epochs=50,
                            batch_size=8,
                            verbose=0,
                            validation_split=0.2
                        )
                    else:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=42
                        )
                        
                        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
                        
                        history = self.nn_model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=100,
                            batch_size=16,
                            callbacks=[early_stop, reduce_lr],
                            verbose=0
                        )
                    
                    # Évaluation
                    nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
                    nn_r2 = r2_score(y, nn_pred)
                    self.cv_scores['neural_network'] = {
                        'mean': nn_r2,
                        'std': 0,
                        'scores': [nn_r2]
                    }
                    
            except Exception as e:
                st.warning(f"⚠️ Erreur réseau neuronal: {e}")
                self.nn_model = None
        
        self.is_trained = True
    
    def predict(self, X):
        """Prédiction avec pondération des modèles"""
        if not self.is_trained or (len(self.base_models) == 0 and self.nn_model is None):
            return np.zeros(len(X)), np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        weights = []
        
        # Prédictions des modèles de base
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
                # Poids basé sur performance R²
                model_r2 = max(0, self.cv_scores.get(name, {'mean': 0.5})['mean'])
                weights.append(model_r2)
            except Exception as e:
                st.warning(f"⚠️ Prédiction échouée pour {name}: {e}")
        
        # Prédiction réseau neuronal
        if self.nn_model is not None:
            try:
                nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
                predictions.append(nn_pred)
                nn_r2 = max(0, self.cv_scores.get('neural_network', {'mean': 0.5})['mean'])
                weights.append(nn_r2)
            except Exception as e:
                st.warning(f"⚠️ Prédiction NN échouée: {e}")
        
        if not predictions:
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Conversion en array numpy
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Éviter division par zéro
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        
        weights = weights / weights.sum()
        
        # Moyenne pondérée
        final_predictions = np.average(predictions, axis=0, weights=weights)
        
        # Confiance basée sur variance des prédictions
        confidence = 1 / (1 + np.std(predictions, axis=0))
        
        return final_predictions, confidence

def load_and_prepare_csv_data(uploaded_file):
    """Chargement et préparation des données CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier chargé: {len(df)} chevaux détectés")
        return df
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        return None

def display_parameter_status(params):
    """Affiche le statut des paramètres"""
    st.subheader("🎛️ Paramètres Actifs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        odds_status = "🟢 ACTIF" if params['weight_odds'] > 0 else "🔴 INACTIF"
        st.markdown(f"**Cotes**: {odds_status} ({params['weight_odds']:.0%})")
    
    with col2:
        hist_status = "🟢 ACTIF" if params['weight_historical'] > 0 else "🔴 INACTIF"
        st.markdown(f"**Historique**: {hist_status} ({params['weight_historical']:.0%})")
    
    with col3:
        jockey_status = "🟢 ACTIF" if params['weight_jockey'] > 0 else "🔴 INACTIF"
        st.markdown(f"**Jockey**: {jockey_status} ({params['weight_jockey']:.0%})")
    
    # Modèles activés
    st.markdown("**🧠 Modèles**: " + ", ".join([
        "RF" if params['use_rf'] else "",
        "GB" if params['use_gb'] else "", 
        "Ridge" if params['use_ridge'] else "",
        "Elastic" if params['use_elastic'] else "",
        "NN" if params['use_nn'] else ""
    ]).strip(", "))

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro+</h1>', unsafe_allow_html=True)
    st.markdown("*Deep Learning & Analyse Prédictive Avancée*")
    
    # ==================== SIDEBAR AVEC PARAMÈTRES LIÉS ====================
    with st.sidebar:
        st.header("⚙️ Configuration des Poids")
        
        # Stocker les valeurs dans des variables qui SONT utilisées
        weight_odds = st.slider("🎯 Poids des Cotes", 0.0, 1.0, 0.45, 0.05, 
                               help="Influence des cotes directes et probables")
        
        weight_historical = st.slider("📊 Poids Historique", 0.0, 1.0, 0.10, 0.05,
                                     help="Influence des performances passées du cheval")
        
        weight_jockey = st.slider("👨 Poids Jockey/Entraîneur", 0.0, 1.0, 0.25, 0.05,
                                 help="Influence des statistiques jockey et entraîneur")
        
        # Vérification cohérence des poids
        total_weights = weight_odds + weight_historical + weight_jockey
        if total_weights > 1.0:
            st.warning(f"⚠️ Total des poids: {total_weights:.0%} > 100%")
        elif total_weights < 0.3:
            st.warning(f"⚠️ Total des poids faible: {total_weights:.0%}")
        
        st.header("🧠 Sélection des Modèles")
        use_rf = st.checkbox("🌲 Random Forest", value=True)
        use_gb = st.checkbox("📈 Gradient Boosting", value=True) 
        use_ridge = st.checkbox("📊 Ridge Regression", value=False)
        use_elastic = st.checkbox("🎯 Elastic Net", value=False)
        use_nn = st.checkbox("🧠 Réseau Neuronal", value=True)
        
        # Paramètres regroupés
        params = {
            'weight_odds': weight_odds,
            'weight_historical': weight_historical, 
            'weight_jockey': weight_jockey,
            'use_rf': use_rf,
            'use_gb': use_gb,
            'use_ridge': use_ridge,
            'use_elastic': use_elastic,
            'use_nn': use_nn
        }
    
    # ==================== INTERFACE PRINCIPALE ====================
    tab1, tab2 = st.tabs(["📁 Analyse CSV", "🌐 Analyse URL"])
    
    with tab1:
        st.subheader("📊 Analyse de Fichier CSV")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file:
            df = load_and_prepare_csv_data(uploaded_file)
            
            if df is not None:
                # Afficher le statut des paramètres
                display_parameter_status(params)
                
                st.subheader("🔍 Aperçu des Données")
                st.dataframe(df.head(), use_container_width=True)
                
                # Métriques de base
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🐎 Chevaux", len(df))
                with col2:
                    st.metric("📈 Colonnes", len(df.columns))
                with col3:
                    if 'cotedirect' in df.columns:
                        st.metric("🎯 Cote moyenne", f"{df['cotedirect'].mean():.1f}")
                
                # Bouton d'analyse
                if st.button("🚀 Lancer l'Analyse avec les Paramètres", type="primary"):
                    with st.spinner("🤖 Entraînement des modèles avec vos paramètres..."):
                        # Initialisation du modèle AVEC les paramètres
                        ml_model = AdvancedHorseRacingML(params)
                        
                        # Préparation des features (respecte les paramètres)
                        X = ml_model.prepare_advanced_features_from_csv(df)
                        
                        # Affichage des features générées
                        st.info(f"🔧 **{len(X.columns)} features** générées selon vos paramètres")
                        
                        # Création des labels (respecte les paramètres)
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
                        st.subheader("🏆 Classement Final (avec vos paramètres)")
                        
                        # Top 10
                        display_cols = ['cheval', 'cotedirect', 'score_ia', 'confiance', 'rang_ia']
                        if 'jockey' in results_df.columns:
                            display_cols.append('jockey')
                        
                        top_10 = results_df.nlargest(10, 'score_ia')[display_cols]
                        
                        for idx, row in top_10.iterrows():
                            conf = row['confiance']
                            if conf > 0.7:
                                conf_emoji, conf_class = "🟢", "confidence-high"
                            elif conf > 0.5:
                                conf_emoji, conf_class = "🟡", "confidence-medium" 
                            else:
                                conf_emoji, conf_class = "🔴", "confidence-low"
                            
                            jockey_info = f" | 👨 {row['jockey']}" if 'jockey' in row else ""
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <strong>{int(row['rang_ia'])}. {row['cheval']}</strong>{jockey_info}<br>
                                📊 Cote: <strong>{row['cotedirect']}</strong> | 
                                🎯 Score: <strong>{row['score_ia']:.3f}</strong> | 
                                {conf_emoji} Confiance: <span class="{conf_class}">{conf:.1%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Métriques de performance
                        st.subheader("📈 Performance des Modèles")
                        if ml_model.cv_scores:
                            cols = st.columns(len(ml_model.cv_scores))
                            for idx, (name, scores) in enumerate(ml_model.cv_scores.items()):
                                with cols[idx % len(cols)]:
                                    emoji = "🌲" if "forest" in name else "📈" if "boost" in name else "🧠" if "neural" in name else "📊"
                                    st.metric(f"{emoji} {name}", f"{scores['mean']:.3f}")
                        
                        # Impact des paramètres
                        st.subheader("🔍 Impact de Vos Paramètres")
                        param_impact = pd.DataFrame({
                            'Paramètre': ['Cotes', 'Historique', 'Jockey/Entraîneur'],
                            'Poids': [params['weight_odds'], params['weight_historical'], params['weight_jockey']],
                            'Statut': ['🟢 Actif' if w > 0 else '🔴 Inactif' for w in 
                                      [params['weight_odds'], params['weight_historical'], params['weight_jockey']]]
                        })
                        
                        st.dataframe(param_impact, use_container_width=True)
                        
                        # Exporter les résultats
                        st.subheader("💾 Exporter les Résultats")
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "📄 Télécharger CSV",
                            csv_data,
                            f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )
    
    with tab2:
        st.subheader("🌐 Analyse par URL")
        st.info("Fonctionnalité en développement...")

if __name__ == "__main__":
    main()
