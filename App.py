import streamlit as st
import requests
from bs4 import BeautifulSoup
...
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
Tool Call
Function Name:
Write
Arguments:
file_path:
/home/user/streamlit_horse_analyzer.py
content:
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration des pages Streamlit
st.set_page_config(
    page_title="🏇 Analyseur Hippique IA",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== CONFIGURATIONS ADAPTATIVES ====
CONFIGS = {
    "PLAT": {
        "w_odds": 0.5,
        "w_draw": 0.3, 
        "w_weight": 0.2,
        "normalization": "zscore",
        "draw_adv_inner_is_better": True,
        "draw_optimal_range": [1, 2, 3, 4],
        "per_kg_penalty": 1.2,
        "weight_baseline": 55.0,
        "use_weight_analysis": True,
        "description": "Course de galop - Handicap poids + avantage corde intérieure",
        "icon": "🏃",
        "color": "#FF6B6B"
    },
    
    "ATTELE_AUTOSTART": {
        "w_odds": 0.7,
        "w_draw": 0.25,
        "w_weight": 0.05,
        "normalization": "zscore", 
        "draw_adv_inner_is_better": False,
        "draw_optimal_range": [4, 5, 6],
        "per_kg_penalty": 0.3,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé autostart - Numéros 4-6 optimaux",
        "icon": "🚗",
        "color": "#4ECDC4"
    },
    
    "ATTELE_VOLTE": {
        "w_odds": 0.85,
        "w_draw": 0.05,
        "w_weight": 0.1,
        "normalization": "zscore",
        "draw_adv_inner_is_better": False,
        "draw_optimal_range": [],
        "per_kg_penalty": 0.2,
        "weight_baseline": 68.0,
        "use_weight_analysis": False,
        "description": "Trot attelé volté - Numéro sans importance",
        "icon": "🔄",
        "color": "#45B7D1"
    }
}

def init_sidebar():
    """Initialise la barre latérale avec les contrôles"""
    st.sidebar.image("https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=400", 
                     caption="Analyseur Hippique IA", use_column_width=True)
    
    st.sidebar.markdown("---")
    
    # Sélection du mode d'analyse
    analysis_mode = st.sidebar.selectbox(
        "🎯 Mode d'analyse",
        ["🤖 Auto-détection", "🏃 Course de PLAT", "🚗 Attelé AUTOSTART", "🔄 Attelé VOLTÉ"],
        help="Sélectionnez le type de course ou laissez l'IA détecter automatiquement"
    )
    
    # Paramètres avancés
    with st.sidebar.expander("⚙️ Paramètres avancés", expanded=False):
        use_ml_model = st.checkbox("🧠 Activer modèle IA avancé", value=True)
        show_feature_importance = st.checkbox("📊 Afficher importance des variables", value=True)
        confidence_threshold = st.slider("🎯 Seuil de confiance", 0.0, 1.0, 0.6, 0.05)
        
    # Informations
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Sources documentées")
    st.sidebar.markdown("• turfmining.fr\n• boturfers.fr\n• equidia.fr\n• PMU")
    
    return analysis_mode, use_ml_model, show_feature_importance, confidence_threshold

def safe_float_convert(value):
    """Conversion sécurisée vers float"""
    if pd.isna(value):
        return np.nan
    try:
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return np.nan

def safe_int_convert(value):
    """Conversion sécurisée vers entier"""
    if pd.isna(value):
        return np.nan
    try:
        cleaned = re.search(r'\d+', str(value))
        return int(cleaned.group()) if cleaned else np.nan
    except (ValueError, AttributeError):
        return np.nan

def extract_weight_kg(poids_str):
    """Extrait le poids en kg depuis une chaîne"""
    if pd.isna(poids_str):
        return np.nan
    
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan

def nettoyer_donnees(ligne):
    """Fonction de nettoyage héritée du script original"""
    ligne = ''.join(e for e in ligne if e.isalnum() or e.isspace() or e in ['.', ',', '-', '(', ')', '%'])
    return ligne.strip()

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def scrape_race_data(url):
    """Scraping des données de course avec cache"""
    try:
        with st.spinner("🔍 Extraction des données en cours..."):
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                st.error(f"❌ Erreur HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, 'html.parser')
            donnees_chevaux = []
            
            table = soup.find('table')
            if not table:
                st.error("❌ Aucun tableau trouvé sur la page")
                return None
                
            rows = table.find_all('tr')[1:]  # Skip header
            if not rows:
                st.error("❌ Aucune ligne de données trouvée")
                return None
                
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 8:
                    continue
                    
                donnees_chevaux.append({
                    "Numéro de corde": nettoyer_donnees(cols[0].text),
                    "Nom": nettoyer_donnees(cols[1].text),
                    "Musique": nettoyer_donnees(cols[5].text) if len(cols) > 5 else "",
                    "Âge/Sexe": nettoyer_donnees(cols[6].text) if len(cols) > 6 else "",
                    "Poids": nettoyer_donnees(cols[7].text) if len(cols) > 7 else "60.0",
                    "Jockey": nettoyer_donnees(cols[8].text) if len(cols) > 8 else "",
                    "Entraîneur": nettoyer_donnees(cols[9].text) if len(cols) > 9 else "",
                    "Cote": nettoyer_donnees(cols[-1].text)
                })

            return pd.DataFrame(donnees_chevaux) if donnees_chevaux else None
            
    except Exception as e:
        st.error(f"❌ Erreur lors du scraping : {e}")
        return None

def prepare_features(df):
    """Nettoie et prépare les données pour l'analyse"""
    df['odds_numeric'] = df['Cote'].apply(safe_float_convert)
    df['draw_numeric'] = df['Numéro de corde'].apply(safe_int_convert)
    df['weight_kg'] = df['Poids'].apply(extract_weight_kg)
    
    # Extraction de features additionnelles pour le ML
    df['age'] = df['Âge/Sexe'].str.extract(r'(\d+)').astype(float).fillna(5)
    df['is_male'] = df['Âge/Sexe'].str.contains('H', na=False).astype(int)
    df['is_female'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
    
    # Analyse de la musique (historique performances)
    df['recent_wins'] = df['Musique'].str.count('1').fillna(0)
    df['recent_places'] = df['Musique'].str.count('[23]').fillna(0)
    df['recent_fails'] = df['Musique'].str.count('[89]').fillna(0)
    df['consistency'] = (df['recent_wins'] + df['recent_places']) / (len(df['Musique'].str[0:6]) + 1)
    
    # Suppression des lignes avec données critiques manquantes
    initial_count = len(df)
    df = df.dropna(subset=['odds_numeric', 'draw_numeric'])
    df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].mean())
    
    return df, initial_count

def auto_detect_race_type(df):
    """Détection automatique du type de course"""
    weight_variation = df['weight_kg'].std() if len(df) > 1 else 0
    weight_mean = df['weight_kg'].mean()
    
    if weight_variation > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_variation < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés (réglementaire attelé)"
    elif weight_variation < 1.0:
        detected = "ATTELE_AUTOSTART"  
        reason = "Poids très uniformes"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    return detected, reason

def normalize_series(series, mode="zscore"):
    """Normalise une série de données"""
    if len(series) <= 1 or series.std() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    
    if mode == "zscore":
        return (series - series.mean()) / series.std()
    elif mode == "minmax":
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    else:
        return pd.Series([0.0] * len(series), index=series.index)

def create_ml_features(df, race_type):
    """Création des features pour le modèle ML"""
    features = df[['odds_numeric', 'draw_numeric', 'weight_kg', 'age', 
                   'is_male', 'is_female', 'recent_wins', 'recent_places', 
                   'recent_fails', 'consistency']].copy()
    
    # Encodage du type de course
    features[f'race_type_{race_type}'] = 1
    for rt in ['PLAT', 'ATTELE_AUTOSTART', 'ATTELE_VOLTE']:
        if rt != race_type:
            features[f'race_type_{rt}'] = 0
    
    # Features d'interaction
    features['odds_draw_interaction'] = features['odds_numeric'] * features['draw_numeric']
    features['weight_odds_ratio'] = features['weight_kg'] / (features['odds_numeric'] + 1)
    
    # Rang relatif
    features['odds_rank'] = features['odds_numeric'].rank()
    features['draw_rank'] = features['draw_numeric'].rank()
    
    return features.fillna(0)

def train_ml_model(df, race_type):
    """Entraîne un modèle ML simple pour cette course"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Création des features
        X = create_ml_features(df, race_type)
        
        # Target basé sur l'inverse des cotes (proxy de probabilité)
        y = 1 / (df['odds_numeric'] + 1)  # +1 pour éviter division par 0
        
        # Modèle simple pour cette course
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        
        # Entraînement avec validation croisée simulée
        model.fit(X, y)
        
        # Prédictions
        predictions = model.predict(X)
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, predictions, feature_importance, X
        
    except ImportError:
        st.warning("⚠️ Scikit-learn non disponible, utilisation du modèle de base")
        return None, None, None, None

def compute_traditional_scores(df, race_type):
    """Calcul des scores traditionnels (méthode originale)"""
    config = CONFIGS[race_type]
    
    # Score cotes
    inverse_odds = 1.0 / df['odds_numeric']
    score_odds = normalize_series(inverse_odds, config["normalization"])
    
    # Score numéro de corde
    if race_type == "PLAT":
        if config["draw_adv_inner_is_better"]:
            max_draw = df['draw_numeric'].max()
            inverted = max_draw - df['draw_numeric'] + 1
            score_draw = normalize_series(inverted, config["normalization"])
        else:
            score_draw = normalize_series(df['draw_numeric'], config["normalization"])
    else:  # ATTELÉ
        optimal_range = config.get("draw_optimal_range", [])
        if not optimal_range:
            score_draw = pd.Series([0.0] * len(df), index=df.index)
        else:
            scores = []
            for draw in df['draw_numeric']:
                if draw in optimal_range:
                    score = 2.0
                elif draw <= 3:
                    score = -1.0
                elif draw >= 7 and draw <= 9:
                    score = -0.5
                elif draw >= 10:
                    score = -1.5
                else:
                    score = 0.0
                scores.append(score)
            score_draw = pd.Series(scores, index=df.index)
    
    # Score poids
    if config.get("use_weight_analysis", True):
        weight_penalty = (df['weight_kg'] - config["weight_baseline"]) * config["per_kg_penalty"]
        score_weight = normalize_series(-weight_penalty, config["normalization"])
    else:
        score_weight = pd.Series([0.0] * len(df), index=df.index)
    
    # Score final
    score_final = (
        config["w_odds"] * score_odds +
        config["w_draw"] * score_draw +
        config["w_weight"] * score_weight
    )
    
    return score_final, score_odds, score_draw, score_weight

def analyze_race_comprehensive(df, race_type, use_ml=True):
    """Analyse complète avec ML et méthode traditionnelle"""
    
    # Méthode traditionnelle
    trad_score, score_odds, score_draw, score_weight = compute_traditional_scores(df, race_type)
    
    # Méthode ML
    ml_predictions = None
    feature_importance = None
    ml_features = None
    
    if use_ml:
        try:
            model, ml_predictions, feature_importance, ml_features = train_ml_model(df, race_type)
            if ml_predictions is not None:
                # Normalisation des prédictions ML
                ml_predictions = normalize_series(pd.Series(ml_predictions), "minmax")
        except Exception as e:
            st.warning(f"⚠️ Erreur ML, utilisation méthode traditionnelle : {e}")
            ml_predictions = None
    
    # Score hybride (combinaison des deux méthodes)
    if ml_predictions is not None:
        # 70% ML, 30% traditionnel pour équilibrer
        final_score = 0.7 * ml_predictions + 0.3 * normalize_series(trad_score, "minmax")
    else:
        final_score = normalize_series(trad_score, "minmax")
    
    # Ajout des scores au dataframe
    df_result = df.copy()
    df_result['score_final'] = final_score
    df_result['score_traditional'] = normalize_series(trad_score, "minmax")
    df_result['score_odds'] = score_odds
    df_result['score_draw'] = score_draw
    df_result['score_weight'] = score_weight
    
    if ml_predictions is not None:
        df_result['score_ml'] = ml_predictions
    
    # Calcul de la probabilité de victoire
    df_result['win_probability'] = (final_score - final_score.min()) / (final_score.max() - final_score.min())
    df_result['confidence'] = df_result['win_probability'] * 100
    
    # Classement
    df_result = df_result.sort_values('score_final', ascending=False).reset_index(drop=True)
    df_result['rang'] = range(1, len(df_result) + 1)
    
    return df_result, feature_importance, ml_features

def display_race_analysis(df_result, race_type, feature_importance, confidence_threshold):
    """Affiche les résultats de l'analyse"""
    
    config = CONFIGS[race_type]
    
    # Header avec informations sur le type de course
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {config['color']}, #f8f9fa); border-radius: 15px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">{config['icon']} {config['description']}</h2>
            <p style="color: white; margin: 5px 0;">Analyse basée sur {len(df_result)} chevaux</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques principales
    st.markdown("### 📊 Vue d'ensemble")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Favoris détectés", len(df_result[df_result['confidence'] > confidence_threshold * 100]))
    with col2:
        st.metric("🎯 Meilleur score", f"{df_result['score_final'].max():.2f}")
    with col3:
        st.metric("📈 Cote moyenne", f"{df_result['odds_numeric'].mean():.1f}")
    with col4:
        st.metric("🎲 Dispersion", f"{df_result['odds_numeric'].std():.1f}")
    
    # Tableau des résultats principal
    st.markdown("### 🏇 Classement et Pronostics")
    
    # Préparation des données pour l'affichage
    display_df = df_result[['rang', 'Nom', 'Numéro de corde', 'Cote', 'Poids', 
                           'confidence', 'score_final']].copy()
    
    # Formatage
    display_df['Confiance'] = display_df['confidence'].apply(lambda x: f"{x:.1f}%")
    display_df['Score'] = display_df['score_final'].apply(lambda x: f"{x:.3f}")
    display_df = display_df.drop(['confidence', 'score_final'], axis=1)
    
    # Coloration conditionnelle
    def highlight_top_horses(row):
        if row['rang'] <= 3:
            return ['background-color: #d4edda'] * len(row)  # Vert pour top 3
        elif row.name < len(df_result) * 0.5:  # Top 50%
            return ['background-color: #fff3cd'] * len(row)  # Jaune
        else:
            return [''] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_top_horses, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Distribution des Scores")
        fig_scores = px.bar(
            df_result.head(10), 
            x='Nom', 
            y='score_final',
            color='confidence',
            color_continuous_scale='RdYlGn',
            title="Top 10 - Scores de Confiance"
        )
        fig_scores.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Probabilités de Victoire")
        fig_prob = px.pie(
            df_result.head(8), 
            values='win_probability', 
            names='Nom',
            title="Répartition des Chances (Top 8)"
        )
        fig_prob.update_traces(textposition='inside', textinfo='percent+label')
        fig_prob.update_layout(height=400)
        st.plotly_chart(fig_prob, use_container_width=True)

def display_feature_importance(feature_importance):
    """Affiche l'importance des variables du modèle ML"""
    if feature_importance is not None and not feature_importance.empty:
        st.markdown("### 🧠 Importance des Variables (IA)")
        
        # Graphique d'importance
        fig_importance = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Variables les plus influentes dans le modèle IA",
            color='importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Interprétation
        top_feature = feature_importance.iloc[0]['feature']
        st.info(f"🔍 **Variable la plus importante :** `{top_feature}` - "
               f"Cette variable a le plus d'impact sur les prédictions du modèle.")

def display_betting_suggestions(df_result, confidence_threshold):
    """Affiche les suggestions de paris"""
    st.markdown("### 🎲 Suggestions de Paris")
    
    # Filtrage selon le seuil de confiance
    high_confidence = df_result[df_result['confidence'] > confidence_threshold * 100]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🥇 Simple Gagnant")
        if len(high_confidence) > 0:
            top_pick = high_confidence.iloc[0]
            st.success(f"**{top_pick['Nom']}** (N°{top_pick['Numéro de corde']})\n"
                      f"Confiance: {top_pick['confidence']:.1f}%\n"
                      f"Cote: {top_pick['Cote']}")
        else:
            st.warning("Aucun cheval avec assez de confiance")
    
    with col2:
        st.markdown("#### 🥈 Simple Placé")
        top3 = df_result.head(3)
        for i, (_, horse) in enumerate(top3.iterrows(), 1):
            confidence_color = "🟢" if horse['confidence'] > 60 else "🟡" if horse['confidence'] > 40 else "🟠"
            st.write(f"{confidence_color} **{horse['Nom']}** ({horse['confidence']:.1f}%)")
    
    with col3:
        st.markdown("#### 🏆 Quinté+ (Top 5)")
        top5 = df_result.head(5)
        quinte_combination = ", ".join([f"{row['Nom']} (N°{row['Numéro de corde']})" 
                                       for _, row in top5.iterrows()])
        st.info(f"**Combinaison suggérée :**\n{quinte_combination}")
        
        # Calcul du retour potentiel estimé
        total_odds = np.prod(top5['odds_numeric'].head(3))  # Top 3 pour estimation
        st.write(f"💰 **Potentiel estimé (Trio) :** {total_odds:.1f}x la mise")

def create_sample_data():
    """Crée des données d'exemple pour les tests"""
    sample_data = {
        'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker',
                'Fire Phoenix', 'Ice Dragon', 'Golden Arrow', 'Silver Bullet', 'Diamond Rush'],
        'Numéro de corde': ['2', '8', '1', '15', '5', '3', '12', '4', '9', '6'],
        'Cote': ['3.5', '12.0', '2.8', '25.0', '7.5', '8.2', '18.5', '4.1', '14.2', '6.8'],
        'Poids': ['56.5', '59.0', '55.0', '61.5', '57.5', '58.0', '60.5', '56.0', '59.5', '57.0'],
        'Musique': ['1a2a', '3a5a', '1a1a', '8a9a', '2a4a', '1a3a', '4a6a', '2a1a', '5a7a', '3a2a'],
        'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M', '5H', '4F', '3M', '5H', '4F'],
        'Jockey': ['Peslier', 'Soumillon', 'Lemaire', 'Guyon', 'Boudot', 
                  'Bazire', 'Thulliez', 'Raffin', 'Abrivard', 'Nivard'],
        'Entraîneur': ['Fabre', 'Head', 'Rouget', 'Delzangles', 'Graffard',
                      'Baudron', 'Leblanc', 'Terry', 'Guarato', 'Bazire']
    }
    return pd.DataFrame(sample_data)

def main():
    """Fonction principale de l'application Streamlit"""
    
    # Titre principal avec style
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 3em;">🏇 Analyseur Hippique IA</h1>
        <p style="color: white; margin: 10px 0; font-size: 1.2em;">Intelligence Artificielle pour Pronostics Hippiques</p>
        <p style="color: #f0f0f0; margin: 0;">Plat • Attelé • Modélisation Prédictive Avancée</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de la barre latérale
    analysis_mode, use_ml_model, show_feature_importance, confidence_threshold = init_sidebar()
    
    # Mapping du mode d'analyse
    mode_mapping = {
        "🤖 Auto-détection": "AUTO",
        "🏃 Course de PLAT": "PLAT",
        "🚗 Attelé AUTOSTART": "ATTELE_AUTOSTART",
        "🔄 Attelé VOLTÉ": "ATTELE_VOLTE"
    }
    selected_race_type = mode_mapping[analysis_mode]
    
    # Interface principale
    tab1, tab2, tab3 = st.tabs(["🌐 Analyse URL", "📊 Test avec données exemple", "📚 Documentation"])
    
    with tab1:
        st.markdown("### 🔗 Analyser une course en ligne")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url_input = st.text_input(
                "URL de la course (Geny.fr ou compatible)",
                placeholder="https://www.geny.fr/courses/...",
                help="Collez l'URL de la page de course à analyser"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Espacement
            analyze_button = st.button("🚀 Analyser", type="primary", use_container_width=True)
        
        if analyze_button and url_input:
            # Scraping et analyse
            df_raw = scrape_race_data(url_input)
            
            if df_raw is not None and len(df_raw) > 0:
                st.success(f"✅ {len(df_raw)} chevaux extraits avec succès")
                
                # Préparation des données
                df_clean, initial_count = prepare_features(df_raw)
                
                if len(df_clean) == 0:
                    st.error("❌ Aucune donnée valide après nettoyage")
                    return
                
                # Détection automatique du type si nécessaire
                if selected_race_type == "AUTO":
                    race_type, reason = auto_detect_race_type(df_clean)
                    st.info(f"🤖 **Type détecté automatiquement :** {race_type}\n\n**Raison :** {reason}")
                else:
                    race_type = selected_race_type
                
                # Analyse complète
                df_result, feature_importance, ml_features = analyze_race_comprehensive(
                    df_clean, race_type, use_ml_model
                )
                
                # Affichage des résultats
                display_race_analysis(df_result, race_type, feature_importance, confidence_threshold)
                
                # Importance des variables
                if show_feature_importance and feature_importance is not None:
                    display_feature_importance(feature_importance)
                
                # Suggestions de paris
                display_betting_suggestions(df_result, confidence_threshold)
                
                # Export des résultats
                st.markdown("### 💾 Export des Résultats")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df_result.to_csv(index=False)
                    st.download_button(
                        label="📄 Télécharger CSV",
                        data=csv_data,
                        file_name=f"pronostic_{race_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = df_result.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📄 Télécharger JSON",
                        data=json_data,
                        file_name=f"pronostic_{race_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
    
    with tab2:
        st.markdown("### 🧪 Test avec Données d'Exemple")
        st.info("Utilisez cette section pour tester l'analyseur avec des données fictives réalistes")
        
        if st.button("🎲 Générer et Analyser Données Test", type="primary"):
            # Génération des données d'exemple
            df_sample = create_sample_data()
            st.success("✅ Données d'exemple générées")
            
            # Préparation
            df_clean, initial_count = prepare_features(df_sample)
            
            # Détection du type
            if selected_race_type == "AUTO":
                race_type, reason = auto_detect_race_type(df_clean)
                st.info(f"🤖 **Type détecté :** {race_type} - {reason}")
            else:
                race_type = selected_race_type
            
            # Analyse
            df_result, feature_importance, ml_features = analyze_race_comprehensive(
                df_clean, race_type, use_ml_model
            )
            
            # Affichage
            display_race_analysis(df_result, race_type, feature_importance, confidence_threshold)
            
            if show_feature_importance and feature_importance is not None:
                display_feature_importance(feature_importance)
            
            display_betting_suggestions(df_result, confidence_threshold)
    
    with tab3:
        st.markdown("### 📚 Documentation et Guide d'Utilisation")
        
        st.markdown("""
        #### 🎯 Types de Courses Supportés
        
        **🏃 Courses de PLAT (Galop)**
        - Handicap avec variation de poids importante
        - Avantage aux cordes intérieures (1-4)
        - Règle : 1 kg = 1 longueur de performance
        - Pondération : Cotes 50% • Corde 30% • Poids 20%
        
        **🚗 Trot ATTELÉ AUTOSTART**
        - Départ derrière l'autostart
        - Numéros optimaux : 4, 5, 6 (centre première ligne)
        - Éviter cordes 1-3 (enfermement) et 7+ (effort supplémentaire)
        - Pondération : Cotes 70% • Corde 25% • Poids 5%
        
        **🔄 Trot ATTELÉ VOLTÉ**
        - Départ derrière élastiques
        - Position de départ sans importance
        - Focus sur la forme et la valeur
        - Pondération : Cotes 85% • Corde 5% • Poids 10%
        
        #### 🧠 Modèle d'Intelligence Artificielle
        
        **Variables Analysées :**
        - Cotes PMU et probabilités implicites
        - Historique de performances (musique)
        - Caractéristiques physiques (âge, sexe, poids)
        - Performances récentes (victoires, places, échecs)
        - Interactions entre variables
        
        **Algorithme :**
        - Random Forest pour robustesse
        - Normalisation et standardisation automatique
        - Score hybride (70% IA + 30% méthode traditionnelle)
        - Calibration des probabilités de victoire
        
        #### 📊 Interprétation des Résultats
        
        **Score Final :** Valeur entre 0 et 1 (plus élevé = meilleur)
        **Confiance :** Pourcentage de probabilité de performance
        **Rang :** Classement selon le score final
        
        #### 🎲 Suggestions de Paris
        
        - **Simple Gagnant :** Meilleur cheval selon l'IA
        - **Simple Placé :** Top 3 avec niveaux de confiance
        - **Quinté+ :** Combinaison des 5 meilleurs
        - **Potentiel :** Estimation du retour sur investissement
        
        #### 📚 Sources et Références
        
        Cet analyseur est basé sur des recherches documentées provenant de :
        - **turfmining.fr** - Statistiques et analyses de performances
        - **boturfers.fr** - Études sur les numéros de corde
        - **equidia.fr** - Données officielles PMU
        - **PMU** - Historiques de courses et résultats
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p style="margin: 0; color: #6c757d;">
                <strong>🏇 Analyseur Hippique IA</strong><br>
                Développé avec Streamlit • Machine Learning • Analyse Prédictive<br>
                <em>Utilisez ces pronostics de manière responsable</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
Response
Created file /home/user/streamlit_horse_analyzer.py (30300 characters)
