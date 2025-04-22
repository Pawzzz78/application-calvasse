# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

def entrainer_modele(fichier_csv="bald_probability.csv"):
    """
    Entraîne le modèle de prédiction de calvitie et le sauvegarde.
    """
    # Charger les données
    df = pd.read_csv(fichier_csv)
    
    # Nettoyer les données (supprimer les lignes avec des valeurs manquantes)
    df_clean = df.dropna()
    
    # Afficher des informations sur les données pour le débogage
    print(f"Nombre de lignes dans le jeu de données: {len(df_clean)}")
    print(f"Colonnes: {df_clean.columns.tolist()}")
    
    # Vérifier si la colonne cible existe
    if "bald_prob" not in df_clean.columns:
        raise ValueError("La colonne 'bald_prob' n'existe pas dans le fichier CSV")
    
    # Séparer les caractéristiques et la cible
    X = df_clean.drop("bald_prob", axis=1)
    y = df_clean["bald_prob"]
    
    # Identifier les caractéristiques catégorielles
    cat_features = [col for col in X.columns if X[col].dtype == "object"]
    num_features = [col for col in X.columns if X[col].dtype != "object"]
    
    print(f"Caractéristiques catégorielles: {cat_features}")
    print(f"Caractéristiques numériques: {num_features}")
    
    # Créer le préprocesseur pour gérer les variables catégorielles et numériques
    preprocesseur = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder="passthrough")
    
    # Créer le pipeline avec le préprocesseur et le modèle de régression linéaire
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocesseur),
        ("regressor", LinearRegression())
    ])
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    pipeline.fit(X_train, y_train)
    
    # Évaluer le modèle
    score = pipeline.score(X_test, y_test)
    print(f"Score R² du modèle: {score}")
    
    # Sauvegarder le modèle
    joblib.dump(pipeline, 'modele_calvitie.joblib')
    
    return pipeline

def predire_calvitie(age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
                   poids, taille, shampoing, est_fumeur, education, stress):
    """
    Prédit la probabilité de calvitie en fonction des caractéristiques de la personne.
    """
    # Charger le modèle ou l'entraîner s'il n'existe pas
    model_file = 'modele_calvitie.joblib'
    if os.path.exists(model_file):
        try:
            pipeline = joblib.load(model_file)
            print("Modèle chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            pipeline = entrainer_modele()
    else:
        print("Entraînement d'un nouveau modèle")
        pipeline = entrainer_modele()
    
    # Créer un DataFrame avec les caractéristiques de la personne
    donnees_personne = pd.DataFrame({
        'age': [age],
        'gender': [genre],
        'job_role': [role_professionnel],
        'province': [province],
        'salary': [salaire],
        'is_married': [est_marie],
        'is_hereditary': [est_hereditaire],
        'weight': [poids],
        'height': [taille],
        'shampoo': [shampoing],
        'is_smoker': [est_fumeur],
        'education': [education],
        'stress': [stress]
    })
    
    print(f"Données en entrée: {donnees_personne.to_dict()}")
    
    # Faire la prédiction
    try:
        probabilite = pipeline.predict(donnees_personne)[0]
        print(f"Prédiction brute: {probabilite}")
        
        # Limiter la valeur entre 0 et 1
        probabilite = max(0, min(1, probabilite))
        
        return probabilite
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        # En cas d'erreur, vérifier si les colonnes correspondent à celles attendues par le modèle
        if hasattr(pipeline, 'feature_names_in_'):
            print(f"Colonnes attendues par le modèle: {pipeline.feature_names_in_.tolist()}")
        raise e

def interpreter_probabilite(probabilite):
    """
    Interprète la probabilité de calvitie et fournit une explication.
    """
    if probabilite < 0.3:
        categorie = "Risque faible"
        explication = "Il y a peu de chances que vous développiez une calvitie significative."
        couleur = "green"
    elif probabilite < 0.6:
        categorie = "Risque modéré"
        explication = "Vous avez un risque modéré de développer une calvitie. Certaines mesures préventives pourraient être utiles."
        couleur = "orange"
    else:
        categorie = "Risque élevé"
        explication = "Vous avez un risque élevé de développer une calvitie. Il serait conseillé de consulter un dermatologue."
        couleur = "red"
    
    return {
        "probabilite": probabilite,
        "categorie": categorie,
        "explication": explication,
        "couleur": couleur
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset_model', methods=['GET'])
def reset_model():
    """Endpoint pour forcer la réinitialisation du modèle"""
    try:
        if os.path.exists('modele_calvitie.joblib'):
            os.remove('modele_calvitie.joblib')
        
        # Entraîner un nouveau modèle
        pipeline = entrainer_modele()
        
        return jsonify({
            "success": True,
            "message": "Le modèle a été réinitialisé avec succès."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Initialisation du modèle au démarrage
@app.before_first_request
def initialize_model():
    """Initialise le modèle au premier démarrage de l'application"""
    try:
        # Forcer la réinitialisation du modèle en le supprimant s'il existe
        if os.path.exists('modele_calvitie.joblib'):
            os.remove('modele_calvitie.joblib')
        
        # Entraîner un nouveau modèle
        entrainer_modele()
        print("Modèle initialisé avec succès au démarrage de l'application")
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = request.form
        
        # Convertir les données
        age = int(data.get('age'))
        genre = data.get('genre')
        role_professionnel = data.get('role_professionnel')
        province = data.get('province')
        salaire = float(data.get('salaire'))
        est_marie = int(data.get('est_marie'))
        est_hereditaire = int(data.get('est_hereditaire'))
        poids = float(data.get('poids'))
        taille = float(data.get('taille'))
        shampoing = data.get('shampoing')
        est_fumeur = int(data.get('est_fumeur'))
        education = data.get('education')
        stress = int(data.get('stress'))
        
        # Faire la prédiction
        probabilite = predire_calvitie(
            age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
            poids, taille, shampoing, est_fumeur, education, stress
        )
        
        # Interpréter le résultat
        resultat = interpreter_probabilite(probabilite)
        
        return render_template('result.html', 
                              probabilite=resultat["probabilite"]*100, 
                              categorie=resultat["categorie"], 
                              explication=resultat["explication"],
                              couleur=resultat["couleur"])
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Récupérer les données JSON
        data = request.json
        
        # Convertir les données
        age = int(data.get('age'))
        genre = data.get('genre')
        role_professionnel = data.get('role_professionnel')
        province = data.get('province')
        salaire = float(data.get('salaire'))
        est_marie = int(data.get('est_marie'))
        est_hereditaire = int(data.get('est_hereditaire'))
        poids = float(data.get('poids'))
        taille = float(data.get('taille'))
        shampoing = data.get('shampoing')
        est_fumeur = int(data.get('est_fumeur'))
        education = data.get('education')
        stress = int(data.get('stress'))
        
        # Faire la prédiction
        probabilite = predire_calvitie(
            age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
            poids, taille, shampoing, est_fumeur, education, stress
        )
        
        # Interpréter le résultat
        resultat = interpreter_probabilite(probabilite)
        
        return jsonify({
            "probabilite": resultat["probabilite"],
            "categorie": resultat["categorie"],
            "explication": resultat["explication"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
