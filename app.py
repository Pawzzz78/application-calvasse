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
    
    # Séparer les caractéristiques et la cible
    X = df_clean.drop("bald_prob", axis=1)
    y = df_clean["bald_prob"]
    
    # Identifier les caractéristiques catégorielles
    cat_features = ["gender", "job_role", "province", "shampoo", "education"]
    
    # Créer le préprocesseur pour gérer les variables catégorielles
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
        pipeline = joblib.load(model_file)
    else:
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
    
    # Faire la prédiction
    probabilite = pipeline.predict(donnees_personne)[0]
    
    # Limiter la valeur entre 0 et 1
    probabilite = max(0, min(1, probabilite))
    
    return probabilite

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