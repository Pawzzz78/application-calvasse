# Calculateur de Probabilité de Calvitie

Une application web simple qui permet de calculer la probabilité qu'une personne développe une calvitie en fonction de diverses caractéristiques personnelles.

## Fonctionnalités

- Interface utilisateur intuitive avec formulaire de saisie
- Prédiction basée sur un modèle de régression linéaire
- Résultats visuels avec interprétation
- API JSON pour l'intégration avec d'autres services

## Installation locale

1. Clonez ce dépôt
2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```
3. Assurez-vous que le fichier `bald_probability.csv` est présent dans le répertoire racine
4. Lancez l'application :
   ```bash
   python app.py
   ```
5. Accédez à l'application dans votre navigateur à l'adresse `http://localhost:5000`

## Déploiement sur des plateformes gratuites

### Déploiement sur Render

1. Créez un compte sur [Render](https://render.com/)
2. Cliquez sur "New" puis "Web Service"
3. Connectez votre dépôt GitHub
4. Configurez le service :
   - Nom : `calculateur-calvitie` (ou autre nom de votre choix)
   - Environment : `Python 3`
   - Build Command : `pip install -r requirements.txt`
   - Start Command : `gunicorn app:app`
5. Cliquez sur "Create Web Service"

### Déploiement sur PythonAnywhere

1. Créez un compte sur [PythonAnywhere](https://www.pythonanywhere.com/)
2. Allez dans l'onglet "Web" et créez une nouvelle application web
3. Choisissez Flask et Python 3.9
4. Configurez le chemin vers votre application : `/home/yourusername/mysite/app.py`
5. Téléchargez les fichiers de votre application via l'onglet "Files"
6. Installez les dépendances :
   ```bash
   pip3 install --user -r requirements.txt
   ```
7. Redémarrez votre application

## Structure des fichiers

- `app.py` : Application Flask principale
- `bald_probability.csv` : Fichier de données pour entraîner le modèle
- `modele_calvitie.joblib` : Modèle entraîné (généré automatiquement)
- `templates/` : Dossier contenant les templates HTML
  - `index.html` : Page d'accueil avec formulaire
  - `result.html` : Page de résultat
  - `error.html` : Page d'erreur
- `requirements.txt` : Liste des dépendances

## API JSON

Pour utiliser l'API, envoyez une requête POST à `/api/predict` avec un payload JSON contenant les caractéristiques de la personne :

```json
{
  "age": 45,
  "genre": "male",
  "role_professionnel": "Employee",
  "province": "Paris",
  "salaire": 50000,
  "est_marie": 1,
  "est_hereditaire": 1,
  "poids": 75,
  "taille": 180,
  "shampoing": "Head & Shoulders",
  "est_fumeur": 0,
  "education": "Bachelor Degree",
  "stress": 6
}
```

La réponse sera au format JSON :

```json
{
  "probabilite": 0.62,
  "categorie": "Risque élevé",
  "explication": "Vous avez un risque élevé de développer une calvitie. Il serait conseillé de consulter un dermatologue."
}
```

## Licence

Ce projet est sous licence MIT.

## Avertissement

Cette application est fournie à titre informatif uniquement et ne constitue pas un avis médical professionnel. Les prédictions sont basées sur un modèle statistique simple et ne doivent pas être utilisées comme unique source d'information pour prendre des décisions médicales. 