# ozone-ml-regression

Projet ML de régression pour prédire la concentration d’ozone.

Ce dépôt ne contient que les fichiers indispensables pour visualiser et exécuter le notebook: le notebook principal et le jeu de données minimal. Les documents volumineux (PDF, exports, etc.) sont volontairement exclus pour garder le dépôt léger.

## Contenu minimal
- `Projet_Machine_Learning.ipynb` — Notebook principal (EDA, nettoyage, modèles k-NN/Lasso/Ridge/Arbres/Ensembles, optimisation, évaluation RMSE)
- `ozone.csv` — Jeu de données utilisé par le notebook
- `requirements.txt` — Dépendances Python pour exécuter le notebook

## Présentation du projet
- Objectif: prédire la variable continue `O3obs` (concentration d’ozone) en minimisant la métrique RMSE.
- Contexte: exploration et modélisation supervisée sur un jeu de données ozone.
- Sortie attendue: comparaison de plusieurs familles de modèles et sélection sur performance.

## Approche et méthodes
- EDA: aperçu des distributions, corrélations et valeurs aberrantes.
- Prétraitement: imputation (SimpleImputer/KNNImputer), encodage catégoriel (LabelEncoder), normalisation/standardisation, éventuellement transformation (skew/Box-Cox) et features (PolynomialFeatures) selon les essais du notebook.
- Modèles testés: KNN, régressions linéaires (OLS, Lasso, Ridge), arbres et forêts (DecisionTreeRegressor, RandomForestRegressor), XGBoost, GaussianProcess, stacking.
- Validation: `train_test_split`, cross-validation et `GridSearchCV` pour l’optimisation d’hyperparamètres.
- Évaluation: RMSE principalement (voir notebook pour les scores détaillés et les figures).

## Résultats
Les scores, graphiques et comparaisons entre modèles sont reportés directement dans `Projet_Machine_Learning.ipynb`. Reportez-vous aux dernières cellules du notebook pour la synthèse des performances et la conclusion.

## Prérequis
- Python 3.10+ recommandé
- `pip` et un environnement virtuel (ex: `venv` ou `conda`)

## Installation rapide
1) Créer et activer un environnement virtuel
   - `python -m venv .venv && source .venv/bin/activate` (Linux/macOS)
   - `py -m venv .venv && .venv\\Scripts\\activate` (Windows PowerShell)
2) Installer les dépendances
   - `pip install -r requirements.txt`
3) Lancer Jupyter et ouvrir le notebook
   - `jupyter lab` ou `jupyter notebook`
   - Ouvrir `Projet_Machine_Learning.ipynb`

## Données
Le notebook lit le fichier `ozone.csv` présent à la racine du projet. Aucune étape supplémentaire n’est requise.

## Structure du notebook
- Chargement des données et EDA
- Nettoyage, imputation et transformations
- Entraînement et comparaison des modèles
- Optimisation des hyperparamètres
- Évaluation finale (RMSE) et interprétation

## Notes
- Les fichiers PDF et autres artefacts non essentiels ne sont pas suivis par Git (`.gitignore`).
- Si vous avez besoin des supports/rapports, ajoutez-les localement sans les committer, ou hébergez-les ailleurs et liez-les ici.

## Licence
Ce projet est sous licence indiquée dans `LICENSE`.
