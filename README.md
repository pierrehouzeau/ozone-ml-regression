# ozone-ml-regression

Projet ML de régression pour prédire la concentration d’ozone.

Ce dépôt ne contient que les fichiers indispensables pour visualiser et exécuter le notebook: le notebook principal et le jeu de données minimal. Les documents volumineux (PDF, exports, etc.) sont volontairement exclus pour garder le dépôt léger.

## Contenu minimal
- `Projet_Machine_Learning.ipynb` — Notebook principal (EDA, nettoyage, modèles k-NN/Lasso/Ridge/Arbres/Ensembles, optimisation, évaluation RMSE)
- `ozone.csv` — Jeu de données utilisé par le notebook
- `requirements.txt` — Dépendances Python pour exécuter le notebook

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

## Notes
- Les fichiers PDF et autres artefacts non essentiels ne sont pas suivis par Git (`.gitignore`).
- Si vous avez besoin des supports/rapports, ajoutez-les localement sans les committer, ou hébergez-les ailleurs et liez-les ici.
