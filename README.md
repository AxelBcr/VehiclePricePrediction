# MLPVehiclePrice

**Estimation du prix de reprise d'un véhicule d'occasion** via un réseau de neurones **Multi-Layer Perceptron** (scikit-learn), exposé par une API **FastAPI** et servi dans un container **Docker**.

> Projet UCO — B2 S4 Application Marketing  
> Auteur : Axel Bouchaud-Roche — Mars 2026

---

## Table des matières

1. [Présentation](#présentation)
2. [Architecture du projet](#architecture-du-projet)
3. [Stack technique](#stack-technique)
4. [Démarrage rapide avec Docker](#démarrage-rapide-avec-docker)
5. [Lancer le service API](#lancer-le-service-api)
6. [Utilisation de l'API](#utilisation-de-lapi)
7. [Import CSV — Prédiction par lot](#import-csv--prédiction-par-lot)
8. [Interface Web](#interface-web)
9. [Entraînement du modèle](#entraînement-du-modèle)
10. [Structure des fichiers](#structure-des-fichiers)

---

## Présentation

Ce projet propose une solution complète pour estimer le prix de reprise d'un véhicule d'occasion en France. Il couvre l'ensemble du pipeline :

- **Analyse exploratoire & feature engineering** sur un dataset de ventes automobiles françaises (2020–2026)
- **Entraînement d'un MLPRegressor** avec pipeline de preprocessing (log, Yeo-Johnson, StandardScaler, OneHotEncoder)
- **API REST** (FastAPI) pour interroger le modèle en temps réel (unitaire ou par lot via CSV)
- **Interface web** embarquée pour une utilisation sans code, avec import/export CSV
- **Containerisation Docker** pour un déploiement reproductible

### Caractéristiques du modèle

| Élément | Détail |
|---|---|
| Algorithme | `MLPRegressor` (scikit-learn) |
| Features numériques | Âge du véhicule, log inverse kilométrage, km/an, année facture, âge³ |
| Features catégorielles | Marque, modèle, type d'énergie, carburant |
| Preprocessing cible | `log1p` → `PowerTransformer` (Yeo-Johnson) → `StandardScaler` |
| Sortie | Prix estimé (€) + intervalle de confiance à 95% |

---

## Architecture du projet

```
MLPVehiclePrice/
├── API/
│   ├── api.py                  # API FastAPI
│   ├── vehicle_price_model.pkl # Modèle sérialisé (pickle)
│   └── client.ipynb            # Notebook client de test
├── frontend/
│   └── index.html              # Interface web (HTML/CSS/JS)
├── modules/
│   ├── inverse_transform_pipeline.py
│   └── plot_pca.py
├── main.ipynb                  # Notebook d'entraînement du modèle
├── vente_vehicule_2026.csv     # Dataset source
├── Dockerfile                  # Image Docker
├── requirements.txt            # Dépendances Python (Docker)
├── .dockerignore
├── build_docker.bat            # Script de build + run (Windows)
├── build_docker.sh             # Script de build + run (Linux / VM GCE)
└── README.md
```

---

## Stack technique

- **Python 3.12** — Langage principal
- **scikit-learn 1.8** — MLPRegressor, preprocessing, évaluation
- **FastAPI** — Framework API REST asynchrone
- **Uvicorn** — Serveur ASGI
- **Docker** — Containerisation
- **HTML/CSS/JS** — Interface web embarquée (vanilla, sans framework)

---

## Démarrage rapide avec Docker

### Prérequis

- **Windows** : [Docker Desktop](https://www.docker.com/products/docker-desktop) (installé automatiquement par le script si absent)
- **Linux (VM GCE)** : Docker Engine (installé automatiquement par le script si absent)

### Commandes manuelles (build uniquement)

Si vous souhaitez construire l'image sans passer par les scripts :

```bash
cd MLPVehiclePrice
docker build -t mlp-vehicle-price-axelbcr:1.0.0 .
```

---

## Lancer le service API

Deux scripts automatisent l'intégralité du processus : vérification / installation de Docker, construction de l'image, et lancement du container. Choisissez celui correspondant à votre OS.

### Windows — `build_docker.bat`

Double-cliquez sur le fichier ou exécutez depuis un terminal :

```batch
cd C:\Users\Axel\PycharmProjects\MLPVehiclePrice
build_docker.bat
```

**Ce que fait le script :**
1. Vérifie si Docker Desktop est installé ; si non, le télécharge et l'installe automatiquement (via `winget` ou téléchargement direct)
2. Vérifie que le daemon Docker répond (sinon, tente d'ouvrir Docker Desktop)
3. Vérifie la présence des fichiers requis (`Dockerfile`, `API/vehicle_price_model.pkl`, `frontend/index.html`)
4. Arrête et supprime un éventuel container existant portant le même nom
5. Construit l'image `mlp-vehicle-price-axelbcr:1.0.0`
6. Propose de lancer le container ; si oui, le démarre sur le port `8001` et ouvre le navigateur

**Accès une fois le container lancé :**

| Ressource | URL |
|---|---|
| Interface web | http://localhost:8001 |
| Swagger UI (docs) | http://localhost:8001/docs |
| ReDoc | http://localhost:8001/redoc |
| Info modèle | http://localhost:8001/model/info |

### Linux / VM GCE — `build_docker.sh`

Transférez le projet sur la VM puis exécutez le script :

```bash
# Depuis votre PC Windows (transfert vers la VM GCE) :
gcloud compute scp --recurse MLPVehiclePrice/ NOM_VM:~/MLPVehiclePrice --zone=ZONE

# Sur la VM :
cd ~/MLPVehiclePrice
chmod +x build_docker.sh
./build_docker.sh
```

**Ce que fait le script :**
1. Vérifie si Docker Engine est installé ; si non, l'installe automatiquement via le dépôt officiel Docker (`apt`)
2. Active le service Docker et ajoute l'utilisateur courant au groupe `docker`
3. Vérifie la présence des fichiers requis
4. Arrête et supprime un éventuel container existant
5. Construit l'image `mlp-vehicle-price-axelbcr:1.0.0`
6. Propose de lancer le container ; si oui, détecte l'**IP externe** de la VM (metadata GCE) et affiche les liens d'accès

**Accès une fois le container lancé :**

| Ressource | URL |
|---|---|
| Interface web | http://IP_EXTERNE_VM:8001 |
| Swagger UI (docs) | http://IP_EXTERNE_VM:8001/docs |
| API Endpoint | POST http://IP_EXTERNE_VM:8001/predict/full |

> **Important** : vérifiez que le port `8001` est ouvert dans le firewall GCE :
> ```bash
> gcloud compute firewall-rules create allow-api-8001 \
>     --allow tcp:8001 \
>     --source-ranges 0.0.0.0/0 \
>     --description "Ouvrir port 8001 pour MLPVehiclePrice API"
> ```

### Arrêter et supprimer le container

```bash
docker stop mlp-vehicle-price-axelbcr
docker rm mlp-vehicle-price-axelbcr

# Supprimer l'image (optionnel) :
docker rmi mlp-vehicle-price-axelbcr:1.0.0
```

---

## Utilisation de l'API

### `POST /predict/full` — Prédiction unitaire

Estime le prix de reprise d'un seul véhicule.

**Requête (JSON) :**

```json
{
    "Kilometrage": 52335.0,
    "Annee_Facture": 2024,
    "Annee_veh": 2020,
    "Marque_veh": "CITROEN",
    "Modele_veh": "C5 AIRCROSS  (1CCE-VP) - VP",
    "Type_Energie": "Thermique",
    "Carburant": "Diesel"
}
```

**Réponse :**

```json
{
    "predicted_price": 18542.37,
    "ic_0.95": [13686.85, 23397.89]
}
```

### `POST /predict/batch` — Prédiction par lot (JSON)

Estime le prix de plusieurs véhicules en un seul appel.

**Requête :**

```json
{
    "vehicles": [
        {
            "Kilometrage": 52335,
            "Annee_Facture": 2024,
            "Annee_veh": 2020,
            "Marque_veh": "CITROEN",
            "Modele_veh": "C5 AIRCROSS",
            "Type_Energie": "Thermique",
            "Carburant": "Diesel"
        },
        {
            "Kilometrage": 9932,
            "Annee_Facture": 2020,
            "Annee_veh": 2017,
            "Marque_veh": "PEUGEOT",
            "Modele_veh": "108 I Ph1",
            "Type_Energie": "Thermique",
            "Carburant": "Essence"
        }
    ]
}
```

**Réponse :**

```json
{
    "results": [
        {
            "index": 0,
            "input": { "..." },
            "predicted_price": 18542.37,
            "ic_low": 13686.85,
            "ic_high": 23397.89,
            "error": null
        },
        {
            "index": 1,
            "input": { "..." },
            "predicted_price": 10757.39,
            "ic_low": 5901.87,
            "ic_high": 15612.91,
            "error": null
        }
    ],
    "total": 2
}
```

### `POST /predict/csv` — Prédiction par lot (upload CSV)

Upload un fichier CSV, retourne un CSV enrichi avec les prédictions. Voir la section [Import CSV](#import-csv--prédiction-par-lot) pour le format attendu.

```bash
curl -X POST http://localhost:8001/predict/csv \
  -F "file=@mes_vehicules.csv" \
  -o predictions_vehicules.csv
```

### `GET /model/info`

Retourne les métadonnées du modèle : type, features, et valeurs acceptées pour les variables catégorielles (marques, énergies, carburants).

### Exemple Python

```python
import requests

response = requests.post("http://localhost:8001/predict/full", json={
    "Kilometrage": 9932.0,
    "Annee_Facture": 2020,
    "Annee_veh": 2017,
    "Marque_veh": "PEUGEOT",
    "Modele_veh": "108 I Ph1",
    "Type_Energie": "Thermique",
    "Carburant": "Essence"
})

print(response.json())
# {'predicted_price': 10757.39, 'ic_0.95': [5901.87, 15612.91]}
```

---

## Import CSV — Prédiction par lot

L'interface web et l'endpoint `POST /predict/csv` permettent d'estimer le prix de reprise de plusieurs véhicules à partir d'un fichier CSV.

### Format du CSV d'entrée

Le fichier doit être au format CSV avec **séparateur virgule** (`,`) et encodage **UTF-8**. Les 7 colonnes suivantes sont **obligatoires** :

| Colonne | Type | Description | Exemple |
|---|---|---|---|
| `Kilometrage` | Nombre | Kilométrage exact du véhicule (valeur précise, pas d'arrondi) | `52335` |
| `Annee_Facture` | Entier | Année de la facture de reprise | `2024` |
| `Annee_veh` | Entier | Année de première immatriculation du véhicule | `2020` |
| `Marque_veh` | Texte | Marque du véhicule (doit correspondre aux marques connues du modèle) | `CITROEN` |
| `Modele_veh` | Texte | Modèle du véhicule | `C5 AIRCROSS` |
| `Type_Energie` | Texte | Type d'énergie : `Thermique`, `Hybride`, `Electrique` | `Thermique` |
| `Carburant` | Texte | Carburant : `Essence`, `Diesel`, `Electrique`, etc. | `Diesel` |

**Exemple de fichier CSV :**

```csv
Kilometrage,Annee_Facture,Annee_veh,Marque_veh,Modele_veh,Type_Energie,Carburant
52335,2024,2020,CITROEN,C5 AIRCROSS,Thermique,Diesel
9932,2020,2017,PEUGEOT,108 I Ph1,Thermique,Essence
120000,2023,2018,RENAULT,CLIO V,Thermique,Essence
45000,2025,2022,VOLKSWAGEN,GOLF VIII,Thermique,Essence
8500,2024,2023,RENAULT,ZOE,Electrique,Electrique
```

**Remarques :**
- L'ordre des colonnes n'a pas d'importance, seuls les noms comptent.
- Les colonnes supplémentaires sont conservées dans le fichier de sortie mais ignorées par le modèle.
- Les marques et carburants disponibles sont consultables via l'endpoint `GET /model/info`.

### Format du CSV de sortie

Le fichier retourné reprend toutes les colonnes du fichier d'entrée et ajoute 4 colonnes :

| Colonne ajoutée | Description |
|---|---|
| `Prix_Predit` | Prix de reprise estimé en euros |
| `IC_95_Bas` | Borne basse de l'intervalle de confiance à 95% |
| `IC_95_Haut` | Borne haute de l'intervalle de confiance à 95% |
| `Erreur` | Message d'erreur si la prédiction a échoué pour cette ligne (vide sinon) |

### Utilisation via l'interface web

1. Ouvrez l'interface web (`http://localhost:8001`)
2. Faites défiler jusqu'à la section **Import CSV — Prédiction par lot**
3. Glissez-déposez votre fichier CSV ou cliquez pour le sélectionner
4. Les résultats s'affichent dans un tableau avec les prédictions
5. Cliquez sur **Exporter les résultats (.csv)** pour télécharger le fichier enrichi

### Utilisation via cURL

```bash
curl -X POST http://localhost:8001/predict/csv \
  -F "file=@mes_vehicules.csv" \
  -o predictions_vehicules.csv
```

### Utilisation via Python

```python
import requests

with open("mes_vehicules.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8001/predict/csv",
        files={"file": ("mes_vehicules.csv", f, "text/csv")}
    )

with open("predictions_vehicules.csv", "wb") as out:
    out.write(response.content)

print("Export terminé : predictions_vehicules.csv")
```

---

## Interface Web

L'interface est accessible directement à la racine du serveur (`http://localhost:8001` ou `http://IP_VM:8001`). Elle est embarquée dans le container Docker et ne nécessite aucune installation supplémentaire.

Fonctionnalités :
- Formulaire de saisie avec les champs du modèle (kilométrage précis, sans arrondi)
- Listes déroulantes alimentées dynamiquement depuis `/model/info`
- Affichage du prix estimé et de l'intervalle de confiance à 95%
- **Import CSV** par glisser-déposer avec aperçu des résultats en tableau
- **Export CSV** des prédictions en un clic
- Design dark theme, responsive (mobile / desktop)
- Liens directs vers Swagger UI, ReDoc et l'endpoint d'info modèle

---

## Entraînement du modèle

Le notebook `main.ipynb` contient le pipeline complet :

1. **Chargement** du CSV `vente_vehicule_2026.csv`
2. **Nettoyage** — filtrage des valeurs aberrantes (kilométrage, prix)
3. **Feature engineering** — variables dérivées (âge, km/an, log inverse km, âge³)
4. **Preprocessing** — `log1p` → `PowerTransformer` (Yeo-Johnson) → `StandardScaler` pour la cible ; `OneHotEncoder` pour les catégorielles
5. **Entraînement** du `MLPRegressor` avec recherche d'hyperparamètres
6. **Évaluation** — MSE, R², résidus
7. **Export** du package complet dans `vehicle_price_model.pkl`

Pour ré-entraîner le modèle, exécutez l'intégralité du notebook puis relancez le build Docker.

---

## Structure des fichiers

| Fichier | Description |
|---|---|
| `Dockerfile` | Définition de l'image Docker (Python 3.12-slim) |
| `requirements.txt` | Dépendances Python pour le container |
| `.dockerignore` | Exclusions du contexte Docker |
| `build_docker.bat` | Script Windows : installe Docker + build + run |
| `build_docker.sh` | Script Linux/GCE : installe Docker + build + run |
| `API/api.py` | Code source de l'API FastAPI (unitaire, batch, CSV) |
| `API/vehicle_price_model.pkl` | Modèle MLP + encodeurs + scalers (pickle) |
| `frontend/index.html` | Interface web single-page avec import/export CSV |
| `main.ipynb` | Notebook d'entraînement et d'analyse |
| `vente_vehicule_2026.csv` | Dataset de ventes automobiles |
