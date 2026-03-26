# MLPVehiclePrice

**Estimation du prix de reprise d'un véhicule d'occasion** via un réseau de neurones **Multi-Layer Perceptron** (scikit-learn), exposé par une API **FastAPI** et servi dans un container **Docker**.

> Projet UCO — B2 S4 Application Marketing  
> Auteur : Axel Bouchaud-Roche — Mars 2026

---

## Table des matières

1. [Présentation](#présentation)
2. [Architecture du projet](#architecture-du-projet)
3. [Stack technique](#stack-technique)
4. [Pipeline de déploiement Docker](#pipeline-de-déploiement-docker)
5. [Utilisation de l'API](#utilisation-de-lapi)
6. [Import CSV — Prédiction par lot](#import-csv--prédiction-par-lot)
7. [Interface Web](#interface-web)
8. [Entraînement du modèle](#entraînement-du-modèle)
9. [Structure des fichiers](#structure-des-fichiers)

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
│   ├── api.py                  # API FastAPI (unitaire, batch, CSV)
│   ├── vehicle_price_model.pkl # Modèle sérialisé (pickle)
│   └── client.ipynb            # Notebook client de test
├── frontend/
│   └── index.html              # Interface web (HTML/CSS/JS)
├── modules/
│   ├── inverse_transform_pipeline.py
│   └── plot_pca.py
├── main.ipynb                  # Notebook d'entraînement du modèle
├── vente_vehicule_2026.csv     # Dataset source
├── Dockerfile                  # Définition de l'image Docker
├── requirements.txt            # Dépendances Python (Docker)
├── .dockerignore               # Exclusions du contexte de build
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

## Pipeline de déploiement Docker

Ce guide décrit pas à pas comment installer Docker, construire l'image du projet et lancer le service API. Deux environnements sont couverts : Windows (local) et Linux (VM Cloud / GCE).

### Étape 1 — Installer Docker

#### Sur Windows

1. Téléchargez **Docker Desktop** depuis [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) et lancez l'installeur.
2. Suivez l'assistant d'installation. Cochez **"Use WSL 2 instead of Hyper-V"** si proposé (recommandé).
3. Redémarrez votre PC si l'installeur le demande.
4. Lancez **Docker Desktop** depuis le menu Démarrer. Attendez que l'icône dans la barre des tâches affiche **"Docker Desktop is running"** (icône verte / baleine stable).
5. Vérifiez l'installation dans un terminal :

```bash
docker --version
# Docker version 28.x.x, build ...
```

#### Sur Linux (VM GCE / Ubuntu / Debian)

Connectez-vous en SSH à votre VM puis exécutez :

```bash
# 1. Mettre à jour les paquets
sudo apt-get update -y

# 2. Installer les prérequis
sudo apt-get install -y ca-certificates curl gnupg

# 3. Ajouter la clé GPG et le dépôt Docker officiel
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Installer Docker Engine
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# 5. Activer et démarrer le service
sudo systemctl enable docker
sudo systemctl start docker

# 6. (Optionnel) Ajouter votre utilisateur au groupe docker pour éviter sudo
sudo usermod -aG docker $USER
newgrp docker

# 7. Vérifier
docker --version
```

### Étape 2 — Transférer le projet (VM uniquement)

Si vous déployez sur une VM distante, transférez le dossier du projet :

```bash
# Depuis votre PC Windows, via gcloud :
gcloud compute scp --recurse C:\Users\Axel\PycharmProjects\MLPVehiclePrice\ NOM_VM:~/MLPVehiclePrice --zone=ZONE

# Ou via scp classique :
scp -r MLPVehiclePrice/ utilisateur@IP_VM:~/MLPVehiclePrice
```

### Étape 3 — Construire l'image Docker

Placez-vous dans le répertoire du projet et lancez le build :

```bash
cd MLPVehiclePrice      # ou ~/MLPVehiclePrice sur la VM
docker build -t mlp-vehicle-price-axelbcr:1.0.0 .
```

Le Dockerfile effectue les opérations suivantes :
1. Part d'une image `python:3.12-slim` (légère, ~150 Mo)
2. Installe les dépendances système (`gcc`) puis les dépendances Python depuis `requirements.txt`
3. Copie le code de l'API (`api.py`), le modèle entraîné (`vehicle_price_model.pkl`) et l'interface web (`frontend/`)
4. Expose le port `8001` et configure le healthcheck
5. Lance Uvicorn au démarrage du container

Vérifiez que l'image est bien créée :

```bash
docker images mlp-vehicle-price-axelbcr
# REPOSITORY                    TAG       IMAGE ID       SIZE
# mlp-vehicle-price-axelbcr     1.0.0     abc123def456   ~800MB
```

### Étape 4 — Lancer le container

```bash
docker run -d --name mlp-vehicle-price-axelbcr -p 8001:8001 mlp-vehicle-price-axelbcr:1.0.0
```

Détail des options :
- `-d` : mode détaché (background)
- `--name` : nom du container pour le manipuler facilement
- `-p 8001:8001` : mappe le port 8001 de la machine hôte vers le port 8001 du container

Vérifiez que le container tourne :

```bash
docker ps
# CONTAINER ID   IMAGE                              STATUS          PORTS
# a1b2c3d4e5f6   mlp-vehicle-price-axelbcr:1.0.0    Up 10 seconds   0.0.0.0:8001->8001/tcp
```

### Étape 5 — Accéder à l'application

#### En local (Windows)

| Ressource | URL |
|---|---|
| Interface web | http://localhost:8001 |
| Swagger UI (docs interactive) | http://localhost:8001/docs |
| ReDoc (docs lecture) | http://localhost:8001/redoc |
| Infos modèle (JSON) | http://localhost:8001/model/info |

#### Sur une VM Cloud (GCE)

Remplacez `IP_EXTERNE` par l'IP externe de votre VM (visible dans la console GCE) :

| Ressource | URL |
|---|---|
| Interface web | http://IP_EXTERNE:8001 |
| Swagger UI | http://IP_EXTERNE:8001/docs |

> **Firewall GCE** : assurez-vous que le port 8001 est ouvert :
> ```bash
> gcloud compute firewall-rules create allow-api-8001 \
>     --allow tcp:8001 \
>     --source-ranges 0.0.0.0/0 \
>     --description "Ouvrir port 8001 pour MLPVehiclePrice API"
> ```

### Gestion du container

```bash
# Voir les logs en temps réel
docker logs -f mlp-vehicle-price-axelbcr

# Arrêter le container
docker stop mlp-vehicle-price-axelbcr

# Redémarrer le container
docker start mlp-vehicle-price-axelbcr

# Supprimer le container (après arrêt)
docker rm mlp-vehicle-price-axelbcr

# Supprimer l'image
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

### Exemples d'appel

#### Python (requests)

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

data = response.json()
print(f"Prix estimé : {data['predicted_price']:.2f} €")
print(f"IC 95% : [{data['ic_0.95'][0]:.2f}, {data['ic_0.95'][1]:.2f}]")
```

#### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8001/predict/full", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        Kilometrage: 52335,
        Annee_Facture: 2024,
        Annee_veh: 2020,
        Marque_veh: "CITROEN",
        Modele_veh: "C5 AIRCROSS",
        Type_Energie: "Thermique",
        Carburant: "Diesel"
    })
});

const data = await response.json();
console.log("Prix estimé :", data.predicted_price);
```

#### cURL

```bash
curl -X POST http://localhost:8001/predict/full \
  -H "Content-Type: application/json" \
  -d '{"Kilometrage":52335,"Annee_Facture":2024,"Annee_veh":2020,"Marque_veh":"CITROEN","Modele_veh":"C5 AIRCROSS","Type_Energie":"Thermique","Carburant":"Diesel"}'
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
- **Documentation API** intégrée avec extraits de code (Python, JavaScript, cURL)
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
| `API/api.py` | Code source de l'API FastAPI (unitaire, batch, CSV) |
| `API/vehicle_price_model.pkl` | Modèle MLP + encodeurs + scalers (pickle) |
| `frontend/index.html` | Interface web single-page avec import/export CSV |
| `main.ipynb` | Notebook d'entraînement et d'analyse |
| `vente_vehicule_2026.csv` | Dataset de ventes automobiles |
