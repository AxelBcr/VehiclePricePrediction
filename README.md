# VehiclePricePrediction

**Estimation du prix de reprise d'un véhicule d'occasion** via un réseau de neurones **Multi-Layer Perceptron** (scikit-learn), exposé par une API **FastAPI** et servi dans un container **Docker**.

> Projet UCO, B2 S4 Application Marketing
> Auteur : Axel Bouchaud-Roche, mars 2026

---

## Quick Start

```bash
# 1. Cloner le projet
git clone https://github.com/AxelBcr/VehiclePricePrediction.git
cd VehiclePricePrediction

# 2. Construire l'image Docker
docker build -t mlp-vehicle-price:1.0.0 .

# 3. Lancer le container
docker run -d --name mlp-vehicle-price -p 8001:8001 mlp-vehicle-price:1.0.0
```

Ouvrez http://localhost:8001 dans votre navigateur. C'est prêt.

---

## Prérequis

| Outil | Windows | Linux (VM) |
|-------|---------|------------|
| **Git** | [git-scm.com](https://git-scm.com/download/win) | `sudo apt-get install -y git` |
| **Docker** | [Docker Desktop](https://www.docker.com/products/docker-desktop) | Voir [installation Docker Linux](#installer-docker-sur-linux-vm) ci-dessous |

> Note : Python n'est pas requis sur la machine hôte. Tout s'exécute dans le container Docker.
> Pour ré-entraîner le modèle via `main.ipynb`, il faut Python 3.12+ et les dépendances listées dans `requirements.txt`.

---

## Table des matières

1. [Quick Start](#quick-start)
2. [Prérequis](#prérequis)
3. [Installation Docker pas à pas](#installation-docker-pas-à-pas)
4. [Utilisation de l'API](#utilisation-de-lapi)
5. [Import CSV : prédiction par lot](#import-csv--prédiction-par-lot)
6. [Interface Web](#interface-web)
7. [Entraînement du modèle](#entraînement-du-modèle)
8. [Architecture du projet](#architecture-du-projet)
9. [Stack technique](#stack-technique)
10. [Troubleshooting](#troubleshooting)

---

## Installation Docker pas à pas

### Installer Docker sur Windows

1. Téléchargez **Docker Desktop** depuis [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) et lancez l'installeur.
2. Cochez **"Use WSL 2 instead of Hyper-V"** si proposé (recommandé).
3. Redémarrez votre PC si demandé.
4. Lancez **Docker Desktop** depuis le menu Démarrer. Attendez que l'icône affiche **"Docker Desktop is running"**.
5. Vérifiez dans un terminal :

```bash
docker --version
# Docker version 28.x.x, build ...
```

### Installer Docker sur Linux (VM)

Connectez-vous en SSH à votre VM puis exécutez :

```bash
# Mettre à jour et installer les prérequis
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg

# Ajouter le dépôt Docker officiel
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installer Docker Engine
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# Activer et démarrer le service
sudo systemctl enable docker
sudo systemctl start docker

# (Optionnel) Utiliser Docker sans sudo
sudo usermod -aG docker $USER
newgrp docker

# Vérifier
docker --version
```

### Cloner et lancer le projet

```bash
git clone https://github.com/AxelBcr/VehiclePricePrediction.git
cd VehiclePricePrediction
docker build -t mlp-vehicle-price:1.0.0 .
docker run -d --name mlp-vehicle-price -p 8001:8001 mlp-vehicle-price:1.0.0
```

Vérifiez que le container tourne :

```bash
docker ps
# CONTAINER ID   IMAGE                       STATUS          PORTS
# a1b2c3d4e5f6   mlp-vehicle-price:1.0.0     Up 10 seconds   0.0.0.0:8001->8001/tcp
```

### Accéder à l'application

En local (Windows) :

| Ressource | URL |
|-----------|-----|
| Interface web | http://localhost:8001 |
| Swagger UI (docs interactives) | http://localhost:8001/docs |
| ReDoc (docs lecture) | http://localhost:8001/redoc |
| Infos modèle (JSON) | http://localhost:8001/model/info |

Sur une VM distante : remplacez `localhost` par l'IP de votre VM. Assurez-vous que le port 8001 est ouvert dans le pare-feu de la VM.

<details>
<summary>Exemple pour Google Cloud (GCE)</summary>

```bash
# Transférer le projet depuis Windows
gcloud compute scp --recurse ./VehiclePricePrediction NOM_VM:~/VehiclePricePrediction --zone=ZONE

# Ouvrir le port 8001
gcloud compute firewall-rules create allow-api-8001 \
    --allow tcp:8001 \
    --source-ranges 0.0.0.0/0 \
    --description "Ouvrir port 8001 pour l'API"
```

Accédez ensuite à `http://IP_EXTERNE_VM:8001`.

</details>

### Gestion du container

```bash
docker logs -f mlp-vehicle-price     # Logs en temps réel
docker stop mlp-vehicle-price        # Arrêter
docker start mlp-vehicle-price       # Redémarrer
docker rm mlp-vehicle-price          # Supprimer le container (après arrêt)
docker rmi mlp-vehicle-price:1.0.0   # Supprimer l'image
```

---

## Utilisation de l'API

### `POST /predict/full` (prédiction unitaire)

```json
// Requête
{
    "Kilometrage": 52335.0,
    "Annee_Facture": 2024,
    "Annee_veh": 2020,
    "Marque_veh": "CITROEN",
    "Modele_veh": "C5 AIRCROSS  (1CCE-VP) - VP",
    "Type_Energie": "Thermique",
    "Carburant": "Diesel"
}

// Réponse
{
    "predicted_price": 18542.37,
    "ic_0.95": [13686.85, 23397.89]
}
```

### `POST /predict/batch` (prédiction par lot JSON)

Envoie un tableau `"vehicles": [...]` contenant plusieurs véhicules. Retourne un objet `"results": [...]` avec `predicted_price`, `ic_low`, `ic_high` et `error` pour chaque véhicule.

```json
// Requête
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

// Réponse
{
    "results": [
        {
            "index": 0,
            "predicted_price": 18542.37,
            "ic_low": 13686.85,
            "ic_high": 23397.89,
            "error": null
        },
        {
            "index": 1,
            "predicted_price": 10757.39,
            "ic_low": 5901.87,
            "ic_high": 15612.91,
            "error": null
        }
    ],
    "total": 2
}
```

### `POST /predict/csv` (prédiction par lot, upload CSV)

Upload un fichier CSV, retourne un CSV enrichi avec les colonnes de prédiction. Voir la section [Import CSV](#import-csv--prédiction-par-lot).

```bash
curl -X POST http://localhost:8001/predict/csv \
  -F "file=@mes_vehicules.csv" \
  -o predictions_vehicules.csv
```

### `GET /model/info`

Retourne les métadonnées du modèle : type, features, et valeurs acceptées pour les variables catégorielles (marques, énergies, carburants).

### Exemples d'appel

#### /predict/full (unitaire)

<details>
<summary>Python</summary>

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

</details>

<details>
<summary>JavaScript</summary>

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

</details>

<details>
<summary>cURL</summary>

```bash
curl -X POST http://localhost:8001/predict/full \
  -H "Content-Type: application/json" \
  -d '{"Kilometrage":52335,"Annee_Facture":2024,"Annee_veh":2020,"Marque_veh":"CITROEN","Modele_veh":"C5 AIRCROSS","Type_Energie":"Thermique","Carburant":"Diesel"}'
```

</details>

#### /predict/batch (lot JSON)

<details>
<summary>Python</summary>

```python
import requests

vehicles = {
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

response = requests.post("http://localhost:8001/predict/batch", json=vehicles)
data = response.json()

for r in data["results"]:
    print(f"Véhicule {r['index']} : {r['predicted_price']:.2f} € [{r['ic_low']:.2f} - {r['ic_high']:.2f}]")
```

</details>

<details>
<summary>cURL</summary>

```bash
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"vehicles":[{"Kilometrage":52335,"Annee_Facture":2024,"Annee_veh":2020,"Marque_veh":"CITROEN","Modele_veh":"C5 AIRCROSS","Type_Energie":"Thermique","Carburant":"Diesel"},{"Kilometrage":9932,"Annee_Facture":2020,"Annee_veh":2017,"Marque_veh":"PEUGEOT","Modele_veh":"108 I Ph1","Type_Energie":"Thermique","Carburant":"Essence"}]}'
```

</details>

#### /predict/csv (lot CSV)

<details>
<summary>Python</summary>

```python
import requests

with open("mes_vehicules.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8001/predict/csv",
        files={"file": ("mes_vehicules.csv", f, "text/csv")}
    )

with open("predictions.csv", "wb") as out:
    out.write(response.content)

print("Export terminé : predictions.csv")
```

</details>

<details>
<summary>cURL</summary>

```bash
curl -X POST http://localhost:8001/predict/csv \
  -F "file=@mes_vehicules.csv" \
  -o predictions.csv
```

</details>

---

## Import CSV : prédiction par lot

### Format du CSV d'entrée

Séparateur virgule (`,`), encodage UTF-8. Les 7 colonnes suivantes sont obligatoires (l'ordre n'a pas d'importance) :

| Colonne | Type | Exemple |
|---------|------|---------|
| `Kilometrage` | Nombre | `52335` |
| `Annee_Facture` | Entier | `2024` |
| `Annee_veh` | Entier | `2020` |
| `Marque_veh` | Texte | `CITROEN` |
| `Modele_veh` | Texte | `C5 AIRCROSS` |
| `Type_Energie` | Texte | `Thermique` / `Hybride` / `Electrique` |
| `Carburant` | Texte | `Diesel` / `Essence` / `Electrique` |

Les colonnes supplémentaires sont conservées dans le fichier de sortie mais ignorées par le modèle. Les valeurs acceptées pour les marques et carburants sont consultables via `GET /model/info`.

Exemple :

```csv
Kilometrage,Annee_Facture,Annee_veh,Marque_veh,Modele_veh,Type_Energie,Carburant
52335,2024,2020,CITROEN,C5 AIRCROSS,Thermique,Diesel
9932,2020,2017,PEUGEOT,108 I Ph1,Thermique,Essence
120000,2023,2018,RENAULT,CLIO V,Thermique,Essence
```

### Format du CSV de sortie

Le fichier retourné reprend toutes les colonnes d'entrée et ajoute : `Prix_Predit`, `IC_95_Bas`, `IC_95_Haut`, `Erreur`.

### Utilisation via l'interface web

1. Ouvrez `http://localhost:8001`
2. Descendez à la section **Import CSV**
3. Glissez-déposez votre fichier CSV ou cliquez pour le sélectionner
4. Les résultats s'affichent en tableau
5. Cliquez **Exporter les résultats (.csv)** pour télécharger

---

## Interface Web

L'interface est à la racine du serveur (`http://localhost:8001`). Le container l'inclut directement, rien à installer en plus.

Elle propose un formulaire de saisie avec des listes déroulantes alimentées par `/model/info`, l'affichage du prix estimé avec IC 95%, et l'import/export CSV par drag-and-drop. La doc API avec des extraits de code Python, JS et cURL est intégrée dans la page. Dark theme, responsive.

---

## Entraînement du modèle

Tout le pipeline d'entraînement est dans `main.ipynb`. Pour le lancer, il faut Python 3.12+ et les dépendances :

```bash
pip install -r requirements.txt
```

Pipeline : chargement du dataset → nettoyage (filtrage aberrants) → feature engineering (âge, km/an, log inverse km, âge³) → preprocessing (`log1p` → Yeo-Johnson → StandardScaler pour la cible, `OneHotEncoder` pour les catégorielles) → entraînement `MLPRegressor` → évaluation (MSE, R²) → export du modèle dans `API/vehicle_price_model.pkl`.

### Caractéristiques du modèle

| Élément | Détail |
|---------|--------|
| Algorithme | `MLPRegressor` (scikit-learn) |
| Features numériques | Âge du véhicule, log inverse kilométrage, km/an, année facture, âge³ |
| Features catégorielles | Marque, modèle, type d'énergie, carburant |
| Preprocessing cible | `log1p` → `PowerTransformer` (Yeo-Johnson) → `StandardScaler` |
| Sortie | Prix estimé (€) + intervalle de confiance à 95% |

Après ré-entraînement, relancez le build Docker pour intégrer le nouveau modèle.

---

## Architecture du projet

```
VehiclePricePrediction/
├── API/
│   ├── api.py                    # API FastAPI (unitaire, batch, CSV)
│   ├── vehicle_price_model.pkl   # Modèle MLP sérialisé + encodeurs + scalers
│   └── client.ipynb              # Notebook client de test de l'API
├── frontend/
│   └── index.html                # Interface web single-page (HTML/CSS/JS)
├── modules/
│   ├── inverse_transform_pipeline.py  # Inversion du pipeline de preprocessing
│   └── plot_pca.py                    # Visualisation PCA (analyse exploratoire)
├── main.ipynb                    # Notebook d'entraînement du modèle
├── Dockerfile                    # Image Docker (Python 3.12-slim)
├── requirements.txt              # Dépendances Python (Docker)
└── README.md
```

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.12 |
| ML | scikit-learn 1.8 (MLPRegressor, preprocessing, évaluation) |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Frontend | HTML/CSS/JS vanilla (sans framework) |

---

## Troubleshooting

**`docker: command not found`**
Docker n'est pas installé ou pas dans le PATH. Sur Windows, vérifiez que Docker Desktop est lancé. Sur Linux, vérifiez l'installation avec `sudo systemctl status docker`.

**`Cannot connect to the Docker daemon`**
Le service Docker n'est pas démarré. Sur Windows : lancez Docker Desktop. Sur Linux : `sudo systemctl start docker`.

**`Bind for 0.0.0.0:8001 failed: port is already allocated`**
Le port 8001 est déjà utilisé. Soit arrêtez le processus qui l'occupe, soit mappez sur un autre port :
```bash
docker run -d --name mlp-vehicle-price -p 9001:8001 mlp-vehicle-price:1.0.0
# → accessible sur http://localhost:9001
```

**`docker build` échoue sur une erreur réseau**
Vérifiez votre connexion internet. Si vous êtes derrière un proxy, configurez-le dans Docker Desktop (Settings → Resources → Proxies).

**Le container démarre mais l'interface ne répond pas**
Vérifiez les logs : `docker logs mlp-vehicle-price`. Attendez quelques secondes que Uvicorn démarre complètement.

**Sur VM distante, impossible d'accéder à l'application**
Assurez-vous que le port 8001 est ouvert dans le pare-feu de la VM (iptables, security group AWS, firewall GCE, etc.).
