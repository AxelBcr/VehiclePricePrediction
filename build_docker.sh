#!/bin/bash
# ============================================================
#  MLPVehiclePrice -- Build & Run Docker Image (Linux/GCE)
#  Auteur : Axel Bouchaud-Roche | UCO B2 S4
# ============================================================

set -e

IMAGE_NAME="mlp-vehicle-price-axelbcr"
VERSION="1.0.0"
PORT=8001

echo ""
echo "  =================================================="
echo "   MLPVehiclePrice -- Docker Build Script (Linux)"
echo "   Image : ${IMAGE_NAME}:${VERSION}"
echo "  =================================================="
echo ""

# --- Fonction : recuperer l'IP externe de la VM ---
get_external_ip() {
    # 1) Metadata GCE (fonctionne sur Google Cloud)
    local gce_ip
    gce_ip=$(curl -s -m 3 -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" 2>/dev/null) || true
    if [ -n "$gce_ip" ] && [ "$gce_ip" != "" ]; then
        echo "$gce_ip"
        return
    fi

    # 2) Metadata AWS (fonctionne sur EC2)
    local aws_ip
    aws_ip=$(curl -s -m 3 "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null) || true
    if [ -n "$aws_ip" ] && [[ "$aws_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "$aws_ip"
        return
    fi

    # 3) Service externe en dernier recours
    local ext_ip
    ext_ip=$(curl -s -m 5 ifconfig.me 2>/dev/null) || true
    if [ -n "$ext_ip" ] && [[ "$ext_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "$ext_ip"
        return
    fi

    # 4) Fallback IP interne
    hostname -I | awk '{print $1}'
}

# --- Verifier et installer Docker si necessaire ---
if ! command -v docker &> /dev/null; then
    echo "[INFO] Docker n'est pas installe. Installation en cours..."
    echo ""

    export DEBIAN_FRONTEND=noninteractive

    echo "[1/4] Mise a jour des paquets..."
    sudo apt-get update -y -qq

    echo "[2/4] Installation des prerequis..."
    sudo apt-get install -y -qq \
        ca-certificates \
        curl \
        gnupg \
        lsb-release > /dev/null

    echo "[3/4] Ajout du depot Docker officiel..."
    sudo install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
        curl -fsSL "https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg" \
            | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
    fi

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y -qq

    echo "[4/4] Installation de Docker Engine..."
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin > /dev/null

    sudo systemctl enable docker
    sudo systemctl start docker

    if ! groups "$USER" | grep -q docker; then
        sudo usermod -aG docker "$USER"
        echo ""
        echo "[INFO] Utilisateur '$USER' ajoute au groupe docker."
        echo "       Pour eviter sudo, deconnectez-vous puis reconnectez-vous,"
        echo "       ou executez : newgrp docker"
        echo ""
    fi

    echo "[OK] Docker installe avec succes !"
    docker --version
    echo ""
else
    echo "[OK] Docker detecte : $(docker --version)"
    echo ""
fi

# --- Verifier que le daemon Docker tourne ---
if ! sudo docker info &> /dev/null; then
    echo "[INFO] Demarrage du daemon Docker..."
    sudo systemctl start docker
    sleep 3
    if ! sudo docker info &> /dev/null; then
        echo "[ERREUR] Impossible de demarrer le daemon Docker."
        exit 1
    fi
fi

echo "[OK] Docker operationnel."
echo ""

# --- Detecter si on a besoin de sudo pour docker ---
DOCKER_CMD="docker"
if ! docker ps &> /dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
    echo "[INFO] Utilisation de sudo pour les commandes Docker."
    echo ""
fi

# --- Se placer dans le repertoire du script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Verifier les fichiers requis ---
if [ ! -f "Dockerfile" ]; then
    echo "[ERREUR] Dockerfile introuvable dans : $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "API/vehicle_price_model.pkl" ]; then
    echo "[ERREUR] Modele introuvable : API/vehicle_price_model.pkl"
    echo "         Entrainez le modele d'abord via main.ipynb."
    exit 1
fi

if [ ! -f "frontend/index.html" ]; then
    echo "[ERREUR] Frontend introuvable : frontend/index.html"
    exit 1
fi

echo "[OK] Fichiers requis presents."
echo ""

# --- Arreter et supprimer un container existant ---
EXISTING=$($DOCKER_CMD ps -aq --filter "name=${IMAGE_NAME}" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "[INFO] Arret et suppression du container existant..."
    $DOCKER_CMD stop "$EXISTING" > /dev/null 2>&1 || true
    $DOCKER_CMD rm "$EXISTING" > /dev/null 2>&1 || true
fi

# --- Build de l'image ---
echo "=================================================="
echo " CONSTRUCTION DE L'IMAGE DOCKER"
echo " ${IMAGE_NAME}:${VERSION}"
echo "=================================================="
echo ""

$DOCKER_CMD build -t "${IMAGE_NAME}:${VERSION}" -t "${IMAGE_NAME}:latest" .

echo ""
echo "[OK] Image construite avec succes !"
echo ""

# --- Afficher l'image ---
echo "=================================================="
echo " IMAGES DOCKER"
echo "=================================================="
$DOCKER_CMD images "$IMAGE_NAME"
echo ""

# --- Demander si on lance le container ---
read -rp "Lancer le container maintenant ? (O/N) : " RUN_CHOICE

if [[ "$RUN_CHOICE" =~ ^[OoYy]$ ]]; then
    echo ""
    echo "[INFO] Lancement du container sur le port ${PORT}..."
    echo ""

    $DOCKER_CMD run -d --name "$IMAGE_NAME" -p "${PORT}:${PORT}" "${IMAGE_NAME}:${VERSION}"

    # Recuperer l'IP externe
    echo "[INFO] Detection de l'IP externe..."
    VM_IP=$(get_external_ip)

    echo ""
    echo "  =================================================="
    echo "   Container lance avec succes !"
    echo ""
    echo "   Interface Web : http://${VM_IP}:${PORT}"
    echo "   API Docs      : http://${VM_IP}:${PORT}/docs"
    echo "   API Endpoint  : POST /predict/full"
    echo ""
    echo "   Arreter : $DOCKER_CMD stop ${IMAGE_NAME}"
    echo "  =================================================="
    echo ""
    echo "   Note : verifiez que le port ${PORT} est ouvert dans"
    echo "   le firewall GCE (VPC > Firewall > allow tcp:${PORT})"
    echo ""
fi

echo "Termine."
