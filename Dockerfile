FROM python:3.12-slim

LABEL maintainer="Axel Bouchaud-Roche"
LABEL description="API de prediction du prix de reprise vehicule (MLP) -- Projet UCO B2 S4"
LABEL version="1.0.0"
LABEL image.name="mlp-vehicle-price-axelbcr"

# Supprimer les warnings debconf en mode non-interactif
ENV DEBIAN_FRONTEND=noninteractive

# Dependances systeme
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dependances Python (supprime le warning pip root)
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copier le code de l'API et le modele
COPY API/api.py ./api.py
COPY API/vehicle_price_model.pkl ./vehicle_price_model.pkl

# Copier l'interface web
COPY frontend/ ./frontend/

# Port expose
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/docs')" || exit 1

# Lancement
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
