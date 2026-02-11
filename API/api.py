# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- Schéma de requête ---
class FullPredictionRequest(BaseModel):
    Kilometrage: float
    Annee_Facture: float
    Annee_veh: float
    Marque_veh: str
    Modele_veh: str
    Type_Energie: str
    Carburant: str

# --- Création de l'app ---
app = FastAPI(title="API Prix Véhicule - Modèle Complet")

# --- Chargement du modèle complet ---
try:
    with open("vehicle_price_model.pkl", "rb") as f:
        model_package = pickle.load(f)

    best_model = model_package["model"]
    scaler_X = model_package["scaler_X"]
    scaler_y = model_package["scaler_y"]
    power_y = model_package["power_y"]
    encoder_marque = model_package["encoder_marque"]
    encoder_modele = model_package["encoder_modele"]
    encoder_energie = model_package["encoder_energie"]
    encoder_carburant = model_package["encoder_carburant"]
    model_std = model_package["model_std"]

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement de vehicle_price_model.pkl : {e}")

# --- Endpoint de prédiction ---
@app.post("/predict/full")
def predict_full(request: FullPredictionRequest):
    try:
        # 1) Features dérivées comme dans le notebook
        current_year = 2026

        not_used = np.clip(request.Annee_Facture - request.Annee_veh, 0, float('inf'))
        cube_not_used = not_used ** 3

        inv_km = 1 / request.Kilometrage if request.Kilometrage > 0 else 0.0
        log_inv_km = np.log1p(inv_km)

        km_per_year = request.Kilometrage / (current_year - request.Annee_veh)

        num_features = np.array([[not_used, log_inv_km, km_per_year, request.Annee_Facture, cube_not_used]])

        # 2) Encodage des variables catégorielles
        marque_enc = encoder_marque.transform([[request.Marque_veh]])
        modele_enc = encoder_modele.transform([[request.Modele_veh]])
        energie_enc = encoder_energie.transform([[request.Type_Energie]])
        carburant_enc = encoder_carburant.transform([[request.Carburant]])

        # 3) Concaténation
        X_full = np.hstack([num_features, marque_enc, modele_enc, energie_enc, carburant_enc])

        # 4) Scaling X
        X_scaled = scaler_X.transform(X_full)

        # 5) Prédiction dans l’espace transformé
        y_pred_scaled = best_model.predict(X_scaled).reshape(-1, 1)

        # 6) Inverse scaler y
        y_pred_pt = scaler_y.inverse_transform(y_pred_scaled)

        # 7) Inverse PowerTransformer (Yeo-Johnson)
        y_pred_log = power_y.inverse_transform(y_pred_pt)

        # 8) Revenir à l’échelle originale (log1p)
        y_pred = np.expm1(y_pred_log)[0, 0]
        y_pred = float(y_pred)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return {"predicted_price": y_pred, "ic_0.95": [y_pred - 1.96 * model_std, y_pred + 1.96 * model_std]}