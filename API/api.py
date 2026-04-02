# api.py — MLPVehiclePrice API + Frontend Server

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import csv
import io
import os

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- Schemas de requete ---
class FullPredictionRequest(BaseModel):
    Kilometrage: float
    Annee_Facture: float
    Annee_veh: float
    Marque_veh: str
    Modele_veh: str
    Type_Energie: str
    Carburant: str


class BatchPredictionRequest(BaseModel):
    vehicles: List[FullPredictionRequest]


# --- Creation de l'app ---
app = FastAPI(
    title="API Prix Vehicule - Modele MLP",
    description="Estimation du prix de reprise d'un vehicule via un reseau de neurones MLP",
    version="1.0.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du modele ---
try:
    with open("vehicle_price_model.pkl", "rb") as f:
        model_package = pickle.load(f)

    best_model        = model_package["model"]
    scaler_X          = model_package["scaler_X"]
    scaler_y          = model_package["scaler_y"]
    power_y           = model_package["power_y"]
    encoder_marque    = model_package["encoder_marque"]
    encoder_modele    = model_package["encoder_modele"]
    encoder_energie   = model_package["encoder_energie"]
    encoder_carburant = model_package["encoder_carburant"]
    model_std         = model_package["model_std"]

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement de vehicle_price_model.pkl : {e}")


def _parse_year(val: str) -> float:
    val = val.strip()
    if not val:
        raise ValueError("Année vide ou manquante")
    # Format ISO : "2023-11-14T00:00:00" → on prend les 4 premiers caractères
    if len(val) >= 4 and (val[4] == '-' or 'T' in val):
        return float(val[:4])
    return float(val)


# --- Fonction de prediction unitaire ---
def _predict_single(req: FullPredictionRequest) -> dict:
    current_year = 2026

    # Division par zéro si Annee_veh == 2026
    age_veh = max(current_year - req.Annee_veh, 1)

    not_used      = float(np.clip(req.Annee_Facture - req.Annee_veh, 0, 100))
    cube_not_used = not_used ** 3

    inv_km     = 1.0 / req.Kilometrage if req.Kilometrage > 0 else 0.0
    log_inv_km = float(np.log1p(inv_km))

    km_per_year = req.Kilometrage / age_veh

    # Vérifie NaN/inf sur chaque feature avant d'appeler le modèle
    feats_check = {
        "not_used":     not_used,
        "log_inv_km":   log_inv_km,
        "km_per_year":  km_per_year,
        "Annee_Facture": req.Annee_Facture,
    }
    for name, val in feats_check.items():
        if not np.isfinite(float(val)):
            raise ValueError(f"Feature '{name}' invalide ({val}) — vérifiez les données d'entrée.")

    num_features = np.array([[not_used, log_inv_km, km_per_year, req.Annee_Facture, cube_not_used]])

    marque_enc    = encoder_marque.transform([[req.Marque_veh]])
    modele_enc    = encoder_modele.transform([[req.Modele_veh]])
    energie_enc   = encoder_energie.transform([[req.Type_Energie]])
    carburant_enc = encoder_carburant.transform([[req.Carburant]])

    X_full   = np.hstack([num_features, marque_enc, modele_enc, energie_enc, carburant_enc])
    X_scaled = scaler_X.transform(X_full)

    y_pred_scaled = best_model.predict(X_scaled).reshape(-1, 1)
    y_pred_pt     = scaler_y.inverse_transform(y_pred_scaled)
    y_pred_log    = power_y.inverse_transform(y_pred_pt)
    y_pred        = float(np.expm1(y_pred_log)[0, 0])

    # y_pred = nan retourné sans exception → crash JSON
    if not np.isfinite(y_pred):
        raise ValueError(
            f"Prédiction non finie ({y_pred}) — données probablement hors distribution."
        )

    # model_std = nan → IC nan
    std = float(model_std) if np.isfinite(float(model_std)) else 0.0

    return {
        "predicted_price": round(y_pred, 2),
        "ic_low":          round(y_pred - 1.96 * std, 2),
        "ic_high":         round(y_pred + 1.96 * std, 2),
    }


# --- Endpoint de prediction unitaire ---
@app.post("/predict/full")
def predict_full(request: FullPredictionRequest):
    try:
        result = _predict_single(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {e}")

    return {
        "predicted_price": result["predicted_price"],
        "ic_0.95": [result["ic_low"], result["ic_high"]],
    }


# --- Endpoint de prediction batch (JSON) ---
@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    results = []
    for i, vehicle in enumerate(request.vehicles):
        try:
            pred = _predict_single(vehicle)
            results.append({
                "index": i,
                "input": vehicle.model_dump(),
                "predicted_price": pred["predicted_price"],
                "ic_low":  pred["ic_low"],
                "ic_high": pred["ic_high"],
                "error": None,
            })
        except Exception as e:
            results.append({
                "index": i,
                "input": vehicle.model_dump(),
                "predicted_price": None,
                "ic_low":  None,
                "ic_high": None,
                "error": str(e),
            })
    return {"results": results, "total": len(results)}


# --- Endpoint upload CSV -> retourne CSV avec predictions ---
@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    REQUIRED = {"Kilometrage", "Annee_Facture", "Annee_veh", "Marque_veh", "Modele_veh", "Type_Energie", "Carburant"}

    try:
        content = await file.read()

        # Decode("utf-8-sig") gère le BOM Excel + fallback latin-1
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        # Délimiteur hardcodé ',' remplacé par auto-détection ';' vs ','
        first_line = text.split('\n')[0]
        delimiter = ';' if first_line.count(';') >= first_line.count(',') else ','

        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

        if reader.fieldnames is None:
            raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

        # Nettoyage des noms de colonnes (BOM résiduel, espaces)
        reader.fieldnames = [f.strip().lstrip('\ufeff') for f in reader.fieldnames]

        missing = REQUIRED - set(reader.fieldnames)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes dans le CSV : {', '.join(sorted(missing))}. "
                       f"Colonnes attendues : {', '.join(sorted(REQUIRED))}",
            )

        output_rows = []
        for row_idx, row in enumerate(reader):
            try:
                req = FullPredictionRequest(
                    Kilometrage=float(row["Kilometrage"]),
                    Annee_Facture=_parse_year(row["Annee_Facture"]),
                    Annee_veh=_parse_year(row["Annee_veh"]),
                    Marque_veh=row["Marque_veh"].strip(),
                    Modele_veh=row["Modele_veh"].strip(),
                    Type_Energie=row["Type_Energie"].strip(),
                    Carburant=row["Carburant"].strip(),
                )
                pred = _predict_single(req)
                output_rows.append({
                    **row,
                    "Prix_Predit": f"{pred['predicted_price']:.2f}",
                    "IC_95_Bas":   f"{pred['ic_low']:.2f}",
                    "IC_95_Haut":  f"{pred['ic_high']:.2f}",
                    "Erreur": "",
                })
            except Exception as e:
                output_rows.append({
                    **row,
                    "Prix_Predit": "",
                    "IC_95_Bas":   "",
                    "IC_95_Haut":  "",
                    "Erreur": str(e),
                })

        if not output_rows:
            raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne de donnees.")

        output = io.StringIO()
        fieldnames = list(output_rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8-sig")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions_vehicules.csv"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du CSV : {e}")


# --- Endpoint d'info modele ---
@app.get("/model/info")
def model_info():
    return {
        "model_type": type(best_model).__name__,
        "features_numeriques":   ["not_used", "log_inv_km", "km_per_year", "Annee_Facture", "cube_not_used"],
        "features_categoriques": ["Marque_veh", "Modele_veh", "Type_Energie", "Carburant"],
        "marques_disponibles":   list(encoder_marque.categories_[0]),
        "energies_disponibles":  list(encoder_energie.categories_[0]),
        "carburants_disponibles": list(encoder_carburant.categories_[0]),
    }


# --- Servir le frontend ---
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(frontend_path):
    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
