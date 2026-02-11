import numpy as np


def inverse_transform_pipeline(transformed_pred, scaler, power_transformer):
    """
    Applique l'inverse transform complet : scaler -> power transformer -> expm1.

    Args:
        transformed_pred (np.ndarray): Données transformées/scalées (1D ou 2D).
        scaler: Scaler fitté (ex: StandardScaler).
        power_transformer: PowerTransformer fitté.

    Returns:
        np.ndarray: Données originales (1D), avec NaN pour valeurs non finies.
    """
    # Étape 1: Reshape 2D et inverse scaler
    transformed_2d = transformed_pred.reshape(-1, 1)
    unscaled_data = scaler.inverse_transform(transformed_2d).ravel()
    unscaled_data = np.where(np.isfinite(unscaled_data), unscaled_data, np.nan)

    # Étape 2: Inverse power transformer
    pt_inverse_data = power_transformer.inverse_transform(unscaled_data.reshape(-1, 1)).ravel()
    pt_inverse_data = np.where(np.isfinite(pt_inverse_data), pt_inverse_data, np.nan)

    # Étape 3: Inverse exp (log -> valeur originale)
    original_data = np.expm1(pt_inverse_data)
    original_data = np.where(np.isfinite(original_data), original_data, np.nan)

    return original_data
