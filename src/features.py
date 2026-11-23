"""Calcul des variables du modèle, partagé entre l'entraînement et l'API."""
import numpy as np
import pandas as pd

VARIABLES_NUMERIQUES = ["Year", "log_pesticides", "avg_temp", "temp_sqr"]
VARIABLES_CATEGORIELLES = ["Crop", "Area"]
VARIABLES_MODELE = VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES


def creer_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées attendues par le modèle.

    Le df d'entrée doit contenir : Crop, Area, Year, pesticides_tonnes, avg_temp.
    On calcule log_pesticides et temp_sqr, puis on renvoie les colonnes
    dans l'ordre attendu par le modèle.
    """
    df = df.copy()
    df["log_pesticides"] = np.log1p(df["pesticides_tonnes"])
    df["temp_sqr"] = df["avg_temp"] ** 2
    return df[VARIABLES_MODELE]