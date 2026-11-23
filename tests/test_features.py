"""Tests du calcul des variables (src/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src.features import VARIABLES_MODELE, creer_variables


@pytest.fixture
def donnees_brutes():
    return pd.DataFrame({
        "Crop": ["Wheat", "Maize"],
        "Area": ["France", "France"],
        "Year": [2013, 2013],
        "pesticides_tonnes": [0.0, 60000.0],
        "avg_temp": [10.0, -2.0],
    })


def test_variables_modele_est_une_liste_simple():
    # Une liste de noms de colonnes, sans liste imbriquée
    assert all(isinstance(nom, str) for nom in VARIABLES_MODELE)


def test_colonnes_et_ordre(donnees_brutes):
    resultat = creer_variables(donnees_brutes)
    assert list(resultat.columns) == VARIABLES_MODELE


def test_log_pesticides(donnees_brutes):
    resultat = creer_variables(donnees_brutes)
    assert resultat["log_pesticides"].iloc[0] == 0.0  # log(1 + 0) = 0
    assert resultat["log_pesticides"].iloc[1] == pytest.approx(np.log1p(60000))


def test_temperature_au_carre(donnees_brutes):
    resultat = creer_variables(donnees_brutes)
    assert resultat["temp_sqr"].tolist() == [100.0, 4.0]  # le carré d'une valeur négative est positif


def test_ne_modifie_pas_les_donnees_d_entree(donnees_brutes):
    colonnes_avant = list(donnees_brutes.columns)
    creer_variables(donnees_brutes)
    assert list(donnees_brutes.columns) == colonnes_avant