"""Tests de la logique de recommandation (src/recommandation.py)."""
import pandas as pd
import pytest

from src.recommandation import calculer_resultats, classer_cultures, predire

RENDEMENTS_MOYENS = pd.Series({"Wheat": 3.0, "Maize": 4.0, "Yams": 10.0})


class ModeleFactice:
    """Remplace le vrai modèle : renvoie toujours 5 t/ha, sauf -1 pour le maïs."""

    def predict(self, X):
        return [-1.0 if culture == "Maize" else 5.0 for culture in X["Crop"]]


def test_predire_ramene_les_valeurs_negatives_a_zero():
    contexte = {"pays": "France", "annee": 2013, "temperature": 11.5, "pesticides_tonnes": 60000}
    rendements = predire(ModeleFactice(), contexte, ["Wheat", "Maize"])
    assert rendements == [5.0, 0.0]


def test_calcul_du_rendement_relatif_et_du_revenu():
    resultats = calculer_resultats(
        ["Wheat"], [6.0], RENDEMENTS_MOYENS, cultures_observees={"Wheat"}, prix={"Wheat": 200}
    )
    assert resultats[0]["rendement_relatif_pct"] == 200.0  # 6 / 3 * 100
    assert resultats[0]["revenu_estime_par_ha"] == 1200.0  # 6 * 200
    assert resultats[0]["culture_observee_dans_pays"] is True


def test_revenu_absent_sans_prix():
    resultats = calculer_resultats(["Wheat"], [6.0], RENDEMENTS_MOYENS, cultures_observees={"Wheat"})
    assert resultats[0]["revenu_estime_par_ha"] is None


def _resultat(culture, relatif, revenu, observee=True):
    return {
        "culture": culture,
        "rendement_predit_t_ha": 1.0,
        "rendement_relatif_pct": relatif,
        "revenu_estime_par_ha": revenu,
        "culture_observee_dans_pays": observee,
    }


def test_classement_par_revenu_puis_cultures_sans_prix():
    resultats = [
        _resultat("Wheat", 120, 1500),
        _resultat("Maize", 150, 2000),
        _resultat("Rice", 200, None),
    ]
    classement = classer_cultures(resultats, prix_fournis=True)
    assert [r["culture"] for r in classement] == ["Maize", "Wheat", "Rice"]


def test_classement_par_rendement_relatif_sans_prix():
    resultats = [_resultat("Wheat", 120, None), _resultat("Maize", 150, None)]
    classement = classer_cultures(resultats, prix_fournis=False)
    assert [r["culture"] for r in classement] == ["Maize", "Wheat"]


def test_cultures_non_observees_toujours_a_la_fin():
    # L'igname a le meilleur revenu, mais n'est pas cultivée dans le pays
    resultats = [
        _resultat("Wheat", 120, 1500),
        _resultat("Yams", 170, 8000, observee=False),
    ]
    classement = classer_cultures(resultats, prix_fournis=True)
    assert [r["culture"] for r in classement] == ["Wheat", "Yams"]