"""Logique métier : prédiction du rendement et classement des cultures.

Ce module ne dépend pas de FastAPI : ses fonctions reçoivent tout ce dont
elles ont besoin en paramètres, ce qui permet de les tester isolément.
"""
import pandas as pd

from src.features import creer_variables


def predire(modele, contexte: dict, cultures: list[str]) -> list[float]:
    """Prédit le rendement (t/ha) de chaque culture pour un même contexte.

    Args:
        modele: pipeline scikit-learn entraîné (preprocessing + modèle).
        contexte: dictionnaire avec les clés "pays", "annee", "temperature"
            et "pesticides_tonnes".
        cultures: liste des cultures à prédire.

    Returns:
        Les rendements prédits, dans le même ordre que `cultures`.
        Les éventuelles valeurs négatives sont ramenées à 0.
    """
    # Une ligne par culture, avec les noms de colonnes attendus par le modèle
    donnees = pd.DataFrame({
        "Crop": cultures,
        "Area": contexte["pays"],
        "Year": contexte["annee"],
        "pesticides_tonnes": contexte["pesticides_tonnes"],
        "avg_temp": contexte["temperature"],
    })
    X = creer_variables(donnees)
    predictions = modele.predict(X)
    return [max(0.0, float(p)) for p in predictions]


def calculer_resultats(
    cultures: list[str],
    rendements: list[float],
    rendements_moyens: pd.Series,
    cultures_observees: set[str],
    prix: dict[str, float] | None = None,
) -> list[dict]:
    """Calcule le rendement relatif et le revenu estimé de chaque culture.

    Args:
        cultures: liste des cultures.
        rendements: rendements prédits (t/ha), dans le même ordre que `cultures`.
        rendements_moyens: rendement moyen de chaque culture (t/ha), indexé par culture.
        cultures_observees: cultures présentes dans les données d'entraînement pour ce pays.
        prix: prix de vente par tonne, facultatif, pour une partie ou toutes les cultures.

    Returns:
        Une liste de dictionnaires, un par culture.
    """
    prix = prix or {}
    resultats = []
    for culture, rendement in zip(cultures, rendements):
        moyenne = float(rendements_moyens[culture])
        revenu = rendement * prix[culture] if culture in prix else None
        resultats.append({
            "culture": culture,
            "rendement_predit_t_ha": round(rendement, 3),
            "rendement_relatif_pct": round(rendement / moyenne * 100, 1),
            "revenu_estime_par_ha": round(revenu, 2) if revenu is not None else None,
            "culture_observee_dans_pays": culture in cultures_observees,
        })
    return resultats


def _trier(resultats: list[dict], prix_fournis: bool) -> list[dict]:
    """Trie un groupe de cultures par revenu estimé (si prix) ou par rendement relatif."""
    if prix_fournis:
        avec_prix = [r for r in resultats if r["revenu_estime_par_ha"] is not None]
        sans_prix = [r for r in resultats if r["revenu_estime_par_ha"] is None]
        avec_prix.sort(key=lambda r: r["revenu_estime_par_ha"], reverse=True)
        sans_prix.sort(key=lambda r: r["rendement_relatif_pct"], reverse=True)
        return avec_prix + sans_prix
    return sorted(resultats, key=lambda r: r["rendement_relatif_pct"], reverse=True)


def classer_cultures(resultats: list[dict], prix_fournis: bool) -> list[dict]:
    """Classe les cultures.

    Les cultures déjà observées dans le pays passent en premier, les autres
    (prédictions sans données de référence pour ce pays) à la fin.
    À l'intérieur de chaque groupe :
    - avec des prix : les cultures ayant un prix d'abord, triées par revenu
      estimé décroissant, puis les autres, triées par rendement relatif ;
    - sans prix : tri par rendement relatif décroissant.
    """
    observees = [r for r in resultats if r["culture_observee_dans_pays"]]
    non_observees = [r for r in resultats if not r["culture_observee_dans_pays"]]
    return _trier(observees, prix_fournis) + _trier(non_observees, prix_fournis)