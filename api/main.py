"""API de prédiction de rendement et de recommandation de cultures."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    ReponsePrediction,
    ReponseRecommandation,
    RequetePrediction,
    RequeteRecommandation,
)
from src.recommandation import calculer_resultats, classer_cultures, predire

# ---------- Chemins ----------
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_MODELES = RACINE / "models"

# Ressources chargées une seule fois, au démarrage de l'API
ressources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et les fichiers de référence au démarrage."""
    ressources["modele"] = joblib.load(DOSSIER_MODELES / "modele_xgboost.joblib")
    ressources["cultures"] = (
        pd.read_csv(DOSSIER_MODELES / "references_cultures.csv")
        .set_index("Crop")["rendement_moyen_t_ha"]
    )
    ressources["pays"] = pd.read_csv(DOSSIER_MODELES / "references_pays.csv").set_index("Area")
    # Pour chaque pays, l'ensemble des cultures présentes dans les données d'entraînement
    paires = pd.read_csv(DOSSIER_MODELES / "references_pays_cultures.csv")
    ressources["cultures_par_pays"] = paires.groupby("Area")["Crop"].apply(set).to_dict()
    yield
    ressources.clear()


app = FastAPI(
    title="API Agritech Answers",
    description="Prédiction du rendement des cultures et recommandation de la culture la plus adaptée.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Vérifications (erreurs HTTP) ----------
def verifier_pays(pays: str) -> None:
    if pays not in ressources["pays"].index:
        raise HTTPException(status_code=422, detail=f"Pays inconnu : {pays}. Voir GET /pays.")


def verifier_culture(culture: str) -> None:
    if culture not in ressources["cultures"].index:
        raise HTTPException(status_code=422, detail=f"Culture inconnue : {culture}. Voir GET /cultures.")


def verifier_prix(prix: dict[str, float] | None) -> None:
    for culture, valeur in (prix or {}).items():
        verifier_culture(culture)
        if valeur <= 0:
            raise HTTPException(status_code=422, detail=f"Le prix de {culture} doit être positif.")


# ---------- Endpoints ----------
@app.get("/health")
def health():
    """Vérifie que l'API fonctionne et que le modèle est chargé."""
    return {"statut": "ok", "modele_charge": "modele" in ressources}


@app.get("/cultures")
def lister_cultures():
    """Liste des cultures connues du modèle."""
    return {"cultures": ressources["cultures"].index.tolist()}


@app.get("/pays")
def lister_pays():
    """Liste des pays connus, avec les valeurs par défaut de température et de pesticides."""
    return {"pays": ressources["pays"].reset_index().to_dict(orient="records")}


@app.post("/predict", response_model=ReponsePrediction)
def predict(requete: RequetePrediction):
    """Prédit le rendement d'une culture dans un contexte donné."""
    verifier_pays(requete.pays)
    verifier_culture(requete.culture)

    rendement = predire(ressources["modele"], requete.model_dump(), [requete.culture])[0]
    cultures_observees = ressources["cultures_par_pays"].get(requete.pays, set())
    return ReponsePrediction(
        culture=requete.culture,
        rendement_predit_t_ha=round(rendement, 3),
        culture_observee_dans_pays=requete.culture in cultures_observees,
    )


@app.post("/recommend", response_model=ReponseRecommandation)
def recommend(requete: RequeteRecommandation):
    """Prédit le rendement de toutes les cultures et les classe."""
    verifier_pays(requete.pays)
    verifier_prix(requete.prix)

    cultures = ressources["cultures"].index.tolist()
    rendements = predire(ressources["modele"], requete.model_dump(), cultures)
    cultures_observees = ressources["cultures_par_pays"].get(requete.pays, set())
    resultats = calculer_resultats(
        cultures, rendements, ressources["cultures"], cultures_observees, requete.prix
    )

    prix_fournis = bool(requete.prix)
    return ReponseRecommandation(
        critere_de_tri="revenu estimé" if prix_fournis else "rendement relatif",
        classement=classer_cultures(resultats, prix_fournis=prix_fournis),
    )