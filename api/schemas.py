from pydantic import BaseModel, Field
from typing import Literal


# ---------- Requêtes ----------
class Contexte(BaseModel):
    """Conditions communes à /predict et /recommend."""
    pays: str = Field(..., examples=["France"])
    annee: int = Field(..., ge=1990, le=2030, examples=[2013])
    temperature: float = Field(
        ..., ge=-10, le=40,
        description="Température moyenne annuelle du pays (°C)",
        examples=[11.5],
    )
    pesticides_tonnes: float = Field(
        ..., ge=0,
        description="Pesticides utilisés dans le pays (tonnes)",
        examples=[60000],
    )


class RequetePrediction(Contexte):
    culture: str = Field(..., examples=["Wheat"])


class RequeteRecommandation(Contexte):
    prix: dict[str, float] | None = Field(
        default=None,
        description="Prix de vente par tonne, facultatif, pour une partie ou toutes les cultures",
        examples=[{"Wheat": 220, "Maize": 180}],
    )


# ---------- Réponses ----------
class ReponsePrediction(BaseModel):
    culture: str
    rendement_predit_t_ha: float
    culture_observee_dans_pays: bool


class ResultatCulture(BaseModel):
    culture: str
    rendement_predit_t_ha: float
    rendement_relatif_pct: float
    revenu_estime_par_ha: float | None
    culture_observee_dans_pays: bool


class ReponseRecommandation(BaseModel):
    critere_de_tri: str
    classement: list[ResultatCulture]

# ---------- Réponses LLM -----------
class MessageChat(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)

class RequeteChat(BaseModel):
    messages: list[MessageChat] = Field(min_length=1, max_length=20)

class ReponseChat(BaseModel):
    reponse: str
    outils_appeles: list[str]