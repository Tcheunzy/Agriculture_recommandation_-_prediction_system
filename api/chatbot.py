"""Assistant conversationnel : LLM + outils de l'API (function calling).

Principe : le LLM comprend la question et choisit un outil ; l'API exécute
l'outil avec le vrai modèle ; le LLM reformule le résultat. Aucun chiffre
ne vient du LLM lui-même.

Le client utilise le format OpenAI, compris par la plupart des fournisseurs
(Groq, Gemini, OpenAI, Mistral...). Changer de fournisseur = changer trois
variables d'environnement : LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
"""
import json
import logging
import os
from typing import Callable

from openai import OpenAI

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
NB_TOURS_MAX = 5  # garde-fou : nombre maximal d'allers-retours avec les outils


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------

def construire_prompt_systeme(cultures: list[str]) -> str:
    """Prompt système, avec la liste exacte des cultures connues du modèle."""
    liste = ", ".join(cultures)
    return f"""Tu es l'assistant d'Agritech Answers. Tu aides des agriculteurs à estimer
le rendement de leurs cultures et à choisir quoi cultiver.

Tu disposes de deux outils :
- predire_rendement : rendement estimé d'UNE culture dans un pays ;
- recommander_cultures : classement des 10 cultures pour un pays.
Cultures disponibles (noms exacts à utiliser dans les outils) : {liste}.

RÈGLES IMPÉRATIVES :
1. Tout chiffre que tu donnes (rendement, classement, revenu) doit venir d'un outil.
   N'estime jamais un rendement toi-même.
2. Tu ne donnes JAMAIS de prix de vente. Tu n'utilises que les prix fournis par l'utilisateur.
   Sans prix, explique que le classement se fait par rendement relatif, et propose à
   l'utilisateur de donner ses prix pour obtenir un classement par revenu.
3. S'il manque le pays ou l'année, demande-les avant d'appeler un outil.
4. Pour une recommandation, présente les 3 premières cultures dans l'ordre exact renvoyé
   par l'outil, avec leur nom en français, et précise le critère de classement renvoyé
   par l'outil (rendement relatif ou revenu estimé). Signale si une culture n'a jamais
   été observée dans le pays.
5. Dès qu'un pays est cité, appelle l'outil, même si ce pays te semble inconnu ou fictif :
   c'est l'outil qui vérifie s'il est dans les données. Si l'outil renvoie une erreur,
   explique-la simplement (par exemple : ce pays n'est pas couvert par nos données).
6. Si la question sort de ce domaine, dis-le poliment.

FORMAT : réponse directe en une à trois phrases, détails seulement si utiles. Rendements
en t/ha, arrondis à 0,1. Ton professionnel, en français, sans émojis.
"""


# ---------------------------------------------------------------------------
# Description des outils (ce que le LLM voit)
# ---------------------------------------------------------------------------

OUTIL_PREDICTION = {
    "type": "function",
    "function": {
        "name": "predire_rendement",
        "description": (
            "Prédit le rendement (en tonnes par hectare) d'UNE culture donnée "
            "dans un pays, pour une année et des conditions données."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "description": "Nom exact de la culture en anglais, tel que listé dans les consignes.",
                },
                "pays": {
                    "type": "string",
                    "description": "Nom du pays en anglais, format FAO (ex. 'France', 'India').",
                },
                "annee": {"type": "integer", "description": "Année entre 1990 et 2030."},
                "temperature": {
                    "type": "number",
                    "description": "Température moyenne annuelle en °C. Optionnel : valeur par défaut du pays si absent.",
                },
                "pesticides_tonnes": {
                    "type": "number",
                    "description": "Pesticides utilisés dans le pays, en tonnes. Optionnel : valeur par défaut du pays si absent.",
                },
            },
            "required": ["culture", "pays", "annee"],
        },
    },
}

OUTIL_RECOMMANDATION = {
    "type": "function",
    "function": {
        "name": "recommander_cultures",
        "description": (
            "Classe les 10 cultures disponibles pour un pays, une année et des conditions "
            "données, afin de recommander la plus adaptée ou la plus rentable. "
            "À utiliser quand l'utilisateur demande quoi cultiver ou compare plusieurs cultures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pays": {
                    "type": "string",
                    "description": "Nom du pays en anglais, format FAO (ex. 'France', 'India').",
                },
                "annee": {"type": "integer", "description": "Année entre 1990 et 2030."},
                "temperature": {
                    "type": "number",
                    "description": "Température moyenne annuelle en °C. Optionnel : valeur par défaut du pays si absent.",
                },
                "pesticides_tonnes": {
                    "type": "number",
                    "description": "Pesticides utilisés dans le pays, en tonnes. Optionnel : valeur par défaut du pays si absent.",
                },
                "prix": {
                    "type": "object",
                    "description": (
                        "Prix de vente par tonne, par culture (noms exacts en anglais). "
                        "À renseigner UNIQUEMENT avec des prix que l'utilisateur a donnés explicitement "
                        "dans la conversation. Ne jamais estimer, deviner ou compléter un prix : "
                        "si l'utilisateur n'a donné aucun prix, omettre ce champ."
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["pays", "annee"],
        },
    },
}

OUTILS = [OUTIL_PREDICTION, OUTIL_RECOMMANDATION]


# ---------------------------------------------------------------------------
# Client et boucle de conversation
# ---------------------------------------------------------------------------

def creer_client() -> OpenAI | None:
    """Crée le client du LLM, ou renvoie None si la clé est absente.

    Ne lève pas d'erreur : sans clé, seul /chat est désactivé, le reste de l'API fonctionne.
    max_retries : le SDK réessaie automatiquement (avec attente croissante) en cas de 429.
    """
    cle = os.getenv("LLM_API_KEY")
    if not cle:
        logger.warning("LLM_API_KEY absente : l'endpoint /chat est désactivé.")
        return None
    logger.info("Client LLM initialisé (%s, modèle : %s).", LLM_BASE_URL, LLM_MODEL)
    return OpenAI(api_key=cle, base_url=LLM_BASE_URL, max_retries=3)


def _lire_arguments(arguments) -> dict:
    """Les arguments d'un appel d'outil arrivent en texte JSON (ou déjà en dict)."""
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}


def repondre(
    client: OpenAI,
    historique: list[dict],
    cultures: list[str],
    executer_outil: Callable[[str, dict], dict],
) -> tuple[str, list[str]]:
    """Fait répondre le LLM, en exécutant les outils qu'il demande.

    historique : messages {"role": "user" | "assistant", "content": "..."}
    executer_outil : fonction fournie par l'API, (nom_outil, arguments) -> résultat (dict)
    Renvoie le texte final et la liste des outils appelés.
    """
    messages = [{"role": "system", "content": construire_prompt_systeme(cultures)}, *historique]
    outils_appeles: list[str] = []

    for _ in range(NB_TOURS_MAX):
        reponse = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=OUTILS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = reponse.choices[0].message

        # Pas d'outil demandé : c'est la réponse finale
        if not message.tool_calls:
            return message.content or "", outils_appeles

        # Le LLM demande un ou plusieurs outils : on garde sa demande dans l'historique...
        # (on ne garde que les champs standard : certains modèles ajoutent un champ
        #  "reasoning" que l'API refuse de recevoir en retour)
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [appel.model_dump() for appel in message.tool_calls],
        })

        # ... puis on exécute chaque outil et on lui renvoie le résultat
        for appel in message.tool_calls:
            nom = appel.function.name
            arguments = _lire_arguments(appel.function.arguments)
            logger.info("Outil demandé : %s %s", nom, arguments)

            resultat = executer_outil(nom, arguments)
            outils_appeles.append(nom)

            messages.append({
                "role": "tool",
                "tool_call_id": appel.id,
                "content": json.dumps(resultat, ensure_ascii=False),
            })

    logger.warning("Nombre maximal de tours atteint sans réponse finale.")
    return (
        "Je n'ai pas réussi à finaliser ma réponse. Pouvez-vous reformuler votre question ?",
        outils_appeles,
    )