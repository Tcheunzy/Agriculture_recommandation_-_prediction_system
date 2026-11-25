"""Interface Streamlit : interroge l'API, sans aucune logique de Machine Learning."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Adresse de l'API : variable d'environnement en déploiement, sinon l'API locale.
# (Sur Streamlit Community Cloud, les secrets de premier niveau sont aussi exposés en variables d'environnement.)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

NOMS_CULTURES = {
    "Cassava": "Manioc",
    "Maize": "Maïs",
    "Plantains and others": "Banane plantain",
    "Potatoes": "Pomme de terre",
    "Rice": "Riz",
    "Sorghum": "Sorgho",
    "Soybean": "Soja",
    "Sweet potatoes": "Patate douce",
    "Wheat": "Blé",
    "Yams": "Igname",
}

st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="wide")


# ---------- Appels à l'API ----------
def appeler_api(methode: str, route: str, donnees: dict | None = None) -> dict | None:
    """Appelle l'API et affiche un message clair en cas d'erreur."""
    try:
        reponse = requests.request(methode, f"{API_URL}{route}", json=donnees, timeout=30)
    except requests.exceptions.RequestException:
        st.error(f"Impossible de joindre l'API ({API_URL}). Vérifiez qu'elle est démarrée.")
        return None
    if reponse.status_code != 200:
        detail = reponse.json().get("detail", reponse.text)
        st.error(f"Erreur de l'API ({reponse.status_code}) : {detail}")
        return None
    return reponse.json()


@st.cache_data(ttl=3600)
def charger_references() -> tuple[pd.DataFrame, list[str]] | None:
    """Liste des pays (avec leurs valeurs par défaut) et des cultures, gardée en cache 1 h."""
    pays = appeler_api("GET", "/pays")
    cultures = appeler_api("GET", "/cultures")
    if pays is None or cultures is None:
        return None
    return pd.DataFrame(pays["pays"]).set_index("Area"), cultures["cultures"]


# ---------- Assistant conversationnel ----------
def afficher_assistant() -> None:
    """Onglet de discussion : envoie la conversation à /chat et affiche la réponse."""
    st.subheader("Assistant")
    st.caption(
        "Posez votre question en langage courant (ex. : « Que cultiver en Inde en 2012 ? »). "
        "Les chiffres viennent toujours du modèle, jamais de l'assistant."
    )

    # L'historique survit aux rechargements de la page grâce à session_state
    if "historique" not in st.session_state:
        st.session_state.historique = []

    if st.button("Nouvelle conversation"):
        st.session_state.historique = []
        st.rerun()

    # Réaffiche toute la conversation à chaque rechargement
    for message in st.session_state.historique:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Votre question…")
    if not question:
        return

    st.session_state.historique.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("L'assistant réfléchit…"):
            try:
                reponse = requests.post(
                    f"{API_URL}/chat",
                    json={"messages": st.session_state.historique[-20:]},
                    timeout=120,
                )
            except requests.RequestException:
                reponse = None

        if reponse is not None and reponse.ok:
            texte = reponse.json()["reponse"]
            st.markdown(texte)
            st.session_state.historique.append({"role": "assistant", "content": texte})
        else:
            if reponse is None:
                detail = "API injoignable (elle peut mettre une minute à démarrer, réessayez)."
            else:
                try:
                    detail = reponse.json().get("detail", reponse.text)
                except ValueError:  # réponse qui n'est pas du JSON (page d'erreur du serveur)
                    detail = f"erreur {reponse.status_code}"
            st.error(f"L'assistant n'a pas pu répondre : {detail}")
            st.session_state.historique.pop()


# ---------- En-tête ----------
st.title("🌾 Agritech Answers")
st.caption("Estimez le rendement de vos cultures et identifiez la culture la plus adaptée à vos conditions.")

references = charger_references()
if references is None:
    st.stop()
df_pays, cultures = references

# ---------- Barre latérale : mode et contexte ----------
with st.sidebar:
    st.header("Paramètres")
    mode = st.radio("Mode", ["Prédiction", "Recommandation", "Assistant"], horizontal=True)

    # Les paramètres ne servent pas à l'assistant : l'utilisateur les donne dans sa question
    if mode != "Assistant":
        pays = st.selectbox("Pays", df_pays.index.tolist(), index=df_pays.index.tolist().index("France"))
        defauts = df_pays.loc[pays]

        annee = st.slider("Année", min_value=1990, max_value=2030, value=2013)

        # Les clés dépendent du pays : changer de pays remet les valeurs par défaut de ce pays
        temperature = st.slider(
            "Température moyenne annuelle du pays (°C)",
            min_value=-5.0, max_value=35.0,
            value=float(round(defauts["avg_temp_defaut"], 1)), step=0.1,
            key=f"temperature_{pays}",
        )
        pesticides_defaut = float(defauts["pesticides_tonnes_defaut"])
        pesticides = st.slider(
            "Pesticides utilisés dans le pays (tonnes)",
            min_value=0.0,
            max_value=float(max(round(pesticides_defaut * 3), 100)),
            value=float(round(pesticides_defaut)),
            step=float(max(round(pesticides_defaut / 100), 1)),
            key=f"pesticides_{pays}",
            help="Indicateur du niveau d'intensification agricole du pays. "
                 f"Valeur récente pour {pays} : {pesticides_defaut:,.0f} t.",
        )

if mode != "Assistant":
    contexte = {"pays": pays, "annee": annee, "temperature": temperature, "pesticides_tonnes": pesticides}

# ---------- Mode Prédiction ----------
if mode == "Prédiction":
    st.subheader("Prédiction du rendement")
    culture = st.selectbox(
        "Culture", cultures,
        index=cultures.index("Wheat") if "Wheat" in cultures else 0,
        format_func=lambda c: NOMS_CULTURES.get(c, c),
    )

    if st.button("Prédire le rendement", type="primary"):
        resultat = appeler_api("POST", "/predict", {**contexte, "culture": culture})
        if resultat:
            st.metric(
                label=f"Rendement estimé — {NOMS_CULTURES.get(culture, culture)} ({pays}, {annee})",
                value=f"{resultat['rendement_predit_t_ha']:.2f} t/ha",
            )
            if not resultat["culture_observee_dans_pays"]:
                st.warning(
                    "Cette culture n'est pas cultivée dans ce pays dans nos données : "
                    "l'estimation repose uniquement sur d'autres pays et doit être interprétée avec prudence."
                )

# ---------- Mode Recommandation ----------
elif mode == "Recommandation":
    st.subheader("Recommandation de culture")

    with st.expander("Prix de vente par tonne (facultatif)"):
        st.caption(
            "Renseignez vos prix de vente (dans la devise de votre choix, la même pour toutes les cultures) "
            "pour classer les cultures par revenu estimé. Laissez 0 si vous ne connaissez pas le prix."
        )
        colonnes = st.columns(2)
        prix = {}
        for i, culture in enumerate(cultures):
            valeur = colonnes[i % 2].number_input(
                NOMS_CULTURES.get(culture, culture), min_value=0.0, value=0.0, step=0.20, key=f"prix_{culture}"
            )
            if valeur > 0:
                prix[culture] = valeur

    if st.button("Obtenir la recommandation", type="primary"):
        resultat = appeler_api("POST", "/recommend", {**contexte, "prix": prix or None})
        if resultat:
            df = pd.DataFrame(resultat["classement"])
            df["Culture"] = df["culture"].map(lambda c: NOMS_CULTURES.get(c, c))
            df["Statut"] = df["culture_observee_dans_pays"].map(
                {True: "Cultivée dans ce pays", False: "Non cultivée dans ce pays"}
            )
            par_revenu = resultat["critere_de_tri"] == "revenu estimé"

            meilleure = df.iloc[0]
            st.success(
                f"Culture recommandée : **{meilleure['Culture']}** "
                f"({meilleure['rendement_predit_t_ha']:.2f} t/ha"
                + (f", revenu estimé {meilleure['revenu_estime_par_ha']:,.0f} par ha)" if par_revenu and pd.notna(meilleure["revenu_estime_par_ha"]) else ")")
                + f" — classement par {resultat['critere_de_tri']}."
            )

            # Graphique : revenu si des prix sont fournis, sinon rendement relatif
            if par_revenu:
                donnees_graph = df.dropna(subset=["revenu_estime_par_ha"])
                axe, titre = "revenu_estime_par_ha", "Revenu estimé par hectare"
            else:
                donnees_graph = df
                axe, titre = "rendement_relatif_pct", "Rendement relatif (% de la moyenne de la culture)"

            graphique = (
                alt.Chart(donnees_graph)
                .mark_bar()
                .encode(
                    x=alt.X(f"{axe}:Q", title=titre),
                    y=alt.Y("Culture:N", sort=None, title=None),
                    color=alt.Color(
                        "Statut:N",
                        scale=alt.Scale(
                            domain=["Cultivée dans ce pays", "Non cultivée dans ce pays"],
                            range=["#2e7d32", "#bdbdbd"],
                        ),
                        legend=alt.Legend(title=None, orient="bottom"),
                    ),
                    tooltip=["Culture", "rendement_predit_t_ha", "rendement_relatif_pct", "revenu_estime_par_ha"],
                )
                .properties(height=40 * len(donnees_graph))
            )
            st.altair_chart(graphique, width="stretch")

            # Tableau détaillé
            tableau = df[["Culture", "rendement_predit_t_ha", "rendement_relatif_pct", "revenu_estime_par_ha", "Statut"]]
            tableau = tableau.rename(columns={
                "rendement_predit_t_ha": "Rendement estimé (t/ha)",
                "rendement_relatif_pct": "Rendement relatif (%)",
                "revenu_estime_par_ha": "Revenu estimé par ha",
            })
            st.dataframe(tableau, hide_index=True, width="stretch")

            st.caption(
                "Rendement relatif : rendement estimé comparé au rendement moyen mondial de la culture. "
                "Les cultures non cultivées dans ce pays sont classées en dernier : leurs estimations sont moins fiables."
            )

# ---------- Mode Assistant ----------
else:
    afficher_assistant()