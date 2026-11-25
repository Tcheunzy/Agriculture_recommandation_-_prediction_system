"""Pipeline d'entraînement : données → modèle évalué et sauvegardé.
 
Étapes : chargement, découpage temporel, entraînement XGBoost,
évaluation sur 2010-2013, suivi MLflow, sauvegarde du modèle
et des tables de référence utilisées par l'API.
 
Usage : python -m src.train
"""
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
 
from src.features import VARIABLES_CATEGORIELLES, VARIABLES_MODELE, VARIABLES_NUMERIQUES
 
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DONNEES = RACINE / "data" / "dataset_consolide.csv"
DOSSIER_MODELES = RACINE / "models"
 
CIBLE = "yield_t_ha"
DERNIERE_ANNEE_TRAIN = 2009
RANDOM_STATE = 42
NB_ANNEES_DEFAUT = 5
 
# Meilleurs paramètres trouvés par RandomizedSearchCV (notebook de modélisation)
HYPERPARAMETRES = {
    "n_estimators": 1000,
    "learning_rate": 0.10215580871496355,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9637281608315128,
    "colsample_bytree": 0.6964101864104046,
    "reg_lambda": 0.1171593739230706,
}
 
 
def charger_donnees(chemin: Path) -> pd.DataFrame:
    """Charge le CSV consolidé (variables déjà calculées : log_pesticides, temp_sqr)."""
    df = pd.read_csv(chemin)
    colonnes_manquantes = set(VARIABLES_MODELE + [CIBLE]) - set(df.columns)
    if colonnes_manquantes:
        raise ValueError(f"Colonnes absentes du CSV : {colonnes_manquantes}")
    return df
 
 
def decouper(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données par date : le passé pour apprendre, le futur pour évaluer."""
    masque_train = df["Year"] <= DERNIERE_ANNEE_TRAIN
    return df[masque_train], df[~masque_train]
 
 
def construire_pipeline() -> Pipeline:
    """Encodage des variables catégorielles + XGBoost, dans un seul objet."""
    preparation = ColumnTransformer(
        transformers=[
            ("categorielles", OneHotEncoder(handle_unknown="ignore"), VARIABLES_CATEGORIELLES),
            ("numeriques", "passthrough", VARIABLES_NUMERIQUES),
        ]
    )
    modele = XGBRegressor(**HYPERPARAMETRES, random_state=RANDOM_STATE)
    return Pipeline(steps=[("preparation", preparation), ("modele", modele)])
 
 
def evaluer(modele, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule les métriques de régression sur un jeu de données."""
    predictions = modele.predict(X)
    return {
        "mae": mean_absolute_error(y, predictions),
        "rmse": root_mean_squared_error(y, predictions),
        "r2": r2_score(y, predictions),
    }
 
 
def sauvegarder_references(df: pd.DataFrame, train: pd.DataFrame, dossier: Path) -> None:
    """Régénère les 3 tables de référence lues par l'API.
 
    Les tables qui dépendent du rendement (la cible) sont calculées sur le train
    uniquement, pour éviter toute fuite d'information du test.
    """
    dossier.mkdir(parents=True, exist_ok=True)
 
    # 1. Rendement moyen par culture (cible → train uniquement)
    cultures = (
        train.groupby("Crop")[CIBLE]
        .mean()
        .rename("rendement_moyen_t_ha")
        .reset_index()
    )
    cultures.to_csv(dossier / "references_cultures.csv", index=False)
 
    # 2. Couples pays-culture observés (train uniquement)
    pays_cultures = (
        train.groupby(["Area", "Crop"])["Year"]
        .nunique()
        .rename("nb_annees")
        .reset_index()
    )
    pays_cultures.to_csv(dossier / "references_pays_cultures.csv", index=False)
 
    # 3. Valeurs par défaut par pays : moyenne des dernières années connues,
    #    plus robuste qu'une seule année. Température et pesticides ne sont
    #    pas la cible : on peut utiliser toutes les années.
    par_pays_annee = df.drop_duplicates(["Area", "Year"])
    derniere = par_pays_annee.groupby("Area")["Year"].transform("max")
    recentes = par_pays_annee[par_pays_annee["Year"] > derniere - NB_ANNEES_DEFAUT]
    pays = (
        recentes.assign(pesticides_tonnes=np.expm1(recentes["log_pesticides"]))
        .groupby("Area")
        .agg(
            avg_temp_defaut=("avg_temp", "mean"),
            pesticides_tonnes_defaut=("pesticides_tonnes", "mean"),
            derniere_annee=("Year", "max"),
        )
        .round(2)
        .reset_index()
    )
    pays.to_csv(dossier / "references_pays.csv", index=False)
 
 
def main() -> None:
    import mlflow
    df = charger_donnees(CHEMIN_DONNEES)
    train, test = decouper(df)
    X_train, y_train = train[VARIABLES_MODELE], train[CIBLE]
    X_test, y_test = test[VARIABLES_MODELE], test[CIBLE]
 
    mlflow.set_experiment("entrainement_production")
    with mlflow.start_run(run_name="xgboost_final"):
        mlflow.log_params(HYPERPARAMETRES)
        mlflow.log_params({
            "derniere_annee_train": DERNIERE_ANNEE_TRAIN,
            "nb_lignes_train": len(train),
            "nb_lignes_test": len(test),
        })
 
        modele = construire_pipeline()
        modele.fit(X_train, y_train)
 
        metriques = evaluer(modele, X_test, y_test)
        mlflow.log_metrics({f"test_{nom}": valeur for nom, valeur in metriques.items()})
 
        DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
        chemin_modele = DOSSIER_MODELES / "modele_xgboost.joblib"
        joblib.dump(modele, chemin_modele)
        mlflow.log_artifact(str(chemin_modele))
 
        sauvegarder_references(df, train, DOSSIER_MODELES)
 
    print(f"Entraînement : {len(train)} lignes (≤ {DERNIERE_ANNEE_TRAIN}) | Test : {len(test)} lignes")
    for nom, valeur in metriques.items():
        print(f"  {nom.upper():<5} test = {valeur:.3f}")
    print(f"Modèle et références sauvegardés dans {DOSSIER_MODELES}")
 
 
if __name__ == "__main__":
    main()