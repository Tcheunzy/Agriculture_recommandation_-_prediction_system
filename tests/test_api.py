"""Tests des endpoints de l'API (api/main.py)."""


def test_health(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["modele_charge"] is True


def test_liste_des_cultures(client):
    reponse = client.get("/cultures")
    assert reponse.status_code == 200
    assert len(reponse.json()["cultures"]) == 10


def test_liste_des_pays_avec_valeurs_par_defaut(client):
    pays = client.get("/pays").json()["pays"]
    assert len(pays) > 0
    assert {"Area", "avg_temp_defaut", "pesticides_tonnes_defaut"} <= set(pays[0])


def test_predict_valide(client, contexte_france):
    reponse = client.post("/predict", json={**contexte_france, "culture": "Wheat"})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["rendement_predit_t_ha"] > 0
    assert corps["culture_observee_dans_pays"] is True


def test_recommend_renvoie_les_10_cultures(client, contexte_france):
    reponse = client.post("/recommend", json=contexte_france)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["critere_de_tri"] == "rendement relatif"
    assert len(corps["classement"]) == 10


def test_recommend_avec_prix(client, contexte_france):
    reponse = client.post("/recommend", json={**contexte_france, "prix": {"Wheat": 220}})
    corps = reponse.json()
    assert corps["critere_de_tri"] == "revenu estimé"
    assert corps["classement"][0]["culture"] == "Wheat"  # seule culture avec un prix


def test_pays_inconnu(client, contexte_france):
    reponse = client.post("/predict", json={**contexte_france, "pays": "Atlantis", "culture": "Wheat"})
    assert reponse.status_code == 422


def test_culture_inconnue(client, contexte_france):
    reponse = client.post("/predict", json={**contexte_france, "culture": "Banana"})
    assert reponse.status_code == 422


def test_prix_negatif(client, contexte_france):
    reponse = client.post("/recommend", json={**contexte_france, "prix": {"Wheat": -10}})
    assert reponse.status_code == 422


def test_temperature_hors_limites(client, contexte_france):
    reponse = client.post("/predict", json={**contexte_france, "temperature": 80, "culture": "Wheat"})
    assert reponse.status_code == 422


def test_champ_manquant(client):
    reponse = client.post("/predict", json={"pays": "France", "culture": "Wheat"})
    assert reponse.status_code == 422

def test_chat_sans_cle_renvoie_503(client):
    reponse = client.post("/chat", json={"messages": [{"role": "user", "content": "Bonjour"}]})
    assert reponse.status_code == 503