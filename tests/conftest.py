"""Configuration partagée par tous les tests."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Client de test de l'API.

    Le bloc `with` déclenche le lifespan : le modèle et les fichiers
    de référence sont chargés comme au vrai démarrage de l'API.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def contexte_france():
    """Contexte valide et réaliste, réutilisé par plusieurs tests."""
    return {"pays": "France", "annee": 2013, "temperature": 11.5, "pesticides_tonnes": 60000}