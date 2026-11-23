# Image de base : Python 3.12, la version utilisée pour entraîner le modèle
FROM python:3.12-slim

# Pas de fichiers .pyc, et logs affichés immédiatement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Bibliothèque système nécessaire à XGBoost (calcul parallèle)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances d'abord : cette étape reste en cache tant que requirements.txt ne change pas
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application et modèle
COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000

# Vérifie toutes les 30 s que l'API répond sur /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]