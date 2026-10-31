"""Fixtures compartilhadas pelos testes."""

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.neural_network import MLPClassifier

from app.main import app


@pytest.fixture(scope="session")
def caminho_modelo(tmp_path_factory):
    """Treina um MLP minúsculo só para os testes e salva em um .joblib temporário."""
    rng = np.random.default_rng(0)
    X = rng.random((200, 784)).astype("float32")
    y = np.arange(200) % 10
    modelo = MLPClassifier(hidden_layer_sizes=(8,), max_iter=20, random_state=0).fit(X, y)
    caminho = tmp_path_factory.mktemp("modelos") / "mlp_teste.joblib"
    joblib.dump(modelo, caminho)
    return caminho


@pytest.fixture
def cliente(caminho_modelo, monkeypatch):
    """Cliente HTTP com a API ligada (lifespan executado) usando o modelo de teste."""
    monkeypatch.setattr("app.main.CAMINHO_MODELO", caminho_modelo)
    with TestClient(app) as c:
        yield c
