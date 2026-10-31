"""Provas da Fase 3: o modelo é lido do disco UMA vez (no lifespan) e nunca na rota."""

import ast
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import app

PASTA_ROUTERS = Path(__file__).resolve().parent.parent / "app" / "routers"
CORPO_VALIDO = {"pixels": [0] * 784}


@pytest.fixture
def contador_de_leituras(monkeypatch):
    """Conta quantas vezes joblib.load é chamado."""
    chamadas = {"n": 0}
    original = joblib.load

    def contador(*args, **kwargs):
        chamadas["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", contador)
    return chamadas


def test_modelo_e_lido_uma_unica_vez_mesmo_com_varias_requisicoes(
    caminho_modelo, contador_de_leituras, monkeypatch
):
    monkeypatch.setattr("app.main.CAMINHO_MODELO", caminho_modelo)

    with TestClient(app) as cliente:
        assert contador_de_leituras["n"] == 1  # carregado na inicialização

        for _ in range(25):
            resposta = cliente.post("/predicao/digito", json=CORPO_VALIDO)
            assert resposta.status_code == 200

        assert contador_de_leituras["n"] == 1  # nenhuma leitura extra nas rotas


def test_mesmo_objeto_de_modelo_atende_todas_as_requisicoes(caminho_modelo, monkeypatch):
    monkeypatch.setattr("app.main.CAMINHO_MODELO", caminho_modelo)

    with TestClient(app) as cliente:
        service = app.state.modelo_service
        modelo_antes = id(service._modelo)

        for _ in range(5):
            cliente.post("/predicao/digito", json=CORPO_VALIDO)

        assert app.state.modelo_service is service
        assert id(service._modelo) == modelo_antes  # o mesmo objeto na RAM


def test_carregar_e_idempotente(caminho_modelo, contador_de_leituras):
    from app.services import ModeloService

    service = ModeloService(caminho_modelo)
    service.carregar()
    service.carregar()
    service.carregar()
    assert contador_de_leituras["n"] == 1


def test_sem_arquivo_nao_le_disco_e_rota_responde_503(tmp_path, contador_de_leituras, monkeypatch):
    monkeypatch.setattr("app.main.CAMINHO_MODELO", tmp_path / "nao_existe.joblib")

    with TestClient(app) as cliente:
        assert cliente.get("/saude").json()["modelo_carregado"] is False
        assert cliente.post("/predicao/digito", json=CORPO_VALIDO).status_code == 503

    assert contador_de_leituras["n"] == 0  # a rota não tentou ler o disco por conta própria


def test_codigo_das_rotas_nao_le_arquivos():
    """Verificação estática: nenhum router importa joblib/pickle nem chama open()/load()."""
    proibidos = {"joblib", "pickle", "dill", "cloudpickle"}

    for arquivo in PASTA_ROUTERS.glob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                assert not ({a.name.split(".")[0] for a in no.names} & proibidos), arquivo.name
            if isinstance(no, ast.ImportFrom) and no.module:
                assert no.module.split(".")[0] not in proibidos, arquivo.name
            if isinstance(no, ast.Call):
                f = no.func
                assert not (isinstance(f, ast.Name) and f.id == "open"), arquivo.name
                assert not (isinstance(f, ast.Attribute) and f.attr == "load"), arquivo.name
