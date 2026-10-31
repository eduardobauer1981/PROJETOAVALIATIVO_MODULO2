"""Provas da Fase 5: endpoints e semântica dos status codes HTTP."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import ModeloService

VALIDO = {"pixels": [0] * 784}


# ---------------------------------------------------------------- GET /health
def test_health_200_com_modelo_na_ram(cliente):
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {
        "status": "ok",
        "servico_ativo": True,
        "modelo_carregado": True,
        "modelo_tipo": "MLPClassifier",
        "mensagem": "Serviço ativo e modelo carregado na memória.",
    }


def test_health_continua_200_e_indica_degradado_sem_o_arquivo(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.CAMINHO_MODELO", tmp_path / "nao_existe.joblib")

    with TestClient(app) as cliente:
        resposta = cliente.get("/health")
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados["status"] == "degradado"
        assert dados["servico_ativo"] is True
        assert dados["modelo_carregado"] is False
        assert dados["modelo_tipo"] is None


def test_arquivo_corrompido_nao_derruba_a_api(tmp_path, monkeypatch):
    corrompido = tmp_path / "corrompido.joblib"
    corrompido.write_bytes(b"isto nao e um modelo")
    monkeypatch.setattr("app.main.CAMINHO_MODELO", corrompido)

    with TestClient(app) as cliente:  # a subida não pode lançar exceção
        assert cliente.get("/health").json()["modelo_carregado"] is False
        resposta = cliente.post("/v1/predict", json=VALIDO)
        assert resposta.status_code == 503
        assert resposta.json() == {
            "detalhe": "Modelo indisponível. Verifique o endpoint /health.",
            "status": 503,
            "id_erro": None,
        }


# --------------------------------------------------------------- POST /v1/predict
def test_predict_200(cliente):
    resposta = cliente.post("/v1/predict", json=VALIDO)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert 0 <= dados["digito_previsto"] <= 9
    assert len(dados["probabilidades"]) == 10


# --------------------------------------------------------------- 4xx
def test_json_malformado_retorna_400_com_corpo_padronizado(cliente):
    resposta = cliente.post("/v1/predict", content="{pixels: [", headers={"Content-Type": "application/json"})

    assert resposta.status_code == 400
    assert resposta.json() == {
        "detalhe": "JSON malformado: o corpo da requisição não é um JSON válido.",
        "status": 400,
        "id_erro": None,
    }


def test_contrato_violado_retorna_422(cliente):
    resposta = cliente.post("/v1/predict", json={"pixels": [0] * 10})
    assert resposta.status_code == 422
    assert resposta.json()["erros"][0]["campo"] == "pixels"


def test_rota_inexistente_retorna_404_padronizado(cliente):
    resposta = cliente.get("/nada")
    assert resposta.status_code == 404
    assert resposta.json() == {"detalhe": "Rota não encontrada.", "status": 404, "id_erro": None}


def test_metodo_errado_retorna_405_com_header_allow(cliente):
    resposta = cliente.get("/v1/predict")
    assert resposta.status_code == 405
    assert resposta.json()["status"] == 405
    assert "POST" in resposta.headers["allow"]


def test_rotas_antigas_foram_substituidas(cliente):
    assert cliente.get("/saude").status_code == 404
    assert cliente.post("/predicao/digito", json=VALIDO).status_code == 404


# --------------------------------------------------------------- 500
def test_falha_na_inferencia_retorna_500_sem_vazar_detalhes_e_a_api_continua(cliente, monkeypatch):
    service = app.state.modelo_service

    def quebra(*args, **kwargs):
        raise ValueError("segredo-interno-123")

    with monkeypatch.context() as m:
        m.setattr(service._modelo, "predict_proba", quebra)
        resposta = cliente.post("/v1/predict", json=VALIDO)

    assert resposta.status_code == 500
    dados = resposta.json()
    assert dados["status"] == 500
    assert isinstance(dados["id_erro"], str) and len(dados["id_erro"]) == 8
    assert "segredo-interno-123" not in resposta.text  # detalhe interno só vai ao log

    # a API não caiu: a próxima requisição funciona normalmente
    assert cliente.post("/v1/predict", json=VALIDO).status_code == 200


def test_excecao_inesperada_retorna_500_e_a_api_continua(caminho_modelo, monkeypatch):
    monkeypatch.setattr("app.main.CAMINHO_MODELO", caminho_modelo)

    def explode(self, pixels):
        raise KeyError("boom-inesperado")

    with TestClient(app, raise_server_exceptions=False) as cliente:
        with monkeypatch.context() as m:
            m.setattr(ModeloService, "prever", explode)
            resposta = cliente.post("/v1/predict", json=VALIDO)

        assert resposta.status_code == 500
        assert resposta.json()["status"] == 500
        assert "boom-inesperado" not in resposta.text
        assert len(resposta.json()["id_erro"]) == 8

        assert cliente.post("/v1/predict", json=VALIDO).status_code == 200
        assert cliente.get("/health").status_code == 200
