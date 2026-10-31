"""Provas da Fase 4: dados inválidos viram HTTP 422 ANTES de chegar ao modelo."""

import copy

import pytest

from app.services import ModeloService

URL = "/predicao/digito"
VALIDO = {"pixels": [0] * 784}


def com(**mudancas):
    """Cópia do corpo válido com alterações."""
    corpo = copy.deepcopy(VALIDO)
    corpo.update(mudancas)
    return corpo


@pytest.fixture
def chamadas_ao_modelo(monkeypatch):
    """Conta quantas vezes o serviço de predição foi realmente acionado."""
    contagem = {"n": 0}
    original = ModeloService.prever

    def espiao(self, pixels):
        contagem["n"] += 1
        return original(self, pixels)

    monkeypatch.setattr(ModeloService, "prever", espiao)
    return contagem


CASOS_INVALIDOS = {
    "pixel como texto numerico": com(pixels=["128"] * 784),
    "pixel como texto livre": com(pixels=["abc"] * 784),
    "pixel booleano": com(pixels=[True] * 784),
    "pixel nulo": com(pixels=[None] * 784),
    "783 pixels": com(pixels=[0] * 783),
    "785 pixels": com(pixels=[0] * 785),
    "lista vazia": com(pixels=[]),
    "pixels nao e lista": com(pixels="0" * 784),
    "pixel negativo": com(pixels=[-1] + [0] * 783),
    "pixel acima de 255": com(pixels=[256] + [0] * 783),
    "campo pixels ausente": {"top_k": 3},
    "campo extra": com(intruso="x"),
    "limiar zero (gt=0)": com(limiar_confianca=0),
    "limiar acima de 1": com(limiar_confianca=1.5),
    "limiar como texto": com(limiar_confianca="0.5"),
    "top_k zero": com(top_k=0),
    "top_k 11": com(top_k=11),
    "top_k como texto": com(top_k="3"),
    "top_k decimal": com(top_k=2.5),
    "identificador vazio": com(identificador=""),
    "identificador com 51 caracteres": com(identificador="a" * 51),
    "identificador numerico": com(identificador=123),
}


@pytest.mark.parametrize("corpo", CASOS_INVALIDOS.values(), ids=CASOS_INVALIDOS.keys())
def test_dado_invalido_retorna_422_e_nao_aciona_o_modelo(cliente, chamadas_ao_modelo, corpo):
    resposta = cliente.post(URL, json=corpo)

    assert resposta.status_code == 422
    assert chamadas_ao_modelo["n"] == 0  # o modelo nunca foi chamado


def test_corpo_vazio_e_json_malformado_retornam_422(cliente, chamadas_ao_modelo):
    assert cliente.post(URL).status_code == 422
    resposta = cliente.post(URL, content="{isso nao e json", headers={"Content-Type": "application/json"})
    assert resposta.status_code == 422
    assert chamadas_ao_modelo["n"] == 0


def test_nan_e_infinito_sao_rejeitados(cliente, chamadas_ao_modelo):
    for valor in ("NaN", "Infinity", "-Infinity"):
        corpo = '{"pixels": [' + valor + ", " + ", ".join(["0"] * 783) + "]}"
        resposta = cliente.post(URL, content=corpo, headers={"Content-Type": "application/json"})
        assert resposta.status_code == 422
    assert chamadas_ao_modelo["n"] == 0


def test_corpo_do_422_e_curto_estruturado_e_nao_repete_os_dados(cliente):
    resposta = cliente.post(URL, json=com(pixels=["abc"] * 784))
    dados = resposta.json()

    assert set(dados) == {"detalhe", "total_erros", "erros"}
    assert dados["total_erros"] == 784        # todos os pixels estavam errados...
    assert len(dados["erros"]) == 10          # ...mas só os 10 primeiros são listados
    assert dados["erros"][0]["campo"] == "pixels.0"
    assert "abc" not in resposta.text         # não ecoa o que foi enviado


def test_erro_indica_o_campo_e_o_tipo(cliente):
    dados = cliente.post(URL, json=com(top_k=11)).json()
    erro = dados["erros"][0]
    assert erro["campo"] == "top_k"
    assert erro["tipo"] == "less_than_equal"


def test_limites_exatos_sao_aceitos(cliente):
    corpo = com(pixels=[0] * 783 + [255], limiar_confianca=1, top_k=10, identificador="a" * 50)
    assert cliente.post(URL, json=corpo).status_code == 200
    assert cliente.post(URL, json=com(limiar_confianca=0.0001, top_k=1, identificador="x")).status_code == 200


def test_inteiros_sao_aceitos_como_pixels(cliente, chamadas_ao_modelo):
    assert cliente.post(URL, json=com(pixels=list(range(256)) * 3 + [0] * 16)).status_code == 200
    assert chamadas_ao_modelo["n"] == 1


def test_contrato_da_resposta_valida(cliente):
    resposta = cliente.post(URL, json=com(identificador="img_01.png", top_k=4, limiar_confianca=1))
    assert resposta.status_code == 200
    dados = resposta.json()

    assert set(dados) == {
        "identificador", "digito_previsto", "confianca", "confiavel",
        "melhores_previsoes", "probabilidades",
    }
    assert dados["identificador"] == "img_01.png"
    assert 0 <= dados["digito_previsto"] <= 9
    assert len(dados["probabilidades"]) == 10
    assert abs(sum(dados["probabilidades"]) - 1) < 1e-6
    assert len(dados["melhores_previsoes"]) == 4
    assert dados["melhores_previsoes"][0]["digito"] == dados["digito_previsto"]
    probs = [m["probabilidade"] for m in dados["melhores_previsoes"]]
    assert probs == sorted(probs, reverse=True)
    assert dados["confiavel"] is False   # limiar 1 só é atingido com 100% de confiança
    assert "pixels" not in dados


def test_valores_padrao_quando_campos_opcionais_sao_omitidos(cliente):
    dados = cliente.post(URL, json=VALIDO).json()
    assert dados["identificador"] is None
    assert len(dados["melhores_previsoes"]) == 3  # top_k padrão
