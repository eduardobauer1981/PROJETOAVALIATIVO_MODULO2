"""Cliente HTTP da API (sem Streamlit): fala com /health e /v1/predict."""

import requests


class ApiIndisponivelError(Exception):
    """A API não respondeu (fora do ar, porta errada, timeout)."""


class ApiRespondeuErroError(Exception):
    """A API respondeu com um status de erro (400, 422, 500, 503...)."""

    def __init__(self, status: int, detalhe: str, id_erro: str | None = None, erros: list | None = None):
        super().__init__(detalhe)
        self.status = status
        self.detalhe = detalhe
        self.id_erro = id_erro
        self.erros = erros or []


def consultar_health(url: str) -> dict | None:
    """Devolve o JSON do /health, ou None se a API estiver fora do ar."""
    try:
        resposta = requests.get(f"{url.rstrip('/')}/health", timeout=3)
        return resposta.json()
    except (requests.RequestException, ValueError):
        return None


def prever(
    url: str,
    pixels: list[int],
    limiar_confianca: float = 0.5,
    top_k: int = 3,
    identificador: str | None = None,
) -> dict:
    corpo = {"pixels": pixels, "limiar_confianca": limiar_confianca, "top_k": top_k}
    if identificador:
        corpo["identificador"] = identificador[:50]

    try:
        resposta = requests.post(f"{url.rstrip('/')}/v1/predict", json=corpo, timeout=15)
    except requests.RequestException as erro:
        raise ApiIndisponivelError(str(erro)) from erro

    if resposta.status_code == 200:
        return resposta.json()

    try:
        dados = resposta.json()
    except ValueError:
        dados = {}
    raise ApiRespondeuErroError(
        status=resposta.status_code,
        detalhe=dados.get("detalhe") or resposta.text[:200] or "Erro sem detalhes.",
        id_erro=dados.get("id_erro"),
        erros=dados.get("erros"),
    )
