"""Tratamento centralizado de erros: toda falha vira uma resposta JSON com o status certo.

Mapa de status codes:
    400  JSON malformado (o corpo nem é um JSON válido)
    422  JSON válido, mas fora do contrato (tipo errado, faixa, tamanho...)
    404  rota inexistente        405  método HTTP não permitido
    503  modelo indisponível (não está na RAM)
    500  falha inesperada (o detalhe interno fica só no log, nunca vai ao usuário)
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErroCampo, ErroResponse, ErroValidacaoResponse
from app.services import ErroDeInferenciaError, ModeloNaoCarregadoError

logger = logging.getLogger("uvicorn.error")

MAX_ERROS = 10

MENSAGENS_HTTP = {
    404: "Rota não encontrada.",
    405: "Método HTTP não permitido para esta rota.",
}


def _erro(status: int, detalhe: str, id_erro: str | None = None, headers: dict | None = None):
    corpo = ErroResponse(detalhe=detalhe, status=status, id_erro=id_erro)
    return JSONResponse(status_code=status, content=corpo.model_dump(), headers=headers)


def _novo_id() -> str:
    return uuid.uuid4().hex[:8]


async def tratar_erro_validacao(request: Request, exc: RequestValidationError) -> JSONResponse:
    erros = exc.errors()

    # 400: o corpo nem chegou a ser um JSON (erro de sintaxe)
    if any(e["type"] == "json_invalid" for e in erros):
        return _erro(400, "JSON malformado: o corpo da requisição não é um JSON válido.")

    # 422: JSON válido, mas que viola o contrato de dados
    resumo = []
    for e in erros[:MAX_ERROS]:
        partes = [str(p) for p in e["loc"] if p != "body"]
        resumo.append(
            ErroCampo(campo=".".join(partes) or "corpo", mensagem=e["msg"], tipo=e["type"])
        )
    corpo = ErroValidacaoResponse(
        detalhe="Dados inválidos: a requisição não respeita o contrato da API.",
        total_erros=len(erros),
        erros=resumo,
    )
    return JSONResponse(status_code=422, content=corpo.model_dump())


async def tratar_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detalhe = MENSAGENS_HTTP.get(exc.status_code, str(exc.detail))
    return _erro(exc.status_code, detalhe, headers=getattr(exc, "headers", None))


async def tratar_modelo_indisponivel(request: Request, exc: ModeloNaoCarregadoError) -> JSONResponse:
    return _erro(503, "Modelo indisponível. Verifique o endpoint /health.")


async def tratar_erro_inferencia(request: Request, exc: ErroDeInferenciaError) -> JSONResponse:
    id_erro = _novo_id()
    logger.error("[%s] Falha na inferência", id_erro, exc_info=exc)
    return _erro(500, "Falha ao executar a inferência. Informe o id_erro ao suporte.", id_erro)


async def tratar_erro_inesperado(request: Request, exc: Exception) -> JSONResponse:
    id_erro = _novo_id()
    logger.error("[%s] Erro inesperado em %s %s", id_erro, request.method, request.url.path, exc_info=exc)
    return _erro(500, "Erro interno inesperado. Informe o id_erro ao suporte.", id_erro)


def registrar_tratadores(app: FastAPI) -> None:
    """Liga todos os tratadores à aplicação (chamado no main.py)."""
    app.add_exception_handler(RequestValidationError, tratar_erro_validacao)
    app.add_exception_handler(StarletteHTTPException, tratar_http_exception)
    app.add_exception_handler(ModeloNaoCarregadoError, tratar_modelo_indisponivel)
    app.add_exception_handler(ErroDeInferenciaError, tratar_erro_inferencia)
    app.add_exception_handler(Exception, tratar_erro_inesperado)
