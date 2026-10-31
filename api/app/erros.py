"""Tratamento do erro 422: resposta curta, em português e sem repetir os dados enviados."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import ErroCampo, ErroValidacaoResponse

MAX_ERROS = 10


async def tratar_erro_validacao(request: Request, exc: RequestValidationError) -> JSONResponse:
    erros = exc.errors()
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
