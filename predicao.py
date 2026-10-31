"""Rotas de negócio: inferência (predição de dígitos)."""

from fastapi import APIRouter, Depends

from app.routers.dependencias import obter_modelo_service
from app.schemas import (
    ErroResponse,
    ErroValidacaoResponse,
    PredicaoRequest,
    PredicaoResponse,
    PrevisaoClasse,
)
from app.services import ModeloService

router = APIRouter(prefix="/v1", tags=["Inferência"])


@router.post(
    "/predict",
    response_model=PredicaoResponse,
    summary="Classifica um dígito manuscrito (imagem 28x28)",
    responses={
        400: {"model": ErroResponse, "description": "JSON malformado (erro de sintaxe)."},
        422: {
            "model": ErroValidacaoResponse,
            "description": "JSON válido, mas fora do contrato (tipo, faixa ou tamanho inválido).",
        },
        500: {"model": ErroResponse, "description": "Falha interna na inferência."},
        503: {"model": ErroResponse, "description": "Modelo indisponível (não está na RAM)."},
    },
)
def predict(
    corpo: PredicaoRequest,
    service: ModeloService = Depends(obter_modelo_service),
) -> PredicaoResponse:
    """Recebe uma imagem 28x28 (784 pixels) e devolve o dígito previsto.

    Quando este código executa, os dados JÁ foram validados pelo Pydantic.
    Falhas do serviço sobem como exceções e são traduzidas em app/erros.py.
    """
    resultado = service.prever(corpo.pixels)
    return PredicaoResponse(
        identificador=corpo.identificador,
        digito_previsto=resultado.digito,
        confianca=resultado.confianca,
        confiavel=resultado.confianca >= corpo.limiar_confianca,
        melhores_previsoes=[
            PrevisaoClasse(digito=d, probabilidade=p) for d, p in resultado.melhores(corpo.top_k)
        ],
        probabilidades=resultado.probabilidades,
    )
