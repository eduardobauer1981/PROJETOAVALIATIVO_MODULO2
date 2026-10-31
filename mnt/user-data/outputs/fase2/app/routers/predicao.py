"""Rotas de negócio: predição de dígitos."""

from fastapi import APIRouter, Depends

from app.routers.dependencias import obter_modelo_service
from app.schemas import ErroValidacaoResponse, PredicaoRequest, PredicaoResponse, PrevisaoClasse
from app.services import ModeloService

router = APIRouter(prefix="/predicao", tags=["Predição"])


@router.post(
    "/digito",
    response_model=PredicaoResponse,
    responses={
        422: {
            "model": ErroValidacaoResponse,
            "description": "Dados fora do contrato (tipo errado, valor fora da faixa, tamanho inválido).",
        }
    },
)
def prever_digito(
    corpo: PredicaoRequest,
    service: ModeloService = Depends(obter_modelo_service),
) -> PredicaoResponse:
    """Recebe uma imagem 28x28 (784 pixels) e devolve o dígito previsto.

    Quando este código executa, os dados JÁ foram validados pelo Pydantic.
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
