"""Rotas de negócio: predição de dígitos."""

from fastapi import APIRouter, Depends

from app.routers.dependencias import obter_modelo_service
from app.schemas import PredicaoRequest, PredicaoResponse
from app.services import ModeloService

router = APIRouter(prefix="/predicao", tags=["Predição"])


@router.post("/digito", response_model=PredicaoResponse)
def prever_digito(
    corpo: PredicaoRequest,
    service: ModeloService = Depends(obter_modelo_service),
) -> PredicaoResponse:
    """Recebe uma imagem 28x28 (784 pixels) e devolve o dígito previsto."""
    resultado = service.prever(corpo.pixels)
    return PredicaoResponse(
        digito_previsto=resultado.digito,
        confianca=resultado.confianca,
        probabilidades=resultado.probabilidades,
    )
