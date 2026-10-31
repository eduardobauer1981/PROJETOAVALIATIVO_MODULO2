"""Rota de monitoramento: checagem de integridade operacional."""

from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(tags=["Monitoramento"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Checagem de integridade (serviço e modelo na RAM)",
)
def health(request: Request) -> HealthResponse:
    """Sempre responde 200 enquanto a API estiver no ar.

    O campo `modelo_carregado` diz se o modelo está na memória RAM; se não estiver,
    `status` vem como 'degradado' e o endpoint de inferência responde 503.
    """
    service = getattr(request.app.state, "modelo_service", None)
    carregado = bool(service and service.carregado)
    return HealthResponse(
        status="ok" if carregado else "degradado",
        servico_ativo=True,
        modelo_carregado=carregado,
        modelo_tipo=service.tipo_modelo if service else None,
        mensagem=(
            "Serviço ativo e modelo carregado na memória."
            if carregado
            else "Serviço ativo, mas o modelo NÃO está carregado. Consulte os logs do servidor."
        ),
    )
