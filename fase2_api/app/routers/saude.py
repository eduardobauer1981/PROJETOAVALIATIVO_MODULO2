"""Rota de verificação de saúde da aplicação."""

from fastapi import APIRouter, Request

from app.schemas import SaudeResponse

router = APIRouter(tags=["Saúde"])


@router.get("/saude", response_model=SaudeResponse)
def verificar_saude(request: Request) -> SaudeResponse:
    carregado = request.app.state.modelo_service.carregado
    return SaudeResponse(
        status="ok" if carregado else "degradado",
        modelo_carregado=carregado,
    )
