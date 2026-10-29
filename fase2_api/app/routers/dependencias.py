"""Dependências compartilhadas entre os roteadores."""

from fastapi import HTTPException, Request, status

from app.services import ModeloService


def obter_modelo_service(request: Request) -> ModeloService:
    """Entrega o serviço criado no ciclo de vida da aplicação."""
    service: ModeloService = request.app.state.modelo_service
    if not service.carregado:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se models/mlp_mnist.joblib existe.",
        )
    return service
