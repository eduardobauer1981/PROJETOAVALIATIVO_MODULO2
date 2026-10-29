"""Ponto de entrada da API: ciclo de vida e inclusão dos roteadores."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import CAMINHO_MODELO
from app.routers import predicao, saude
from app.services import ModeloService

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Inicialização: carrega o modelo uma única vez ---
    service = ModeloService(CAMINHO_MODELO)
    try:
        service.carregar()
    except FileNotFoundError as erro:
        # A API sobe mesmo assim: /saude mostra "degradado" e /predicao responde 503
        logger.error("%s", erro)
    app.state.modelo_service = service

    yield

    # --- Encerramento: libera o modelo da memória ---
    service.descarregar()


app = FastAPI(
    title="API de Reconhecimento de Dígitos (MNIST)",
    description="Classifica dígitos manuscritos com uma rede neural MLP.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(saude.router)
app.include_router(predicao.router)
