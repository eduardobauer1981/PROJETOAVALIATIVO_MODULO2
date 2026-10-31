"""Ponto de entrada da API: ciclo de vida, tratadores de erro e roteadores."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import CAMINHO_MODELO
from app.erros import registrar_tratadores
from app.routers import health, predicao
from app.services import ModeloService

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Inicialização: carrega o modelo na RAM uma única vez ---
    service = ModeloService(CAMINHO_MODELO)
    try:
        service.carregar()
    except FileNotFoundError as erro:
        logger.error("%s", erro)
    except Exception:
        # Arquivo corrompido, versão incompatível etc.: a API NÃO cai.
        logger.exception("Falha ao carregar o modelo; a API seguirá em modo degradado.")
    # Em qualquer falha: /health mostra 'degradado' e /v1/predict responde 503.
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

registrar_tratadores(app)

app.include_router(health.router)
app.include_router(predicao.router)
